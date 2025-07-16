"""
RAG Agent Implementation - Restored from original myChatbot
This module implements the create_react_agent functionality with Azure AI Search integration
"""
import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import threading
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor

from langchain.chains import RetrievalQA
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema import Document as LCDocument
from langchain.callbacks.base import BaseCallbackHandler
import tiktoken

# Import shared components for consistency
from .utils import create_chat_openai_client, get_embedding_model
from .azure_search_backend import get_azure_search_backend

logger = logging.getLogger(__name__)

# Configuration
@dataclass
class AppConfig:
    # Rate limiting
    max_requests_per_minute: int = 60
    max_requests_per_hour: int = 1000
    
    # Connection pooling
    max_concurrent_requests: int = 10
    connection_pool_size: int = 20
    
    # Session management
    session_timeout_minutes: int = 30
    max_sessions: int = 1000
    
    # LLM settings
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.7
    max_tokens: int = 1000
    
    # Queue settings
    queue_timeout_seconds: int = 30
    max_queue_size: int = 100

config = AppConfig()

# Token counting callback
class AggregatingTokenTracker(BaseCallbackHandler):
    def __init__(self, model_name="gpt-35-turbo"):
        super().__init__()
        self.model_name = model_name
        self.reset()
    
    def reset(self):
        """Resets the counters for a new run/session."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0
    
    def on_llm_end(self, response, **kwargs):
        """
        Called after each LLM call; aggregates tokens + estimated cost.
        Uses token_usage if present, else calculates using tiktoken.
        """
        usage = response.llm_output.get("token_usage", {})
        # Prefer usage reported by the LLM (if reliable)
        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)
        total = usage.get("total_tokens", 0)
        
        # Fallback to local calculation if usage missing or suspiciously zero
        if total == 0 or prompt + completion == 0:
            prompt_text = kwargs.get("prompts", [""])[0]
            generations = response.generations[0]
            completion_texts = [gen.text for gen in generations]
            enc = None
            try:
                enc = tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                enc = tiktoken.get_encoding("cl100k_base")
            prompt = len(enc.encode(prompt_text))
            completion = sum(len(enc.encode(text)) for text in completion_texts)
            total = prompt + completion
        
        # Aggregate totals
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total
        
        # Compute cost - adjust pricing for your Azure region and agreement
        cost_per_prompt_token = 0.005 / 1000  # example: $0.005 per 1k prompt tokens
        cost_per_completion_token = 0.015 / 1000  # example: $0.015 per 1k completion tokens
        cost = (prompt * cost_per_prompt_token) + (completion * cost_per_completion_token)
        self.total_cost += cost
        
        print(
            f"[AggregatingTokenTracker] Iteration: Prompt={prompt}, "
            f"Completion={completion}, Total={total}, "
            f"Accumulated Cost=${self.total_cost:.4f}"
        )
    
    def get_totals(self):
        """Returns aggregated token usage and cost as a dictionary."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost,
        }

# Session Manager with updated memory handling
@dataclass
class SessionData:
    session_id: str
    chat_history: ChatMessageHistory
    agent_executor: Optional[AgentExecutor] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    request_count: int = 0

class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, SessionData] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task = None
    
    async def _create_session_internal(self, session_id: str) -> SessionData:
        """Internal method to create session - assumes lock is already held"""
        if len(self._sessions) >= config.max_sessions:
            # Remove oldest session
            oldest_session = min(
                self._sessions.values(),
                key=lambda s: s.last_accessed
            )
            del self._sessions[oldest_session.session_id]
            logger.info(f"Removed oldest session: {oldest_session.session_id}")
        
        # Creating new session with updated memory
        chat_history = ChatMessageHistory()
        
        session_data = SessionData(
            session_id=session_id,
            chat_history=chat_history
        )
        
        # Create agent executor with isolated memory
        session_data.agent_executor = self._create_agent_executor(chat_history)
        
        self._sessions[session_id] = session_data
        logger.info(f"Created new session: {session_id}")
        
        return session_data

    async def create_session(self, session_id: str) -> SessionData:
        """Public method to create a new session - acquires lock"""
        async with self._lock:
            return await self._create_session_internal(session_id)
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get existing session or create new one"""
        async with self._lock:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                session.last_accessed = datetime.now()
                return session
            else:
                return await self._create_session_internal(session_id)
    
    def _create_agent_executor(self, chat_history: ChatMessageHistory) -> AgentExecutor:
        """Create agent executor with tools and memory - ORIGINAL RAG FUNCTIONALITY"""
        try:
            llm_client = create_chat_openai_client()
            if not llm_client:
                raise Exception("LLM client not available")
                
            # LLM with connection pooling
            llm = llm_client
            
            def format_azure_search_results(docs):
                """Format Azure AI Search results with metadata"""
                formatted = []
                for doc in docs:
                    metadata = doc.metadata
                    content = doc.page_content.strip()
                    title = metadata.get("title", "Untitled Document")
                    url = metadata.get("page_url", "No URL")
                    search_score = metadata.get("search_score", "N/A")
                    document_id = metadata.get("document_id", "Unknown")
                    
                    formatted_content = f"""🔹 *Title*: {title}
*URL*: {url}
*Document ID*: {document_id}
*Search Score*: {search_score}
*Content*:
{content}
"""
                    formatted.append(LCDocument(page_content=formatted_content, metadata=metadata))
                return formatted

            def azure_ai_search_tool(query: str) -> str:
                """Azure AI Search tool function"""
                try:
                    vectorstore = get_azure_search_backend()
                    embedding_model = get_embedding_model()
                    
                    # Get query embedding
                    query_embedding = embedding_model.embed_query(query)
                    
                    # Perform hybrid search
                    results = vectorstore.hybrid_search(
                        query=query,
                        embedding=query_embedding,
                        top_k=5
                    )
                    
                    # Convert results to LangChain Documents
                    retrieved_docs = []
                    for result in results:
                        retrieved_docs.append(LCDocument(
                            page_content=result.get('content', ''),
                            metadata={
                                "title": result.get('title', ''),
                                "page_url": result.get('page_url', ''),
                                "document_id": result.get('document_id', ''),
                                "chunk_index": result.get('chunk_index', ''),
                                "search_score": result.get('@search.score', 0)
                            }
                        ))
                    
                    if not retrieved_docs:
                        return "No relevant documents found in Azure AI Search."
                    
                    # Format the results
                    formatted_docs = format_azure_search_results(retrieved_docs)
                    return "\n\n---\n\n".join([doc.page_content for doc in formatted_docs])
                    
                except Exception as e:
                    logger.error(f"Error in Azure AI Search tool: {str(e)}")
                    return f"Error performing Azure AI Search: {str(e)}"

            def azure_ai_filtered_search_tool(query_and_filters: str) -> str:
                """Azure AI Search tool with filtering capability"""
                try:
                    # Parse the input to extract query and potential filters
                    # Expected format: "query|document_id:value|title_filter:value"
                    parts = query_and_filters.split("|")
                    query = parts[0].strip()
                    
                    document_id = None
                    title_filter = None
                    
                    for part in parts[1:]:
                        if ":" in part:
                            key, value = part.split(":", 1)
                            key = key.strip()
                            value = value.strip()
                            
                            if key == "document_id":
                                document_id = value
                            elif key == "title_filter":
                                title_filter = value
                    
                    vectorstore = get_azure_search_backend()
                    embedding_model = get_embedding_model()
                    
                    # Build filter string
                    filters = []
                    if document_id:
                        filters.append(f"document_id eq '{document_id}'")
                    if title_filter:
                        filters.append(f"search.ismatch('{title_filter}', 'title')")
                    filter_str = " and ".join(filters) if filters else None
                    
                    # Get query embedding
                    query_embedding = embedding_model.embed_query(query)
                    
                    # Perform filtered search
                    results = vectorstore.hybrid_search(
                        query=query,
                        embedding=query_embedding,
                        filter=filter_str,
                        top_k=5
                    )
                    
                    # Convert results to LangChain Documents
                    retrieved_docs = []
                    for result in results:
                        retrieved_docs.append(LCDocument(
                            page_content=result.get('content', ''),
                            metadata={
                                "title": result.get('title', ''),
                                "page_url": result.get('page_url', ''),
                                "document_id": result.get('document_id', ''),
                                "chunk_index": result.get('chunk_index', ''),
                                "search_score": result.get('@search.score', 0)
                            }
                        ))
                    
                    if not retrieved_docs:
                        return f"No relevant documents found with filters: {filter_str}"
                    
                    # Format the results
                    formatted_docs = format_azure_search_results(retrieved_docs)
                    filter_info = f"Applied filters: {filter_str}\n\n" if filter_str else ""
                    return filter_info + "\n\n---\n\n".join([doc.page_content for doc in formatted_docs])
                    
                except Exception as e:
                    logger.error(f"Error in Azure AI filtered search tool: {str(e)}")
                    return f"Error performing Azure AI filtered search: {str(e)}"

            # Create the tools
            azure_search_tool = Tool(
                name="azure_ai_search",
                func=azure_ai_search_tool,
                description="Use this to search the Azure AI Search index for relevant Confluence documentation. Input should be a search query string. Returns relevant content with titles, URLs, and search scores."
            )
            
            azure_filtered_search_tool = Tool(
                name="azure_ai_filtered_search",
                func=azure_ai_filtered_search_tool,
                description="Use this to search Azure AI Search with filters. Input format: 'query|document_id:value|title_filter:value'. Example: 'ARB process|document_id:383516798' or 'governance|title_filter:architecture'. Returns filtered results."
            )

            # Define the tools list (prioritize Azure AI Search)
            tools = [
                azure_search_tool,
                azure_filtered_search_tool,
            ]

            # Original prompt template from main.py
            template = """You are an intelligent ARB(Architecture Review Board) assistant of GSK GSC(Global Supply Chain) that answers questions by always using internal Confluence documentation through Azure AI Search.
You can think step-by-step, decide how to use tools, and always explain your reasoning.

---
Available Tools:
You have access to these search tools (use in order of preference):
{tools}

• **azure_ai_search**: Primary search tool using Azure AI Search for Confluence documentation. Provides hybrid search with relevance scores.
• **azure_ai_filtered_search**: Use when you need to filter results by document ID or title. Input format: 'query|document_id:value|title_filter:value'

---
Tool Usage Guidelines:
1. Always start with azure_ai_search for general queries
2. Use azure_ai_filtered_search when you need specific documents or want to filter by title
3. For follow-up questions in the same conversation, consider using filtered search if you know the relevant document ID

---
When a user submits a query:

1. Carefully interpret the request — especially for ARB processes, artefacts, governance workflows, lifecycle stages, or review board interactions.
2. Never respond with any generic information.
3. If query seems unclear then politely ask user to give more keywords or details regarding query.
4. If query seems harmful or irrelevant then politely deny to answer.
5. If user greets you then greet back politely telling about yourself.

---
Your Process:
Use the following format exactly when answering questions:

Question: the input question you must answer
Thought: what you contextually think about the question
Action: the tool to use [{tool_names}]
Action Input: the input to the tool and the chat history containing previous answer if any exists
Observation: the result of the action/tool
... (you can repeat Thought → Action → Action Input → Observation if needed and also ensure to logically reason the result with available contextual information to finetune it and conclude in every iteration)
Thought: I now know the answer
Final Answer: the answer to the original question, including proper citation(s).

Always try to articulate the relevant information in detail and include it in final answer to give it as background.
Always ensure to provide any resource URL(if available) in final answer.
Ensure to provide the source document information(if available) in final answer.
Include search scores when available to indicate relevance.
If any tabular data is available then include it in tabular format within final answer.
If any artifact is available which is relevant then include it in final answer with proper formatting & appropriate dimensions(especially if images are available).

Always cite your sources in this format (if available):
Title: <document title>  
URL: <document page_url>
Document ID: <document_id>
Search Score: <search_score>

If you can't find a good answer in the documentation, say:
"I couldn't find relevant information in the Azure AI Search index."

---
Begin!
Previous conversation:
{chat_history}

Question: {input}
{agent_scratchpad}"""
            
            prompt = PromptTemplate.from_template(template)

            # Create the react agent - THIS IS THE MISSING FUNCTIONALITY
            agent = create_react_agent(llm, tools, prompt)
            
            # Create a function to get chat history
            def get_session_history(session_id: str) -> BaseChatMessageHistory:
                return chat_history
            
            # Create agent executor with proper memory
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=15,
                max_execution_time=120,
                return_intermediate_steps=False,
            )
            
            # Wrap with message history
            agent_with_chat_history = RunnableWithMessageHistory(
                agent_executor,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
            )
            
            return agent_with_chat_history
            
        except Exception as e:
            logger.error(f"Error creating agent executor: {e}")
            raise

    async def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        async with self._lock:
            now = datetime.now()
            expired_sessions = [
                session_id for session_id, session in self._sessions.items()
                if now - session.last_accessed > timedelta(minutes=config.session_timeout_minutes)
            ]
            
            for session_id in expired_sessions:
                del self._sessions[session_id]
                logger.info(f"Removed expired session: {session_id}")
    
    @property
    def active_sessions_count(self) -> int:
        return len(self._sessions)

# Global session manager
rag_session_manager = SessionManager()

async def process_rag_chat_request(session_id: str, message: str, user_id: int, session_context: List[Dict] = None) -> Dict[str, Any]:
    """
    Process a chat request using the original RAG functionality with create_react_agent
    This restores the missing functionality from the original myChatbot
    """
    try:
        print(f"RAG Agent: Processing request for session {session_id}")
        print(f"RAG Agent: Session context length: {len(session_context) if session_context else 0}")
        
        # Get session (creates if doesn't exist)
        session = await rag_session_manager.get_session(session_id)
        session.request_count += 1
        
        # CRITICAL FIX: Always update session's chat history with provided context
        if session_context:
            # Clear existing history and rebuild from context
            session.chat_history.clear()
            print(f"RAG Agent: Loading {len(session_context)} context messages")
            
            for ctx in session_context:
                if ctx['role'] == 'user':
                    session.chat_history.add_user_message(ctx['content'])
                elif ctx['role'] == 'assistant':
                    session.chat_history.add_ai_message(ctx['content'])
            
            print(f"RAG Agent: Chat history now has {len(session.chat_history.messages)} messages")
       
        # Process with agent
        logger.info(f"Processing RAG message for session {session_id}: {message[:50]}...")
       
        token_tracker = AggregatingTokenTracker("gpt-4o")
       
        # Add timeout to prevent hanging requests
        try:
            result = await asyncio.wait_for(
                session.agent_executor.ainvoke(
                    {"input": message},
                    config={"callbacks": [token_tracker], "configurable": {"session_id": session_id}}
                ),
                timeout=120  # 2 minute timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"RAG request timeout for session {session_id}")
            raise Exception("Request timeout - please try again with a shorter question")
       
        total = token_tracker.get_totals()
        logger.info(f"RAG Token usage per session {session_id}: Total={total}")
        
        print(f"RAG Agent: Successfully processed request for session {session_id}")
       
        return {
            "response": result["output"],
            "session_id": session_id,
            "token_usage": total,
            "timestamp": datetime.now()
        }
       
    except Exception as e:
        logger.error(f"Error processing RAG request for session {session_id}: {e}")
        print(f"RAG Agent Error: {e}")
        raise

# Function to create callbacks list (from original)
def create_callbacks() -> List[Any]:
    """Create callbacks list for LangChain operations"""
    callbacks = []
    
    # Add Azure App Insights callback if available
    try:
        from app_insights_callback import AppInsightsHandler
        app_insights_callback = AppInsightsHandler()
        callbacks.append(app_insights_callback)
        logger.info("AppInsights callback added successfully")
    except Exception as e:
        logger.warning(f"AppInsights callback not available: {e}")
    
    return callbacks