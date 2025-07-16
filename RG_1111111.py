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
        try:
            # Safe access to llm_output
            llm_output = getattr(response, 'llm_output', None) or {}
            usage = llm_output.get("token_usage", {}) if llm_output else {}
            
            # Prefer usage reported by the LLM (if reliable)
            prompt = usage.get("prompt_tokens", 0)
            completion = usage.get("completion_tokens", 0)
            total = usage.get("total_tokens", 0)
            
            # Fallback to local calculation if usage missing or suspiciously zero
            if total == 0 or prompt + completion == 0:
                try:
                    prompt_text = kwargs.get("prompts", [""])[0]
                    generations = getattr(response, 'generations', [[]])[0]
                    completion_texts = [getattr(gen, 'text', '') for gen in generations]
                    
                    enc = None
                    try:
                        enc = tiktoken.encoding_for_model(self.model_name)
                    except KeyError:
                        enc = tiktoken.get_encoding("cl100k_base")
                    
                    prompt = len(enc.encode(str(prompt_text)))
                    completion = sum(len(enc.encode(str(text))) for text in completion_texts)
                    total = prompt + completion
                except Exception as e:
                    print(f"[AggregatingTokenTracker] Error calculating tokens: {e}")
                    prompt = completion = total = 0
            
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
        except Exception as e:
            print(f"[AggregatingTokenTracker] Error in on_llm_end: {e}")
    
    def get_totals(self):
        """Returns aggregated token usage and cost as a dictionary."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost,
        }

# Agent executor creation function - moved from SessionManager
def create_agent_executor(chat_history: ChatMessageHistory) -> AgentExecutor:
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

            # Create the tools with error handling wrappers
            def safe_azure_search(query: str) -> str:
                """Safe wrapper for Azure AI Search tool"""
                try:
                    if not query or not query.strip():
                        return "Error: Empty search query provided. Please provide a valid search term."
                    return azure_ai_search_tool(query.strip())
                except Exception as e:
                    logger.error(f"Azure AI Search tool error: {e}")
                    return f"Azure AI Search encountered an error: {str(e)}. Please try rephrasing your query."

            def safe_azure_filtered_search(query_and_filters: str) -> str:
                """Safe wrapper for Azure AI filtered search tool"""
                try:
                    if not query_and_filters or not query_and_filters.strip():
                        return "Error: Empty search query provided. Please provide a valid search term with optional filters."
                    return azure_ai_filtered_search_tool(query_and_filters.strip())
                except Exception as e:
                    logger.error(f"Azure AI filtered search tool error: {e}")
                    return f"Azure AI filtered search encountered an error: {str(e)}. Please try rephrasing your query."

            azure_search_tool = Tool(
                name="azure_ai_search",
                func=safe_azure_search,
                description="Use this to search the Azure AI Search index for relevant Confluence documentation. Input should be a search query string. Returns relevant content with titles, URLs, and search scores."
            )
            
            azure_filtered_search_tool = Tool(
                name="azure_ai_filtered_search",
                func=safe_azure_filtered_search,
                description="Use this to search Azure AI Search with filters. Input format: 'query|document_id:value|title_filter:value'. Example: 'ARB process|document_id:383516798' or 'governance|title_filter:architecture'. Returns filtered results."
            )

            # Define the tools list (prioritize Azure AI Search)
            tools = [
                azure_search_tool,
                azure_filtered_search_tool,
            ]
            
            # Debug: Print available tool names
            tool_names = [tool.name for tool in tools]
            print(f"[DEBUG] Available tools: {tool_names}")

            # Original prompt template from main.py - RESTORED
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
            print(f"[DEBUG] Creating agent with tools: {[t.name for t in tools]}")
            agent = create_react_agent(llm, tools, prompt)
            print(f"[DEBUG] Agent created successfully")
            
            # Create a function to get chat history
            def get_session_history(session_id: str) -> BaseChatMessageHistory:
                return chat_history
            
            # Enhanced error handler that provides proper responses
            def handle_parsing_error(error) -> str:
                """Enhanced handler that always provides a proper response"""
                error_msg = str(error)
                logger.warning(f"Agent parsing error: {error_msg}")
                
                # Extract the original input from the error context if possible
                try:
                    # Try to get the current input from the agent's context
                    current_input = kwargs.get('input', '') if 'kwargs' in locals() else ''
                except:
                    current_input = ''
                
                # Provide contextual responses based on error type
                if "could not parse llm output" in error_msg.lower():
                    if any(greeting in current_input.lower() for greeting in ['hello', 'hi', 'hey', 'greetings']):
                        return """Hello! I am an intelligent ARB (Architecture Review Board) assistant for GSK GSC (Global Supply Chain). I specialize in helping with ARB processes, artifacts, governance workflows, lifecycle stages, and review board interactions using our internal Confluence documentation. How can I assist you today?"""
                    else:
                        # For non-greeting queries, try to provide a helpful response
                        return f"""I understand you're asking about: "{current_input}". Let me help you with that.

I am an ARB (Architecture Review Board) assistant for GSK GSC (Global Supply Chain). I can search our internal Confluence documentation to provide information about ARB processes, artifacts, governance workflows, lifecycle stages, and review board interactions.

To get the most accurate information, could you please provide more specific details about what aspect of ARB processes you'd like to know about?"""
                
                elif "not a valid tool" in error_msg.lower():
                    return """I encountered a technical issue while searching for information. However, I can still help you!

I am an ARB (Architecture Review Board) assistant for GSK GSC (Global Supply Chain). I have access to internal Confluence documentation about ARB processes, artifacts, governance workflows, lifecycle stages, and review board interactions.

Please rephrase your question and I'll do my best to provide you with relevant information."""
                
                elif "iteration" in error_msg.lower() or "time limit" in error_msg.lower():
                    return """I apologize for the delay in processing your request. Let me provide you with some general guidance:

As an ARB (Architecture Review Board) assistant for GSK GSC (Global Supply Chain), I can help with:
- ARB processes and procedures
- Architecture artifacts and documentation
- Governance workflows and approvals
- Lifecycle stages and review requirements
- Review board interactions and submissions

Please try asking a more specific question about any of these topics, and I'll provide detailed information from our Confluence documentation."""
                
                else:
                    return f"""I encountered a technical issue but I'm here to help!

I am an intelligent ARB (Architecture Review Board) assistant for GSK GSC (Global Supply Chain). I can provide information about ARB processes, artifacts, governance workflows, lifecycle stages, and review board interactions using our internal Confluence documentation.

Please rephrase your question or ask about any specific ARB-related topic you need assistance with."""

            # Create agent executor with better error handling and reasonable limits
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=handle_parsing_error,
                max_iterations=10,  # Restored to reasonable level with better error handling
                max_execution_time=90,  # Increased timeout with fallback mechanisms
                return_intermediate_steps=False,
                early_stopping_method="generate",  # Stop early on errors
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

# Removed duplicate session management - using app.py ChatbotSessionManager instead

async def process_rag_chat_request(session_id: str, message: str, user_id: int, session_context: List[Dict] = None) -> Dict[str, Any]:
    """
    Process a chat request using RAG functionality - now stateless, relies on app.py session management
    """
    try:
        print(f"RAG Agent: Processing request for session {session_id}")
        print(f"RAG Agent: Session context length: {len(session_context) if session_context else 0}")
        
        # Create chat history from provided context
        chat_history = ChatMessageHistory()
        if session_context:
            print(f"RAG Agent: Loading {len(session_context)} context messages")
            for ctx in session_context:
                if ctx['role'] == 'user':
                    chat_history.add_user_message(ctx['content'])
                elif ctx['role'] == 'assistant':
                    chat_history.add_ai_message(ctx['content'])
            print(f"RAG Agent: Chat history now has {len(chat_history.messages)} messages")
        
        # Create agent executor for this request
        agent_executor = create_agent_executor(chat_history)
        
        # Process with agent
        logger.info(f"Processing RAG message for session {session_id}: {message[:50]}...")
        
        token_tracker = AggregatingTokenTracker("gpt-4o")
        
        # Add timeout to prevent hanging requests with better error handling
        try:
            result = await asyncio.wait_for(
                agent_executor.ainvoke(
                    {"input": message},
                    config={"callbacks": [token_tracker], "configurable": {"session_id": session_id}}
                ),
                timeout=60  # 1 minute timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"RAG request timeout for session {session_id}")
            # Provide a fallback response instead of raising an exception
            result = {
                "output": "I apologize, but your request is taking longer than expected to process. This might be due to a complex query or system load. Please try rephrasing your question or breaking it into smaller, more specific parts."
            }
        except Exception as e:
            logger.error(f"RAG agent execution error for session {session_id}: {e}")
            
            # Enhanced fallback mechanism - try to provide actual answers
            try:
                # Try to use the search tools directly as fallback
                if any(greeting in message.lower() for greeting in ['hello', 'hi', 'hey', 'greetings']):
                    fallback_response = """Hello! I am an intelligent ARB (Architecture Review Board) assistant for GSK GSC (Global Supply Chain). I specialize in helping with ARB processes, artifacts, governance workflows, lifecycle stages, and review board interactions using our internal Confluence documentation.

I can help you with:
- ARB processes and procedures
- Architecture artifacts and documentation
- Governance workflows and approvals
- Lifecycle stages and review requirements
- Review board interactions and submissions

How can I assist you today?"""
                else:
                    # Try to search directly using the tools
                    try:
                        vectorstore = get_azure_search_backend()
                        embedding_model = get_embedding_model()
                        query_embedding = embedding_model.embed_query(message)
                        
                        # Perform direct search
                        results = vectorstore.hybrid_search(
                            query=message,
                            embedding=query_embedding,
                            top_k=3
                        )
                        
                        if results:
                            # Format results
                            formatted_results = []
                            for result in results:
                                title = result.get('title', 'Untitled Document')
                                content = result.get('content', '')[:500] + "..." if len(result.get('content', '')) > 500 else result.get('content', '')
                                url = result.get('page_url', 'No URL')
                                score = result.get('@search.score', 'N/A')
                                
                                formatted_results.append(f"""**{title}**
Content: {content}
URL: {url}
Search Score: {score}""")
                            
                            fallback_response = f"""Based on your query about "{message}", I found the following information from our ARB documentation:

{chr(10).join(formatted_results)}

This information is from our internal Confluence documentation. If you need more specific details, please refine your question."""
                        else:
                            fallback_response = f"""I understand you're asking about "{message}". While I encountered a technical issue with the advanced search, I can still help you with ARB-related topics.

As an ARB (Architecture Review Board) assistant for GSK GSC (Global Supply Chain), I can provide information about:
- ARB processes and procedures
- Architecture artifacts and documentation
- Governance workflows and approvals
- Lifecycle stages and review requirements
- Review board interactions and submissions

Please try rephrasing your question or ask about a specific ARB topic."""
                    except Exception as search_error:
                        logger.error(f"Fallback search also failed: {search_error}")
                        fallback_response = f"""I understand you're asking about "{message}". While I'm experiencing some technical difficulties, I'm still here to help with ARB-related questions.

As an ARB (Architecture Review Board) assistant for GSK GSC (Global Supply Chain), I can provide guidance on ARB processes, artifacts, governance workflows, lifecycle stages, and review board interactions.

Please try asking your question in a different way, and I'll do my best to assist you."""
                
                result = {"output": fallback_response}
                
            except Exception as fallback_error:
                logger.error(f"Fallback mechanism failed: {fallback_error}")
                # Final fallback
                result = {
                    "output": f"""I apologize for the technical difficulties. I am an ARB (Architecture Review Board) assistant for GSK GSC (Global Supply Chain), and I'm here to help with ARB processes, artifacts, governance workflows, lifecycle stages, and review board interactions.

Please try rephrasing your question about "{message}" and I'll do my best to provide you with relevant information from our documentation."""
                }
        
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