from contextlib import asynccontextmanager
import traceback
from typing import Optional, List, Dict, Any
from server import get_user_details, get_user_from_auth, review_document
import json
from werkzeug.datastructures import FileStorage
from src.blob_util import read_doc_from_blob
import os
import asyncio
from datetime import datetime, timedelta
from src.custom_excemptions import SQLConnectionError, UnAuthorizedError
from src.db_acccess import (get_user, create_feedback, get_user_task, get_task_result, get_task_summary, get_feedback, create_tables)
from src.models.design_task import Tasks
from dotenv import load_dotenv
from azure.identity import ClientSecretCredential
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request, Header, Response, status
from src.sharepoint_util import read_criteria
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from sqlalchemy.orm import Session
from pydantic import BaseModel
import uuid
from src.authentication import BearerAuth
# from src.logging import LOGGER
import sys

sys.path.append('myChatbot')

from myChatbot.models import get_db as get_chatbot_db, User as ChatbotUser, ChatHistory, Feedback as ChatbotFeedback
from myChatbot.auth import verify_microsoft_sso_token, verify_token, create_access_token
from myChatbot.db_utils import (save_chat_history, get_session_chat_history, save_feedback as save_chatbot_feedback,get_user_stats, get_system_stats, create_user as create_chatbot_user, get_user_by_email)
#Using shared components

from src.shared_components import (get_shared_chat_client, get_shared_embedding_model)
from myChatbot.config import config as chatbot_config

BLOB_ACCOUNT_URL = os.getenv("BLOB_ACCOUNT_URL")
TENANT_ID= os.getenv("TENANT_ID")
CLIENT_ID= os.getenv("CLIENT_ID")
CLIENT_SECRET= os.getenv("CLIENT_SECRET")

@asynccontextmanager
async def lifespan(app:FastAPI):
    create_tables()
    yield

#Session management for chatbot
class ChatbotSessionManager:
    """Manage chatbot sessions with cleanup and proper context handling"""
    def __init__(self):
        self.sessions = {}
        self.cleanup_interval = 3600  #1hr

    def create_session(self, user_id: int, session_id: str = None) -> str:
        """Create session with provided session_id or generate new one"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        self.sessions[session_id] = {
            'user_id': user_id,
            'created_at': datetime.utcnow(),
            'last_activity': datetime.utcnow(),
            'message_count': 0,
            'context': []
        }
        print(f"Created session {session_id} for user {user_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        return self.sessions.get(session_id)
    
    def get_or_create_session(self, session_id: str, user_id: int) -> Dict:
        """Get existing session or create new one with provided session_id"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            # Verify user ownership
            if session['user_id'] == user_id:
                return session
            else:
                print(f"Session {session_id} belongs to different user, creating new session")
        
        # Create new session with provided session_id
        self.create_session(user_id, session_id)
        return self.sessions[session_id]
    
    def update_session_activity(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id]['last_activity'] = datetime.utcnow()
            self.sessions[session_id]['message_count'] += 1

    def get_session_context(self, session_id: str) -> List[Dict]:
        """Get session context for RAG agent"""
        session = self.get_session(session_id)
        if session:
            return session.get('context', [])
        return []
    
    def update_session_context(self, session_id: str, user_message: str, assistant_response: str, timestamp: datetime):
        """Update session context with new messages"""
        if session_id in self.sessions:
            context = self.sessions[session_id]['context']
            context.append({
                'role': 'user',
                'content': user_message,
                'timestamp': timestamp
            })
            context.append({
                'role': 'assistant',
                'content': assistant_response,
                'timestamp': timestamp
            })
            # Keep only last 20 messages to prevent memory issues
            if len(context) > 20:
                self.sessions[session_id]['context'] = context[-20:]

    def cleanup_expired_sessions(self):
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.cleanup_interval)
        expired_sessions = [
            sid for sid, session in self.sessions.items()
            if session['last_activity'] < cutoff_time
        ]
        for session_id in expired_sessions:
            del self.sessions[session_id]
        if expired_sessions:
            print(f"Cleaned up {len(expired_sessions)} expired chatbot sessions")

# App configuration
APP_NAME = "GSC Design Review Bot with Chatbot"
APP_VERSION = "1.0.0"
APP_HOST = "0.0.0.0" #later az app domain
APP_PORT = "8000"
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# Global managers
chatbot_session_manager = ChatbotSessionManager()

# Background tasks management
background_tasks_active = False
cleanup_task = None

async def session_cleanup_task():
    global background_tasks_active
    while background_tasks_active:
        try:
            chatbot_session_manager.cleanup_expired_sessions()
            await asyncio.sleep(300)  # Run every 5 minutes
        except Exception as e:
            print(f"Error in session cleanup task: {e}")
            await asyncio.sleep(60)

async def start_background_tasks():
    global background_tasks_active, cleanup_task
    background_tasks_active = True
    cleanup_task = asyncio.create_task(session_cleanup_task())
    print("Background tasks started")

async def cleanup_background_tasks():
    global background_tasks_active, cleanup_task
    background_tasks_active = False
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
    print("Background tasks cleaned up")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting Enhanced Design Review Bot with Chatbot...")
    await start_background_tasks()
    print("Application startup complete")
    yield
    # Shutdown
    print("Shutting down application...")
    await cleanup_background_tasks()
    print("Application shutdown complete")

    
# Create FastAPI app
app = FastAPI(
    title=APP_NAME,
    description="GSC Design Review Bot+GSC Chatbot",
    version=APP_VERSION,
    lifespan=lifespan
)

# CORS middleware (simple configuration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Re-Configure
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class UserCreate(BaseModel):
    email_id: str
    username: str

class TaskCreate(BaseModel):
    app_ciid: str
    doc_name: str

class FeedbackCreate(BaseModel):
    task_id: int
    section: str
    sr_no: int
    summary: str
    positive_feedback: bool

class DesignReviewRequest(BaseModel):
    app_ciid: str
    design_doc: str
    user_email: str


# Chatbot models (SSO-based, no password required)
class ChatbotUserSSO(BaseModel):
    email_id: str
    username: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    response_id: str
    timestamp: datetime

class ChatbotFeedbackRequest(BaseModel):
    response_id: str
    rating: Optional[int] = None
    is_helpful: Optional[bool] = None
    feedback_text: Optional[str] = None
    feedback_category: Optional[str] = None


# Rate limiting for chatbot
class RateLimiter:
    def __init__(self):
        self.requests = {}
        self.window = 3600  # 1 hour
        self.max_requests = 100

    def is_allowed(self, client_id: str) -> bool:
        now = datetime.utcnow()
        if client_id not in self.requests:
            self.requests[client_id] = []
        # Clean old requests
        cutoff = now - timedelta(seconds=self.window)
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > cutoff
        ]
        # Check limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        # Add current request
        self.requests[client_id].append(now)
        return True

rate_limiter = RateLimiter()


# SSO Authentication dependency
async def get_current_sso_user(token: str = Depends(OAuth2PasswordBearer(tokenUrl="sso/token"))):
    """Get current user from Microsoft SSO token"""
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # Verify SSO token using auth module
        user_data = verify_microsoft_sso_token(token)
        if not user_data:
            raise credentials_exception
        # Get or create user in database
        db_session = next(get_chatbot_db())
        try:
            existing_user = get_user_by_email(db_session, user_data["email_id"])
            if not existing_user:
                # Create new user
                new_user = create_chatbot_user(
                    db_session,
                    user_data["email_id"],
                    user_data["username"] or user_data["email_id"].split('@')[0]
                )
                user_data["user_id"] = new_user.id
            else:
                user_data["user_id"] = existing_user.id
        finally:
            db_session.close()
        return user_data
    except Exception as e:
        print(f"SSO authentication failed: {e}")
        raise credentials_exception

async def get_current_active_sso_user(current_user: dict = Depends(get_current_sso_user)):
    """Get current active SSO user"""
    return current_user

#globalexception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

#Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "design_review": "active",
            "chatbot": "active",
            "database": "connected"
        },
        "active_sessions": len(chatbot_session_manager.sessions)
    }

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "GSC Chatbot",
        "services": {
            "design_review": "Design review functionality",
            "GSC chatbot": "Advanced conversational AI with analytics",
            "admin": "System administration and monitoring"
        },
        "docs": "/docs",
        "health": "/health"
    }


# SSO ENDPOINTS
@app.post("/sso/validate")
async def validate_sso_token_and_create_user(
    current_user: dict = Depends(get_current_active_sso_user)
):
    """Validate SSO token and create/get user"""
    try:
        return {
            "message": "SSO token validated successfully",
            "user": current_user,
            "authentication_type": "microsoft_sso"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating SSO token: {str(e)}")
    

@app.post("/sso/token")
async def get_sso_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """OAuth2 compatible token endpoint for SSO"""
    try:
        # In SSO flow, the token comes from Microsoft
        # This endpoint is for OAuth2 compatibility
        return {
            "access_token": form_data.username,  # The SSO token itself
            "token_type": "bearer",
            "authentication_type": "microsoft_sso"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing SSO token: {str(e)}")
    

@app.get("/sso/validate-token")
async def validate_current_sso_token(
    current_user: dict = Depends(get_current_active_sso_user)
):
    """Validate current SSO token and return user information"""
    return {
        "message": "SSO token validated successfully",
        "user": current_user,
        "authentication_type": "microsoft_sso"
    }

@app.post("/sessions")
async def create_new_session(
    current_user: dict = Depends(get_current_active_sso_user)
):
    """Create a new chat session for the user"""
    try:
        session_id = str(uuid.uuid4())
        chatbot_session_manager.create_session(current_user['user_id'], session_id)
        
        return {
            "session_id": session_id,
            "message": "New session created successfully",
            "user_id": current_user['user_id']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")




@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(
    request: ChatRequest,
    current_user: dict = Depends(get_current_active_sso_user),
    http_request: Request = None
):
    """Main chat endpoint with RAG functionality and SSO authentication"""
    try:
        # Rate limiting using SSO user ID
        client_id = f"sso_user_{current_user['user_id']}"
        if not rate_limiter.is_allowed(client_id):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        # CRITICAL FIX: Use session_id from frontend, don't create new ones
        session_id = request.session_id
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        
        print(f"Processing chat for session_id: {session_id}, user_id: {current_user['user_id']}")
        
        # Get or create session with the provided session_id
        session = chatbot_session_manager.get_or_create_session(session_id, current_user['user_id'])
        
        # Update session activity
        chatbot_session_manager.update_session_activity(session_id)

        # Load existing chat history for this session into memory context
        try:
            db_session = next(get_chatbot_db())
            existing_history = get_session_chat_history(db_session, session_id)
            
            # Update session context with existing chat history if not already loaded
            if existing_history and len(session.get('context', [])) == 0:
                session['context'] = []
                for chat in existing_history:
                    session['context'].append({
                        'role': 'user',
                        'content': chat.question,
                        'timestamp': chat.timestamp
                    })
                    session['context'].append({
                        'role': 'assistant',
                        'content': chat.answer,
                        'timestamp': chat.timestamp
                    })
                print(f"Loaded {len(existing_history)} chat history records for session {session_id}")
            db_session.close()
        except Exception as e:
            print(f"Error loading chat history: {e}")

        # Generate AI response using RAG functionality with session context
        try:
            from myChatbot.rag_agent import process_rag_chat_request
            rag_result = await process_rag_chat_request(
                session_id=session_id,
                message=request.message,
                user_id=current_user['user_id'],
                session_context=session.get('context', [])
            )
            response_text = rag_result["response"]
            response_id = str(uuid.uuid4())
            token_usage = rag_result.get("token_usage", {})

            # Save to database with correct session_id
            try:
                db_session = next(get_chatbot_db())
                chat_record = save_chat_history(
                    db=db_session,
                    session_id=session_id,  # Use the same session_id from frontend
                    user_id=current_user['user_id'],
                    question=request.message,
                    answer=response_text,
                    response_id=response_id
                )
                timestamp = chat_record.timestamp
                
                # Update session context in memory using the manager method
                chatbot_session_manager.update_session_context(
                    session_id, request.message, response_text, timestamp
                )
                
                db_session.close()
                print(f"Saved chat to database with session_id: {session_id}")
            except Exception as e:
                print(f"Error saving chat history: {e}")
                timestamp = datetime.utcnow()

        except Exception as e:
            print(f"Error generating RAG response: {e}")
            response_text = "I apologize, but I'm having trouble accessing the knowledge base right now. Please try again."
            response_id = str(uuid.uuid4())
            timestamp = datetime.utcnow()
            token_usage = {}

        # Enhanced response with token usage and session info
        chat_response = ChatResponse(
            response=response_text,
            session_id=session_id,  # Return the same session_id
            response_id=response_id,
            timestamp=timestamp
        )
        response_dict = chat_response.dict()
        response_dict["authentication_type"] = "microsoft_sso"
        response_dict["user_email"] = current_user["email_id"]
        response_dict["chat_history_id"] = getattr(chat_record, 'id', None) if 'chat_record' in locals() else None
        
        # Add token usage information
        if token_usage:
            response_dict.update({
                "prompt_tokens": token_usage.get("prompt_tokens", 0),
                "completion_tokens": token_usage.get("completion_tokens", 0),
                "total_tokens": token_usage.get("total_tokens", 0),
                "total_cost": token_usage.get("total_cost_usd", 0.0)
            })
        
        return response_dict

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")
   

@app.post("/chatbot/feedback")
async def submit_chatbot_feedback(
    feedback: ChatbotFeedbackRequest,
    current_user = Depends(get_current_active_sso_user),
    db: Session = Depends(get_chatbot_db)
):
    #Submit feedback for a chat response
    try:
        # Find the chat history record by response_id
        chat_record = db.query(ChatHistory).filter(
            ChatHistory.response_id == feedback.response_id,
            ChatHistory.user_id == current_user['user_id']
        ).first()
       
        if not chat_record:
            raise HTTPException(status_code=404, detail="Chat response not found")
       
        # Save feedback
        feedback_record = save_chatbot_feedback(
            db=db,
            chat_history_id=chat_record.id,
            user_id=current_user['user_id'],
            session_id=chat_record.session_id,
            rating=feedback.rating,
            is_helpful=feedback.is_helpful,
            feedback_text=feedback.feedback_text,
            feedback_category=feedback.feedback_category
        )
       
        return {
            "message": "Feedback submitted successfully",
            "feedback_id": feedback_record.id
        }
       
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting feedback: {str(e)}")

@app.get("/chatbot/feedback/stats")
async def get_chatbot_feedback_stats(
    current_user = Depends(get_current_active_sso_user),
    db: Session = Depends(get_chatbot_db)
):
    #Get feedback statistics for chatbot
    try:
        # Get total feedback count
        total_feedback = db.query(ChatbotFeedback).count()
       
        # Get helpful feedback count
        helpful_feedback = db.query(ChatbotFeedback).filter(
            ChatbotFeedback.is_helpful == True
        ).count()
       
        # Calculate helpfulness rate
        helpfulness_rate = (helpful_feedback / total_feedback * 100) if total_feedback > 0 else 0
       
        # Get average rating
        from sqlalchemy import func
        avg_rating_result = db.query(func.avg(ChatbotFeedback.rating)).filter(
            ChatbotFeedback.rating.isnot(None)
        ).scalar()
        average_rating = float(avg_rating_result) if avg_rating_result else None
       
        return {
            "total_feedback": total_feedback,
            "helpful_feedback": helpful_feedback,
            "helpfulness_rate": round(helpfulness_rate, 1),
            "average_rating": round(average_rating, 1) if average_rating else None
        }
       
    except Exception as e:
        # logger.error(f"Error getting feedback stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get feedback statistics")

@app.get("/chatbot/feedback/my-feedback")
async def get_user_feedback_history(
    current_user=Depends(get_current_active_sso_user),
    db: Session=Depends(get_chatbot_db)
):
    """
    Fetch the feedback history for the current user.
    """
    try:
        # Log the current user
        print(f"Fetching feedback for user_id: {current_user['user_id']}")
       
        # Ensure database session is valid
        if not db:
            raise HTTPException(status_code=500, detail="Database connection failed")
       
        # Get user's feedback with related chat history
        feedback_query = db.query(ChatbotFeedback).filter(
            ChatbotFeedback.user_id == current_user['user_id']
        ).order_by(ChatbotFeedback.timestamp.desc()).limit(10)
       
        if not feedback_query:
            return {
                "feedback_history": [],
                "total_count": 0
            }
       
        feedback_history = []
        for feedback in feedback_query:
            # Log each feedback record
            print(f"Processing feedback ID: {feedback.id}")
           
            # Get related chat history if available
            chat_question = None
            if feedback.chat_history_id:
                chat_record = db.query(ChatHistory).filter(
                    ChatHistory.id == feedback.chat_history_id
                ).first()
                if chat_record:
                    chat_question = chat_record.question
           
            feedback_history.append({
                "id": feedback.id,
                "rating": feedback.rating,
                "is_helpful": feedback.is_helpful,
                "feedback_text": feedback.feedback_text,
                "feedback_category": feedback.feedback_category,
                "is_accurate": feedback.is_accurate,
                "is_relevant": feedback.is_relevant,
                "is_clear": feedback.is_clear,
                "is_complete": feedback.is_complete,
                "timestamp": feedback.timestamp.isoformat() if feedback.timestamp else None,
                "chat_question": chat_question or "No question available"
            })
       
        return {
            "feedback_history": feedback_history,
            "total_count": len(feedback_history)
        }
       
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get feedback history")

@app.get("/chatbot/system/status")
async def get_chatbot_system_status():
    #Get chatbot system status and metrics
    try:
        return {
            "active_sessions": len(chatbot_session_manager.sessions),
            "background_tasks_active": background_tasks_active,
            "openai_service_status": "healthy",
            "database_status": "connected",
            "rate_limiting": {
                "enabled": True,
                "max_requests": rate_limiter.max_requests,
                "window_seconds": rate_limiter.window
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting system status: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=DEBUG_MODE,
        workers=1 if DEBUG_MODE else 4
    )