import streamlit as st
import requests
import uuid
import json
from datetime import datetime
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database_utils import *
from db_utils import get_top_questions
import functools
from dotenv import load_dotenv
from database_utils import AzureSQLConnector
from sqlalchemy.orm import sessionmaker

load_dotenv()

connector = AzureSQLConnector()
engine = connector.create_sqlalchemy_engine()
if not engine:
    raise ConnectionError("Could not create SQLAlchemy engine. Check connection details.")

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
db = SessionLocal()

# Streamlit Database Optimization
# Thread pool for database operations to prevent UI blocking
db_thread_pool = ThreadPoolExecutor(max_workers=5, thread_name_prefix="streamlit_db")

# Persistent requests session for better performance or connection pooling
@functools.lru_cache(maxsize=1)
def get_requests_session():
    session = requests.Session()
    session.headers.update({
        'Connection': 'keep-alive',
        'Accept-Encoding': 'gzip, deflate',
        'User-Agent': 'GSC-ARB-Chatbot-Frontend/1.0'
    })
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=10,
        pool_maxsize=20,
        max_retries=3
    )
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

@st.cache_data(ttl=300)
def get_top_questions_cached(limit=5):
    def _get_questions():
        try:
            thread_db = SessionLocal()
            try:
                return get_top_questions(thread_db, limit=limit)
            finally:
                thread_db.close()
        except Exception as e:
            st.error(f"Error loading FAQs: {e}")
            return []
    
    # Database operation in thread pool
    future = db_thread_pool.submit(_get_questions)
    try:
        # Wait for result with timeout to prevent hanging
        return future.result(timeout=10)
    except Exception as e:
        st.warning(f"Could not load FAQs: {e}")
        return []

@st.cache_data(ttl=60)
def test_api_connection_cached():
    try:
        session = get_requests_session()
        response = session.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except requests.exceptions.RequestException as e:
        return False, str(e)

@st.cache_data(ttl=120)
def get_feedback_stats_cached(headers):
    session = get_requests_session()
    for attempt in range(3):
        try:
            stats_response = session.get(f"{API_BASE_URL}/chatbot/feedback/stats", headers=headers, timeout=10)
            if stats_response.status_code == 200:
                return True, stats_response.json()
            else:
                return False, f"Status: {stats_response.status_code}"
        except requests.exceptions.RequestException as e:
            if attempt == 2:
                return False, str(e)
            time.sleep(0.5 * (attempt + 1))
    return False, "Max retries exceeded"

@st.cache_data(ttl=180)
def get_user_feedback_cached(headers):
    try:
        feedback_response = requests.get(f"{API_BASE_URL}/chatbot/feedback/my-feedback", headers=headers, timeout=10)
        if feedback_response.status_code == 200:
            return True, feedback_response.json()
        else:
            return False, f"Status: {feedback_response.status_code}"
    except Exception as e:
        return False, str(e)

@st.cache_data(ttl=30)
def get_system_status_cached(headers):
    try:
        status_response = requests.get(f"{API_BASE_URL}/chatbot/system/status", timeout=10, headers=headers)
        if status_response.status_code == 200:
            return True, status_response.json()
        else:
            return False, f"Status: {status_response.status_code}"
    except Exception as e:
        return False, str(e)

# Page config
st.info("Note: Ask Questions related to ARB Process, Platform Provisioning")

# SSO Authentication
import requests
import streamlit as st
from msal import ConfidentialClientApplication
from dotenv import load_dotenv
import os
import extra_streamlit_components as stx

def get_manager():
    cm = stx.CookieManager()
    return cm

if 'cm' not in st.session_state:
    st.session_state.cm = get_manager()
    
CONTROLLER = st.session_state.cm

def initialize_client():
    load_dotenv()
    client_id = os.getenv("AZURE_AD_CLIENT_ID")
    tenant_id = os.getenv("AZURE_AD_TENANT_ID")
    secret = os.getenv("AZURE_AD_CLIENT_SECRET")
    url = f"https://login.microsoftonline.com/{tenant_id}"

    return ConfidentialClientApplication(client_id=client_id,authority=url,client_credential=secret)

def acquire_access_token(app:ConfidentialClientApplication, code, scopes, redirect_uri):
    return app.acquire_token_by_authorization_code(code, scopes=scopes, redirect_uri=redirect_uri)

def fetch_user_data(access_token):
    headers = {"Authorization": f"Bearer {access_token}"}
    graph_api_endpoint = "https://graph.microsoft.com/v1.0/me"
    response = requests.get(graph_api_endpoint, headers=headers)
    return response.json()

# Helper functions
def nav_to(url):
    nav_script = f"<meta http-equiv='refresh' content='0; url={url}'>"
    st.write(nav_script, unsafe_allow_html=True)

def authenticate(app:ConfidentialClientApplication):
    scopes = ["User.Read"]
    redirect_url = os.getenv("AZURE_AD_REDIRECT_URL")
    auth_url = app.get_authorization_request_url(scopes,redirect_uri=redirect_url)
    if len(list(st.query_params)) == 0:
        nav_to(auth_url)

    if st.query_params.get("code"):
        print(list(st.query_params))
        st.session_state["auth_code"] = st.query_params.get("code")
        token_result = acquire_access_token(app, st.session_state.auth_code, scopes, redirect_uri=redirect_url)
        if "access_token" in token_result:
            print(token_result)
            username = token_result['id_token_claims']['name']
            email_id = token_result['id_token_claims']['email']
            access_token = token_result['access_token']
            refresh_token = token_result['refresh_token']
            id_token = token_result['id_token']
            CONTROLLER.set('access_token',access_token)
            st.session_state['username'] =  username
            st.session_state['email_id'] =  email_id
            return True
        else:
            return False

def login():
    print("login")
    app = initialize_client()
    user_data = authenticate(app)
    if user_data:
        print(user_data)
        st.session_state["authenticated"] = True
        redirect_url = os.getenv("REDIRECT_URL")
        st.rerun()

if __name__ == "__main__":
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    if st.session_state['authenticated'] == False:
        login()
    else:
        # FAQs
        st.sidebar.header("FAQs")
        try:
            # Cached version that runs in background thread
            top_questions = get_top_questions_cached(limit=5)
            for q, cnt in top_questions:
                if st.sidebar.button(f"{q[:50]}... ({cnt}× asked)", key=f"faq_{hash(q)}"):
                    st.session_state.question = q
        except Exception as e:
            st.sidebar.warning("FAQs temporarily unavailable")

        # Initialize session state
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())

        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "feedback_given" not in st.session_state:
            st.session_state.feedback_given = {}

        # Memory optimization by limiting message history to prevent memory issues
        def optimize_message_history():
            MAX_MESSAGES = 100
            if len(st.session_state.messages) > MAX_MESSAGES:
                st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]
                valid_message_ids = {f"msg_{i}" for i in range(len(st.session_state.messages))}
                st.session_state.feedback_given = {
                    k: v for k, v in st.session_state.feedback_given.items() if k in valid_message_ids
                }

        # Helper function to test API connection (uses cached version)
        def test_api_connection():
            # Wrapper that uses cached version for better performance
            return test_api_connection_cached()

        # Helper function to validate SSO token with backend
        @st.cache_data(ttl=300)
        def validate_sso_token_with_backend(token):
            try:
                session = get_requests_session()
                headers = {"Authorization": f"Bearer {token}"}
                response = session.get(f"{API_BASE_URL}/sso/validate-token", headers=headers, timeout=10)
                if response.status_code == 200:
                    return True, response.json()
                else:
                    return False, f"Status: {response.status_code}"
            except Exception as e:
                return False, str(e)

        # Enhanced Feedback UI
        def render_feedback_ui(message_id, message_index=None):
            # Render feedback UI for a specific message
            if message_id in st.session_state.feedback_given:
                st.success("Feedback submitted! Thank you..!!")
                return
            
            # Get the message data
            if message_index is not None and message_index < len(st.session_state.messages):
                message = st.session_state.messages[message_index]
                response_id = message.get("response_id")
                chat_history_id = message.get("chat_history_id")
            else:
                response_id = None
                chat_history_id = None
            
            st.markdown("---")
            st.markdown("**Was this response helpful?**")
            
            # Create unique keys for this message
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button("👍 Helpful", key=f"up_{message_id}"):
                    submit_feedback(
                        response_id=response_id,
                        chat_history_id=chat_history_id,
                        is_helpful=True,
                        message_id=message_id
                    )
            
            with col2:
                if st.button("👎 Not Helpful", key=f"down_{message_id}"):
                    submit_feedback(
                        response_id=response_id,
                        chat_history_id=chat_history_id,
                        is_helpful=False,
                        message_id=message_id
                    )
            
            # Detailed feedback form
            with st.expander("Provide detailed feedback (optional)"):
                rating = st.select_slider(
                    "Rate this response (1-5 stars)",
                    options=[1, 2, 3, 4, 5],
                    value=3,
                    key=f"rating_{message_id}"
                )
                
                feedback_category = st.selectbox(
                    "What aspect needs improvement?",
                    ["accuracy", "helpfulness", "clarity", "completeness", "relevance", "other"],
                    key=f"category_{message_id}"
                )
                
                feedback_text = st.text_area(
                    "Additional comments",
                    placeholder="Tell us how we can improve...",
                    key=f"text_{message_id}"
                )
                
                # Detailed ratings
                col_acc, col_rel, col_clear, col_comp = st.columns(4)
                with col_acc:
                    is_accurate = st.checkbox("Accurate", key=f"acc_{message_id}")
                with col_rel:
                    is_relevant = st.checkbox("Relevant", key=f"rel_{message_id}")
                with col_clear:
                    is_clear = st.checkbox("Clear", key=f"clear_{message_id}")
                with col_comp:
                    is_complete = st.checkbox("Complete", key=f"comp_{message_id}")
                
                if st.button("Submit Detailed Feedback", key=f"submit_{message_id}"):
                    submit_feedback(
                        response_id=response_id,
                        chat_history_id=chat_history_id,
                        rating=rating,
                        is_helpful=None,
                        feedback_text=feedback_text,
                        feedback_category=feedback_category,
                        is_accurate=is_accurate,
                        is_relevant=is_relevant,
                        is_clear=is_clear,
                        is_complete=is_complete,
                        message_id=message_id
                    )

        def submit_feedback(response_id=None, chat_history_id=None, message_id=None, **feedback_data):
            # Submit feedback to the API with enhanced error handling
            try:
                # Debug logging
                print(f"Submitting feedback - response_id: {response_id}, chat_history_id: {chat_history_id}")
                
                # Retrieve token from the cookie
                token = CONTROLLER.get("access_token")
                if not token:
                    st.error("Session expired. Please log in again.")
                    st.stop()

                # Add the  token to the headers
                headers = {"Authorization": f"Bearer {token}"}
                
                # Enhanced logic to find response identifiers
                if not response_id and not chat_history_id:
                    # Try to find from the most recent assistant message
                    for msg in reversed(st.session_state.messages):
                        if msg["role"] == "assistant":
                            response_id = msg.get("response_id")
                            chat_history_id = msg.get("chat_history_id")
                            if response_id or chat_history_id:
                                print(f"Found identifiers from recent message: response_id={response_id}, chat_history_id={chat_history_id}")
                                break
                
                feedback_payload = {
                    "response_id": response_id,
                    "chat_history_id": chat_history_id,
                    "session_id": st.session_state.session_id,  # Add session_id
                    **feedback_data
                }
                
                # Remove message_id from payload as it's only for UI state
                feedback_payload.pop("message_id", None)
                
                # Remove None values to avoid sending unnecessary data
                feedback_payload = {k: v for k, v in feedback_payload.items() if v is not None}
                
                # If no identifiers at all, we can add a flag for backend to use the latest chat
                if not response_id and not chat_history_id:
                    feedback_payload["use_latest_chat"] = True
                
                print(f"Final feedback payload: {feedback_payload}")
                
                response = requests.post(
                    f"{API_BASE_URL}/chatbot/feedback",
                    json=feedback_payload,
                    headers=headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    st.session_state.feedback_given[message_id] = True
                    st.success("Thank you for your feedback!")
                    st.rerun()
                else:
                    error_detail = "Unknown error"
                    try:
                        if response.headers.get('content-type', '').startswith('application/json'):
                            error_json = response.json()
                            error_detail = error_json.get('detail', response.text)
                        else:
                            error_detail = response.text
                    except:
                        error_detail = f"HTTP {response.status_code}: {response.reason}"
                    
                    st.error(f"Failed to submit feedback: {error_detail}")
                    print(f"Feedback submission failed: {response.status_code} - {error_detail}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Network error submitting feedback: {str(e)}")
                print(f"Network error: {e}")
            except Exception as e:
                st.error(f"Error submitting feedback: {str(e)}")
                print(f"Unexpected error: {e}")

        # Main UI
        st.title("GSC ARB Chatbot")

        # Validate SSO token with backend
        token = CONTROLLER.get("access_token")
        if token:
            token_valid, token_data = validate_sso_token_with_backend(token)
            if not token_valid:
                st.error("SSO token validation failed. Please refresh and login again.")
                st.stop()

        # Connection status
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**Session ID:** `{st.session_state.session_id}`")
            if "username" in st.session_state:
                st.write(f"**User:** {st.session_state.username}")
            if "email_id" in st.session_state:
                st.write(f"**Email:** {st.session_state.email_id}")

        with col2:
            is_connected, health_data = test_api_connection()
            if is_connected:
                st.success("ARB Chatbot API Connected")
            else:
                st.error("API Disconnected")

        # Display chat messages
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "timestamp" in message:
                    st.caption(f"*{message['timestamp']}*")
                
                # Feedback UI for assistant messages
                if message["role"] == "assistant":
                    render_feedback_ui(
                        message_id=f"msg_{i}",
                        message_index=i
                    )

        # Chat input
        st.toast("Welcome..!!")
        if prompt := st.chat_input("What's on your mind?"):
            if not test_api_connection()[0]:
                st.error("Cannot connect to ARB Chatbot API.")
                st.stop()

            # Add user message to session state
            user_message = {
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
            st.session_state.messages.append(user_message)
            optimize_message_history()
            with st.chat_message("user"):
                st.markdown(prompt)
                st.caption(f"*{user_message['timestamp']}*")

            # Get response from API
            with st.chat_message("assistant"):
                message_placeholder = st.empty()

                with st.spinner("ARB Chatbot is Generating Response..."):
                    try:
                        # Retrieve token from the cookie
                        token = CONTROLLER.get("access_token")
                        if not token:
                            st.error("Session expired. Please log in again.")
                            st.stop()

                        # Add the token to the headers
                        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

                        # Prepare request data
                        request_data = {
                            "message": prompt,
                            "session_id": st.session_state.session_id
                        }

                        # Persistent session for better performance
                        session = get_requests_session()
                        response = session.post(
                            f"{API_BASE_URL}/chat",
                            json=request_data,
                            timeout=120,
                            headers=headers
                        )

                        if response.status_code == 200:
                            result = response.json()
                            assistant_response = result["response"]
                            response_id = result.get("request_id") or result.get("response_id")
                            chat_history_id = result.get("chat_history_id")

                            # Generate a response_id if not provided by API
                            if not response_id:
                                response_id = str(uuid.uuid4())
                                print(f"Generated fallback response_id: {response_id}")

                            # Extract token usage details
                            prompt_tokens = result.get("prompt_tokens")
                            completion_tokens = result.get("completion_tokens")
                            total_tokens = result.get("total_tokens")
                            total_cost = result.get("total_cost")

                            # Clear placeholder and show response
                            message_placeholder.empty()
                            st.markdown(assistant_response)
                            st.info(f"prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, total_tokens={total_tokens}, total_cost={total_cost}")

                            # Add assistant response to chat history with identifiers
                            assistant_message = {
                                "role": "assistant",
                                "content": assistant_response,
                                "timestamp": datetime.now().strftime("%H:%M:%S"),
                                "response_id": response_id,
                                "chat_history_id": chat_history_id
                            }
                            st.session_state.messages.append(assistant_message)
                            st.caption(f"*{assistant_message['timestamp']}*")

                            # Show feedback UI for this new message
                            message_id = f"msg_{len(st.session_state.messages) - 1}"
                            render_feedback_ui(
                                message_id=message_id,
                                message_index=len(st.session_state.messages) - 1
                            )

                        elif response.status_code == 429:
                            message_placeholder.error("Rate limit exceeded. Please wait before sending another message.")
                        elif response.status_code == 503:
                            message_placeholder.warning("ARB Chatbot Server busy. Request queued for processing.")
                        else:
                            error_detail = response.text
                            try:
                                error_json = response.json()
                                error_detail = error_json.get("detail", error_detail)
                            except:
                                pass
                            message_placeholder.error(f"Error {response.status_code}: {error_detail}. Please refresh the page and try again.")

                    except requests.exceptions.Timeout:
                        message_placeholder.error("Request timed out after 120 seconds. Please re-submit your query.")
                    except requests.exceptions.ConnectionError:
                        message_placeholder.error("Connection error. Is the FastAPI server running on port 8000?")
                    except requests.exceptions.RequestException as e:
                        message_placeholder.error(f"Request error: {str(e)}")

        # Sidebar
        with st.sidebar:
            st.header("System Controls")
            if st.button("Refresh Status"):
                get_system_status_cached.clear()
            try:
                success, status_data = get_system_status_cached(headers={"Authorization": f"Bearer {CONTROLLER.get('access_token')}"})
                if success:
                    st.success("Chatbot System Status")
                    st.json(status_data)
                else:
                    st.error(f"Status check failed: {status_data}")
            except Exception as e:
                st.error(f"Could not fetch status: {e}")
            
            st.divider()
            
            # Feedback Statistics - Optimized with caching
            st.subheader("Feedback Stats")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("View My Feedback", use_container_width=True):
                    # Clear cache for fresh data when explicitly requested
                    get_user_feedback_cached.clear()
            
            with col2:
                if st.button("Overall Stats", use_container_width=True):
                    # Clear cache for fresh data when explicitly requested
                    get_feedback_stats_cached.clear()
            
            # Show cached feedback data
            try:
                headers = {"Authorization": f"Bearer {CONTROLLER.get('access_token')}"}
                success, feedback_data = get_user_feedback_cached(headers)
                if success and feedback_data.get("feedback_history"):
                    st.subheader("Your Feedback History")
                    for feedback in feedback_data["feedback_history"][:3]:  # Show last 3 for better performance
                        with st.expander(f"Feedback from {feedback['timestamp'][:10]}"):
                            st.write(f"**Rating:** {feedback.get('rating', 'N/A')}")
                            st.write(f"**Helpful:** {'Yes' if feedback.get('is_helpful') else 'No' if feedback.get('is_helpful') is False else 'N/A'}")
                            if feedback.get('feedback_text'):
                                st.write(f"**Comment:** {feedback['feedback_text']}")
                            if feedback.get('chat_question'):
                                st.write(f"**Question:** {feedback['chat_question'][:100]}...")
                elif success:
                    st.info("No feedback history found.")
            except Exception as e:
                st.warning("Feedback history temporarily unavailable")
            
            # Show cached stats
            try:
                headers = {"Authorization": f"Bearer {CONTROLLER.get('access_token')}"}
                success, stats = get_feedback_stats_cached(headers)
                if success:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Feedback", stats.get("total_feedback", 0))
                    with col2:
                        st.metric("Helpfulness Rate", f"{stats.get('helpfulness_rate', 0):.1f}%")
                    with col3:
                        if stats.get("average_rating"):
                            st.metric("Average Rating", f"{stats['average_rating']:.1f}/5")
            except Exception as e:
                st.warning("Stats temporarily unavailable")
            
            st.divider()
            
            # Session Controls
            if st.button("Clear Session", use_container_width=True):
                st.session_state.messages = []
                st.session_state.feedback_given = {}
                st.success("Session cleared!")
                st.rerun()
            
            if st.button("New Session", use_container_width=True):
                # Create new session via backend API
                try:
                    token = CONTROLLER.get("access_token")
                    if token:
                        headers = {"Authorization": f"Bearer {token}"}
                        session = get_requests_session()
                        response = session.post(f"{API_BASE_URL}/sessions", headers=headers, timeout=10)
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.session_id = result["session_id"]
                            st.session_state.messages = []
                            st.session_state.feedback_given = {}
                            st.success("New session created successfully!")
                        else:
                            # Fallback to local session creation
                            st.session_state.session_id = str(uuid.uuid4())
                            st.session_state.messages = []
                            st.session_state.feedback_given = {}
                            st.success("New session started!")
                    else:
                        # Fallback to local session creation
                        st.session_state.session_id = str(uuid.uuid4())
                        st.session_state.messages = []
                        st.session_state.feedback_given = {}
                        st.success("New session started!")
                except Exception as e:
                    # Fallback to local session creation
                    st.session_state.session_id = str(uuid.uuid4())
                    st.session_state.messages = []
                    st.session_state.feedback_given = {}
                    st.success("New session started!")
                    print(f"Error creating backend session: {e}")
                
                st.rerun()
            
            st.divider()
            
            # API Health Check
            st.subheader("ARB Chatbot API Health")
            health_status, health_info = test_api_connection()
            
            if health_status:
                st.success("ARB Chatbot API is healthy")
                if health_info:
                    with st.expander("Health Details"):
                        st.json(health_info)
            else:
                st.error("ARB Chatbot API is down")
                st.error(f"Error: {health_info}")
                
                st.subheader("Troubleshooting Steps:")
                st.markdown("""
                1. **Check with Team ARB**
                --> POC: Harshit, Yogesh, Sri (Line Manager)
                """)
            
            st.divider()
            
            # Session Info
            st.subheader("Session Info")
            st.write(f"**Messages:** {len(st.session_state.messages)}")
            st.write(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
            
            # Download chat history
            if st.session_state.messages:
                chat_history = "\n\n".join([
                    f"**{msg['role'].title()}** ({msg.get('timestamp', 'N/A')}):\n{msg['content']}"
                    for msg in st.session_state.messages
                ])
                
                st.download_button(
                    label="Download Chat",
                    data=chat_history,
                    file_name=f"chat_history_{st.session_state.session_id[:8]}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

        # Footer
        st.markdown("---")
        st.markdown("**GSC ARB Chatbot** - Team ARB")

        # Cleanup function
        import atexit

        def cleanup_resources():
            try:
                db_thread_pool.shutdown(wait=False)
                if db:
                    db.close()
            except:
                pass

        # Register cleanup function
        atexit.register(cleanup_resources)