import streamlit as st
import time
import os
import sys
import uuid
import json
import datetime
from upstash_redis import Redis
import extra_streamlit_components as stx

# --- Project Imports ---
# Add project root to path to allow for clean imports when running with `streamlit run`
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import initialize_components
from config import config

# --- Page Configuration ---
st.set_page_config(
    page_title="2IS Master's Program Assistant",
    page_icon="🎓",
    layout="centered"
)

# --- Enhanced Custom CSS for UTC Branding ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root variables for UTC branding */
    :root {
        --utc-red: #C41E3A;
        --utc-dark-red: #A01729;
        --utc-light-red: #E8344A;
        --utc-gold: #FFD700;
        --utc-cream: #FDF5E6;
        --utc-dark-gray: #2C2C2C;
        --utc-medium-gray: #666666;
        --utc-light-gray: #F8F9FA;
        --utc-white: #FFFFFF;
    }
    
    /* Global font styling */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container styling */
    .main .block-container {
        padding-top: 1rem;
        max-width: 900px;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Header section */
    .utc-header {
        background: linear-gradient(135deg, var(--utc-red) 0%, var(--utc-dark-red) 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(196, 30, 58, 0.15);
        text-align: center;
    }
    
    .utc-header h1 {
        color: var(--utc-white) !important;
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .utc-header .stCaption {
        color: var(--utc-cream) !important;
        font-size: 1.1rem !important;
        margin-bottom: 0 !important;
    }
    
    /* Logo styling */
    .logo-container {
        display: flex;
        justify-content: center;
        margin-bottom: 1.5rem;
    }
    
    .logo-container img {
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .logo-container img:hover {
        transform: scale(1.02);
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1cypcdb {
        background: linear-gradient(180deg, var(--utc-red) 0%, var(--utc-dark-red) 100%) !important;
    }
    
    .sidebar .stMarkdown h2 {
        color: var(--utc-white) !important;
        font-weight: 600 !important;
        border-bottom: 2px solid rgba(255,255,255,0.2);
        padding-bottom: 0.5rem;
        margin-bottom: 1rem !important;
    }
    
    .sidebar .stButton > button {
        background: var(--utc-white) !important;
        color: var(--utc-red) !important;
        border: 2px solid var(--utc-white) !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
    }
    
    .sidebar .stButton > button:hover {
        background: transparent !important;
        color: var(--utc-white) !important;
        border: 2px solid var(--utc-white) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
    }
    
    /* Sidebar metrics */
    .sidebar .stMetric {
        background: rgba(255, 255, 255, 0.1) !important;
        padding: 1.2rem !important;
        border-radius: 12px !important;
        margin: 1rem 0 !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .sidebar .stMetric [data-testid="metric-container"] {
        color: var(--utc-white) !important;
    }
    
    .sidebar .stMetric [data-testid="metric-container"] > div:first-child {
        color: var(--utc-cream) !important;
        font-size: 0.9rem !important;
    }
    
    /* Sidebar captions and text */
    .sidebar .stMarkdown {
        color: var(--utc-cream) !important;
    }
    
    .sidebar .stCaption {
        color: rgba(255, 255, 255, 0.8) !important;
        font-size: 0.9rem !important;
    }
    
    /* Link button in sidebar */
    .sidebar .stLinkButton > a {
        background: var(--utc-gold) !important;
        color: var(--utc-dark-gray) !important;
        border: 2px solid var(--utc-gold) !important;
        font-weight: 600 !important;
        text-decoration: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.2rem !important;
        display: block !important;
        text-align: center !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
    }
    
    .sidebar .stLinkButton > a:hover {
        background: var(--utc-white) !important;
        color: var(--utc-red) !important;
        border: 2px solid var(--utc-white) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
    }
    
    /* Divider in sidebar */
    .sidebar hr {
        border-color: rgba(255, 255, 255, 0.3) !important;
        margin: 1.5rem 0 !important;
    }
    
    /* Chat input styling */
    .stChatInput > div > div {
        border-radius: 12px !important;
    }
    
    .stChatInput > div > div > div > div {
        border: 2px solid var(--utc-red) !important;
        border-radius: 12px !important;
        font-size: 1rem !important;
    }
    
    .stChatInput > div > div > div > div:focus-within {
        border-color: var(--utc-light-red) !important;
        box-shadow: 0 0 0 3px rgba(196, 30, 58, 0.1) !important;
    }
    
    /* Chat messages */
    .stChatMessage {
        border-radius: 16px !important;
        margin: 1.5rem 0 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
    }
    
    .stChatMessage[data-testid="chat-message-user"] {
        background: linear-gradient(135deg, var(--utc-red) 0%, var(--utc-light-red) 100%) !important;
        border: none !important;
    }
    
    .stChatMessage[data-testid="chat-message-user"] .stMarkdown {
        color: var(--utc-white) !important;
        font-weight: 500 !important;
    }
    
    .stChatMessage[data-testid="chat-message-assistant"] {
        background: var(--utc-light-gray) !important;
        border: 2px solid var(--utc-cream) !important;
    }
    
    .stChatMessage[data-testid="chat-message-assistant"] .stMarkdown {
        color: var(--utc-dark-gray) !important;
        line-height: 1.6 !important;
    }
    
    /* Sources caption styling */
    .stChatMessage .stCaption {
        background: rgba(196, 30, 58, 0.1) !important;
        padding: 0.5rem 1rem !important;
        border-radius: 8px !important;
        margin-top: 1rem !important;
        border-left: 4px solid var(--utc-red) !important;
        color: var(--utc-medium-gray) !important;
        font-size: 0.85rem !important;
    }
    
    /* Success messages */
    .stSuccess {
        background: linear-gradient(135deg, var(--utc-red) 0%, var(--utc-light-red) 100%) !important;
        color: var(--utc-white) !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-top-color: var(--utc-red) !important;
    }
    
    /* Scrollbar styling for webkit browsers */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--utc-light-gray);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--utc-red);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--utc-dark-red);
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        .utc-header {
            padding: 1.5rem;
        }
        
        .utc-header h1 {
            font-size: 1.8rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Enhanced Header with Logo ---
st.markdown('<div class="logo-container">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    try:
        st.image("utc_logo.png", width=280)
    except:
        st.markdown("🎓 **Université Toulouse Capitole**")
st.markdown('</div>', unsafe_allow_html=True)

# Enhanced header section
st.markdown("""
<div class="utc-header">
    <h1>2IS Master's Program Assistant</h1>
    <p class="stCaption">Your intelligent guide to the Information Systems Engineering Master's program</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def get_cookie_manager():
    """
    Returns a CookieManager instance. Caching this prevents re-initialization
    on every script run and handles the initial setup gracefully.
    """
    return stx.CookieManager()

cookies = get_cookie_manager()

# --- PERSISTENT HISTORY MANAGEMENT WITH REDIS ---
@st.cache_resource
def get_redis_connection():
    """Establishes a connection to the Upstash Redis database."""
    try:
        # This automatically reads UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN from st.secrets
        conn = Redis.from_env()
        conn.ping() # Test the connection
        return conn
    except Exception as e:
        st.error(f"Failed to connect to Redis history database: {e}. Please check your Upstash secrets in Streamlit Cloud.")
        return None

redis_conn = get_redis_connection()

def save_history(session_id, user_id, messages, conn):
    """Saves the chat history of a session to Redis."""
    if not all([session_id, user_id, conn]):
        return
    try:
        # Use a pipeline for atomic operations
        pipe = conn.pipeline()
        # Save the actual chat messages
        pipe.set(f"history:{session_id}", json.dumps(messages), ex=2592000) # Expire after 30 days
        # Add this session to the user's sorted set of sessions, scored by timestamp
        pipe.zadd(f"user_sessions:{user_id}", {session_id: time.time()})
        pipe.execute()
    except Exception as e:
        st.error(f"Error saving history: {e}")

def load_history(session_id, conn):
    """Loads the chat history of a session from Redis."""
    if not all([session_id, conn]):
        return []
    try:
        data = conn.get(f"history:{session_id}")
        if data:
            return json.loads(data)
    except Exception as e:
        # Don't show error for a simple load, just log it if needed
        # st.error(f"Error loading history: {e}")
        pass
    return []

def delete_history(session_id, user_id, conn):
    """Deletes the history for a session from Redis."""
    if not all([session_id, user_id, conn]):
        return
    try:
        pipe = conn.pipeline()
        pipe.delete(f"history:{session_id}") # Delete the history itself
        pipe.zrem(f"user_sessions:{user_id}", session_id) # Remove from user's list
        pipe.execute()
    except Exception as e:
        st.error(f"Error deleting history: {e}")

def get_user_sessions(user_id, conn):
    """Retrieves all session IDs for a user, sorted from newest to oldest."""
    if not all([user_id, conn]):
        return []
    try:
        # ZREVRANGE gets items from a sorted set in reverse order (highest score first)
        return conn.zrevrange(f"user_sessions:{user_id}", 0, -1)
    except Exception:
        return []

# --- Caching and Initialization ---
@st.cache_resource
def load_rag_graph():
    """
    Loads all the necessary components for the RAG system.
    Using @st.cache_resource ensures this heavy operation runs only once.
    """
    rag_graph = initialize_components()
    if rag_graph is None:
        st.error("❌ Failed to initialize the RAG system. Please check the logs.", icon="🚨")
        st.stop()
    return rag_graph

rag_graph = load_rag_graph()

# --- USER IDENTITY MANAGEMENT ---
# This logic uses the more robust extra-streamlit-components cookie manager.
if 'user_id' not in st.session_state:
    # Get the user_id from the browser's cookies. It will return None if not found.
    user_id = cookies.get('user_id')
    
    # If the cookie doesn't exist, this is a new user.
    if not user_id:
        user_id = str(uuid.uuid4())
        # Set the cookie in the user's browser, making it expire in a year.
        cookies.set('user_id', user_id, expires_at=datetime.datetime.now() + datetime.timedelta(days=365))
    
    # Store the user_id in the session state for this run.
    st.session_state.user_id = user_id

# --- Enhanced Sidebar for Controls and Info ---
with st.sidebar:
    # Add secondary logo to sidebar
    try:
        st.image("ut1.png", width=180)
    except:
        st.markdown("🎓 **UTC 2IS Assistant**")
    
    st.markdown("## Controls & Information")
    
    if st.button("➕ New Chat", use_container_width=True):
        # A new chat is just a navigation to a URL without a session_id
        st.query_params.clear()
        st.rerun()

    st.divider()

    st.markdown("### 📜 Chat History")
    user_id = st.session_state.get("user_id")
    if user_id and redis_conn:
        user_sessions = get_user_sessions(user_id, redis_conn)
        if user_sessions:
            for sess_id in user_sessions:
                history = load_history(sess_id, redis_conn)
                if history:
                    # Use the first user message as the title, truncate if long
                    title = history[0].get('content', 'Chat')
                    title = (title[:35] + '...') if len(title) > 35 else title
                    
                    # Use columns to add a delete button
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        if st.button(title, key=f"session_{sess_id}", use_container_width=True):
                            st.query_params["session_id"] = sess_id
                            st.rerun()
                    with col2:
                        if st.button("🗑️", key=f"delete_{sess_id}", use_container_width=True, help="Delete this conversation"):
                            delete_history(sess_id, user_id, redis_conn)
                            st.rerun()
        else:
            st.caption("No past conversations yet.")

    st.divider()
    
    # Enhanced stats display
    stats = rag_graph.vector_store.get_stats()
    st.metric(
        label="📚 Knowledge Base", 
        value=f"{stats.get('total_documents', 'N/A')} docs"
    )
    
    if stats.get('syllabus_documents'):
        st.metric(
            label="📋 Syllabus Courses", 
            value=f"{stats.get('syllabus_documents', 0)} courses"
        )
    
    st.caption(f"🤖 **Model:** `{config.OPENROUTER_MODEL}`")
    st.caption(f"🔍 **Embeddings:** sentence-transformers")
    
    st.divider()
    
    if config.SYLLABUS_PDF_URL:
        st.caption("📖 **Need the complete syllabus?**")
        st.link_button("📥 Download Syllabus PDF", config.SYLLABUS_PDF_URL)
    
    st.markdown("---")
    st.caption("💡 **Tip:** Ask specific questions about courses, teachers, prerequisites, or program structure for the best results!")

# --- Enhanced Chat Logic ---

# --- SESSION ID & HISTORY MANAGEMENT ---
if "session_id" not in st.query_params:
    session_id = str(uuid.uuid4())
    st.query_params["session_id"] = session_id
else:
    session_id = st.query_params["session_id"]

# Store in session_state for easy access throughout the script
st.session_state["session_id"] = session_id # Store for easy access

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = load_history(session_id, redis_conn)

# Display past messages from history
if not st.session_state.messages:
    st.info("Welcome! Ask me anything about the 2IS Master's program to start the conversation.")

# Add a clear button only if there is history
if st.session_state.messages:
    if st.button("🗑️ Clear This Conversation"):
        delete_history(session_id, st.session_state.user_id, redis_conn)
        st.rerun()
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            st.caption(f"📎 **Sources:** {', '.join(message['sources'])}")

# Accept user input with enhanced placeholder
if prompt := st.chat_input("💬 Ask me anything about the 2IS Master's program..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # Placeholder for the main answer
        full_response = ""
        sources = []
        
        # Nodes that generate the primary answer
        answer_generating_nodes = ["generate", "conversational_response", "off_topic_response", "find_and_list_courses", "no_docs_fallback"]
        
        # Prepare conversation history for the model
        history_for_model = [
            {"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages
        ]
        
        # Use a spinner for the initial part
        with st.spinner("🤔 Thinking..."):
            # Use the new stream method to get results as they are generated
            for event in rag_graph.stream(prompt, history_for_model):
                # The event is a dictionary with one key (the node name)
                node_name, node_output = next(iter(event.items()))
                
                # Check if this node generated an answer
                if node_name in answer_generating_nodes:
                    answer = node_output.get("answer")
                    if answer:
                        full_response = answer
                        sources = node_output.get("sources", [])
                        # Display the answer immediately
                        message_placeholder.markdown(full_response)
                        if sources:
                            st.caption(f"📎 **Sources:** {', '.join(sources)}")
                
                # If we have an answer, we can stop waiting. This improves the user experience
                # by hiding the spinner as soon as the answer is displayed.
                if full_response:
                    break

        # so we don't need to handle the `evaluate_answer` node here anymore.
        # This simplifies the code and speeds up the user experience.
        
        # If no response was generated at all, show a fallback message.
        if not full_response:
            message_placeholder.markdown("I'm sorry, I encountered an issue and couldn't generate a response. Please try again.")
            # Add this fallback to the history as well
            full_response = "I'm sorry, I encountered an issue and couldn't generate a response."


    # Add assistant response to chat history
    if full_response:
        assistant_message = {"role": "assistant", "content": full_response, "sources": sources}
        st.session_state.messages.append(assistant_message)
        # Save the updated history to the file
        save_history(session_id, st.session_state.user_id, st.session_state.messages, redis_conn)