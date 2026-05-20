from requests import get
import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time

load_dotenv(override=True)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 0;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        width: 100%;
        position: relative;
        left: 0;
        right: 0;
        padding-top: 3rem;
        padding-bottom: 2.5rem;
    }
    
    /* Ensure header is at the very top and fills width */
    .stApp > header {
        background-color: transparent;
    }
    
    /* Remove padding from main container to make header full width */
    .main .block-container {
        padding-top: 0rem;
        max-width: 100%;
    }
    
    /* Chat message styling - Clean and simple */
    .user-message {
        background-color: #e3f2fd;
        color: #1a237e;
        padding: 0.75rem 1rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        font-size: 0.95rem;
    }
    
    .assistant-message {
        background-color: #f5f5f5;
        color: #212121;
        padding: 0.75rem 1rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
        max-width: 80%;
        border-left: 3px solid #667eea;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        font-size: 0.95rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e9ecef;
    }
    
    /* Button styling - Simple and clean */
    .stButton > button {
        background-color: #ffffff;
        color: #4a5568;
        border: 1px solid #cbd5e0;
        padding: 0.4rem 1rem;
        font-weight: 500;
        transition: all 0.2s;
        border-radius: 0.375rem;
        font-size: 0.875rem;
        width: auto;
        min-width: 100px;
    }
    
    .stButton > button:hover {
        background-color: #f7fafc;
        border-color: #a0aec0;
        transform: translateY(-1px);
    }
    
    /* Clear chat button specific style */
    button:has(div:contains("Clear Chat")) {
        border-color: #fc8181;
        color: #e53e3e;
    }
    
    button:has(div:contains("Clear Chat")):hover {
        background-color: #fff5f5;
        border-color: #fc8181;
    }
    
    /* Success message styling */
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    /* Green success box for documents loaded */
    .green-info-box {
        background: #d4edda;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        font-size: 0.875rem;
        color: #155724;
    }
    
    .warning-box {
        background: #fff3e0;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
        font-size: 0.875rem;
    }
    
    /* Custom info message styling (different from status) */
    .custom-info-message {
        background: #e8eaf6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        color: #1a237e;
        font-weight: 500;
    }
    
    /* Example questions styling */
    .example-questions {
        background: #fafafa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    
    /* Hide default info styling */
    .stAlert {
        background-color: transparent !important;
        padding: 0 !important;
    }
    
    /* Form button styling - hide send button */
    .stForm button {
        display: none;
    }
    
    /* Remove top padding from main content */
    .block-container {
        padding-top: 0rem;
    }
</style>
""", unsafe_allow_html=True)

prompt_template = """
Answer the following question based only on the provided context. Be concise but thorough:
<context>
    {context}
</context>
<question>
    {input}
</question>
"""

llm = ChatOpenAI(
    model="gpt-4o", 
    temperature=0.3, 
    api_key=os.getenv("OPENROUTER_API_KEY"), 
    base_url="https://openrouter.ai/api/v1",
    max_tokens=1000
)

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False

def display_chat_history():
    """Display chat history with proper styling - no icons"""
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message"><strong>You:</strong><br>{message["content"]}</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message"><strong>Assistant:</strong><br>{message["content"]}</div>', 
                       unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="RAG Chatbot - Smart Document Assistant", 
        layout="wide",
        page_icon="📚",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    # MAIN HEADER - At the very top, full width with more top padding
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0; font-size: 2rem;">📚 RAG Document Assistant</h1>
        <p style="color: white; margin-top: 0.5rem; margin-bottom: 0;">Intelligent Document Q&A with Retrieval Augmented Generation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # SIDEBAR - Document Management at the very top of sidebar
    with st.sidebar:
        st.markdown("## Document Management")
        
        with st.expander("Upload Documents", expanded=True):
            pdf_docs = st.file_uploader(
                label="Upload your PDF documents", 
                accept_multiple_files=True,
                help="Support for PDF files only"
            )
            
            if pdf_docs and st.button("Submit", use_container_width=True):
                with st.spinner(""):
                    try:
                        content = ""
                        
                        for idx, pdf in enumerate(pdf_docs):
                            reader = PdfReader(pdf)
                            for page in reader.pages:
                                content += page.extract_text()
                        
                        # Text splitting
                        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                            chunk_size=512, chunk_overlap=50
                        )
                        chunks = splitter.split_text(content)
                        
                        # Create embeddings
                        embedding_model = OpenAIEmbeddings(              
                            model='openai/text-embedding-ada-002',
                            api_key=os.getenv("OPENROUTER_API_KEY"),
                            base_url="https://openrouter.ai/api/v1"
                        )
                        
                        vector_store = Chroma.from_texts(
                            chunks,
                            embedding_model,
                            collection_name="data_collection",
                        )
                        
                        st.session_state.retriever = vector_store.as_retriever(
                            search_kwargs={"k": 3},
                        )
                        st.session_state.documents_loaded = True
                        
                        # Clear previous chat when new documents are loaded
                        st.session_state.messages = []
                        
                        time.sleep(1)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error processing documents: {str(e)}")
        
        # Document status in sidebar - using green box for success
        if st.session_state.documents_loaded:
            st.markdown("""
            <div class="green-info-box">
                ✅ Documents loaded and ready!<br>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-box">
                ⚠️No documents loaded<br>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick tips
        with st.expander("💡 Quick Tips"):
            st.markdown("""
            - Upload one or multiple PDF documents
            - Ask specific questions about the content
            - The assistant answers ONLY from your documents
            - Chat history resets when you load new documents
            """)
    
    # MAIN CONTENT AREA - Chat interface only
    st.markdown("### Chat Interface")
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    if st.session_state.documents_loaded:
        # Using form for Enter key support - no send button
        with st.form(key="chat_form", clear_on_submit=True):
            user_question = st.text_input(
                "Ask your question:",
                placeholder="Type your question here and press Enter...",
                key="user_input",
                label_visibility="collapsed"
            )
            submitted = st.form_submit_button("Submit", type="primary")
        
        # Clear chat button
        if st.button("Clear Chat", use_container_width=False, key="clear_chat"):
            st.session_state.messages = []
            st.rerun()
        
        if submitted and user_question:
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": user_question})
            
            with st.spinner("Thinking..."):
                try:
                    # Retrieve relevant context
                    context_docs = st.session_state.retriever.invoke(user_question)
                    context_list = [d.page_content for d in context_docs]
                    context_text = " --- ".join(context_list)
                    
                    # Generate response
                    prompt = prompt_template.format(context=context_text, input=user_question)
                    resp = llm.invoke(prompt)
                    
                    # Add assistant response to history
                    st.session_state.messages.append({"role": "assistant", "content": resp.content})
                    
                    # Rerun to update chat display
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    else:
        # Custom styled info message (different from status)
        st.markdown("""
        <div class="custom-info-message">
            <strong>Ready to start?</strong><br>
            Upload your PDF documents in the sidebar to begin asking questions about their content.
        </div>
        """, unsafe_allow_html=True)
        

if __name__ == "__main__":
    main()