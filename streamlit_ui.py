import streamlit as st
from time import sleep
from src.llm_model import query_solution, query_rewritting, query_fred_api_needed
from src.loaders import load_pdfs
from src.data_preprocessing import pdf_splitter
from src.embedders import ChromaDatabase
from src.query import VectorRetriever
from src.constants import MODELS_AVAILABLE, PDF_DATA_FOLDER_DIR, default_embedding_model
from dotenv import load_dotenv
from datetime import datetime

# Loading API Keys
load_dotenv()

# Add logs to Testing Area
def add_log(message: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append({"time": timestamp, "message": message})

# Clear chat hisotory and logs, including backed chat history
def clear_chat_n_logs():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.session_state.logs = []


# General page settings
st.set_page_config(
    page_title="RAG System",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initializing required variables
if "query_rewritting_indicator" not in st.session_state:
    st.session_state.query_rewritting_indicator = False
if 'pdfs_loaded'not in st.session_state:
    st.session_state.pdfs_loaded = False
if 'chroma_database' not in st.session_state:
    st.session_state.chroma_database = ChromaDatabase(embedding_model=default_embedding_model, distance_metric='cosine')


# Left side bar - Model & Data interactive management
with st.sidebar:
    st.title("💼 RAG Economic Analyst Bot")
    #st.caption("RAG Economic Analyst Assistant")
    
    # Model settings
    st.markdown("## ⚙️ Model Settings")
    model_to_use = st.selectbox("Choose a model", MODELS_AVAILABLE, key="selected_model")
    temperature = st.slider("Temperature", 0.1, 2.0, 0.3, 0.1)
    top_k = st.slider("Top K Contexts", 1, 15, 5, 1)
    
    # Feature to obtain 3 semantically similar queries to the original one
    if st.sidebar.checkbox("Rewrite the Query 3 times"):
        st.session_state.query_rewritting_indicator = True

    st.divider()

    # Data settings
    st.markdown("## 📂 Data Settings")
    chunk_size = st.slider("Chunk Size", 200, 2000, 800, 100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50, 5)

    # re-run embeddings, saved in memory, not local
    if st.button("🔄 Re-run Embeddings"):
        st.session_state.chroma_database.erase_chroma_db()
        add_log(f'Re-running embeddings...')
        
        add_log(f'Loading PDFs & Financial Statements...')

        # only load pdfs once
        if not st.session_state.pdfs_loaded:
            st.session_state.pdf_docs = load_pdfs()
            st.session_state.pdfs_loaded = True

        add_log(f'Loaded {len(st.session_state.pdf_docs)} pdfs')

        add_log(f'Chunking Data...')
        chunks_pdf = pdf_splitter(st.session_state.pdf_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        add_log(f'Total Chunks {len(chunks_pdf)}')

        add_log(f'Embedding chunks...')
        st.session_state.chroma_database.embed_documents(chunks_pdf)
        st.session_state.vector_retriever = VectorRetriever(st.session_state.chroma_database.get_vector_db())
        st.session_state.embeddings_run = True
        add_log(f'Vector database has been saved')

    st.divider()
    st.button("🗑️ Clear Chat", on_click=clear_chat_n_logs)

    st.divider()
    st.markdown("Developed by **Julian Restrepo**")

col1, col2 = st.columns([2, 1])  # left = chat, right = extra panel

with col1:
    
    # chat area
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! how can I help you today?"}
        ]

    
    for message in st.session_state.messages:
        avatar = "🧠" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # general function assitant reply
    def generate_response(query, context, fred_data:str=None):
        conversation_history = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}"
            for m in st.session_state.messages
        )
        return query_solution(query,context, conversation_history , temperature, model_to_use, fred_data=fred_data)
    

    # user input
    if prompt := st.chat_input("Type your question here..."):
        # 1. Store and show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # As embeddings are saved in-memory, run embeddings once before first run
        if not st.session_state.pdfs_loaded:
            st.session_state.messages.append({"role": "assistant", "content": "Run embeddings once before querying."})
        else:
        
            
            with st.chat_message("assistant", avatar="🧠"):
                with st.spinner("Thinking..."):

                    
                    if st.session_state.query_rewritting_indicator:

                        add_log(f'Processing query...')
                        add_log(f'Requerying query')
                        query_list = query_rewritting(prompt)
                        add_log(f'Queries generated {query_list}')
                        context = st.session_state.vector_retriever.multiple_query(query_list, top_k=top_k)
                        add_log(f'Top {top_k} context were retrieved')
                        add_log(context)
                        fred_data = query_fred_api_needed(prompt)
                        add_log(fred_data)
                        response = generate_response(prompt, context, fred_data)

                        placeholder = st.empty()
                        full_response = ""
                        for token in response:
                            full_response += token
                            placeholder.markdown(full_response)
                            sleep(0.01)
                        placeholder.markdown(full_response)

                    # only running 1 query
                    else:

                        add_log(f'Processing query...')
                        context = st.session_state.vector_retriever.single_query(prompt, top_k=top_k)
                        add_log(context)
                        add_log(f'Top {top_k} context were retrieved')
                        fred_data = query_fred_api_needed(prompt)
                        add_log(fred_data)
                        response = generate_response(prompt, context, fred_data)
                        add_log('Query answered')
                        placeholder = st.empty()
                        full_response = ""
                        for token in response:
                            full_response += token
                            placeholder.markdown(full_response)
                            sleep(0.01)
                        placeholder.markdown(full_response)


                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun() 

    
with col2:
    
    st.header("📊 Testing Area")
    st.write("Summary of logs and back-end processes.")

    # log repository
    if "logs" not in st.session_state:
        st.session_state.logs = []

    # updating logs
    if st.session_state.logs:
        for entry in st.session_state.logs:  # newest first
            st.markdown(f"**[{entry['time']}]** {entry['message']}")
    else:
        st.info("Please run embeddings before making your first query.")


# style settings
st.markdown("""
<style>
    .stChatInput textarea {border-radius: 12px !important;}
    .stButton>button {width: 100%; border-radius: 8px;}
    [data-testid="stSidebar"] {background-color: #f8f9fa;}
    .stChatMessage {border-radius: 12px; padding: 8px;}

    /* Keep chat input fixed at bottom */
    [data-testid="stChatInput"] {
        position: fixed !important;
        bottom: 0 !important;
        left:var(--sidebar-width, 250px) ;
        right:0;
        background-color: white;
        padding: 10px 15px;
        box-shadow: 0 -2px 6px rgba(0,0,0,0.05);
        z-index: 1000;
    }
    [data-testid="stChatMessageContainer"] {
        padding-bottom: 100px !important;
    }
        [data-testid="stSidebar"][aria-expanded="false"] ~ div [data-testid="stChatInput"] {
        left: 0 !important;
</style>
""", unsafe_allow_html=True)
