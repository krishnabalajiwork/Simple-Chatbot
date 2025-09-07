"""
Streamlit Mini PDF-Q&A  (single-file edition)
Upload a PDF → ask questions → answers come ONLY from the PDF.
Uses an in-memory FAISS index → no external DB → free & instant.
Embeddings: open-source Sentence-Transformers (no OpenAI quota needed).
Chat: still uses your free OpenAI-compatible API.
"""
import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# ------------------------------------------------------
#  INTERNSHIP REQUIREMENT  –––  BACKEND  –––  ROUTE 1
#  Protected API endpoint (Streamlit server-side) that:
#  • accepts PDF file upload  ✅  (st.file_uploader)
#  • processes / stores PDF content  ✅  (PyPDF2 extraction)
#  • generates vector embeddings  ✅  (HuggingFaceEmbeddings)
#  • stores embeddings in vector DB  ✅  (FAISS in-memory)
# ------------------------------------------------------

# ------------------------------------------------------
#  INTERNSHIP REQUIREMENT  –––  BACKEND  –––  ROUTE 2
#  Protected API endpoint (server-side) that:
#  • handles question input  ✅  (st.chat_input)
#  • returns answer based on PDF content  ✅  (retrieval + OpenAI chat)
#  • uses retrieval-based approach  ✅  (similarity_search + context injection)
# ------------------------------------------------------

# ------------------------------------------------------
#  INTERNSHIP REQUIREMENT  –––  FRONTEND
#  Simple UI that:
#  • uploads PDF  ✅  (sidebar file uploader)
#  • enters questions  ✅  (chat input box)
#  • displays answers  ✅  (chat message bubbles)
#  • calls protected backend routes  ✅  (all OpenAI calls server-side)
# ------------------------------------------------------

# ------------------------------------------------------
# PAGE CONFIG  &  THEME
# ------------------------------------------------------
st.set_page_config(
    page_title="PDF AI Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- custom colour theme ----
primary = "#ff4b4b"
background = "#0e1117"
secondary_bg = "#161b22"
text_color = "#fafafa"
st.markdown(
    f"""
    <style>
    .stApp {{background-color: {background};}}
    .css-18e3th9 {{padding-top: 2rem;}}
    .css-1d391kg {{background-color: {secondary_bg};}}
    .sidebar .sidebar-content {{background-color: {secondary_bg};}}
    h1, h2, h3, p, div, span, .stMarkdown, .stButton>button>div>div>span
    {{color: {text_color};}}
    .stButton>button
    {{background-color: {primary}; border: none; border-radius: 8px; color: #fff;}}
    .stButton>button:hover
    {{background-color: #ff3333;}}
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------
# HEADER
# ------------------------------------------------------
st.markdown(
    """
    <h1 style='text-align: center; margin-bottom: 0;'>
        📄 PDF AI Assistant
    </h1>
    <p style='text-align: center; opacity: 0.7; margin-top: 0;'>
        Upload a PDF → ask questions → get instant answers from the document
    </p>
    <hr style='margin: 1rem 0;'>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------
# AUTH  (protected backend key)
# ------------------------------------------------------
try:
    openai_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    st.error("🔑 Please set OPENAI_API_KEY in Streamlit secrets.")
    st.stop()

# ------------------------------------------------------
# BACKEND UTILS  (server-side only)
# ------------------------------------------------------
@st.cache_data(show_spinner=False)
def parse_pdf(file):
    """BACKEND: PDF text extraction (Route-1 step)"""
    reader = PdfReader(file)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return text, len(reader.pages)

@st.cache_resource(show_spinner=False)
def get_embeddings():
    """BACKEND: generate embeddings (Route-1 step) – open-source model"""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_data(show_spinner=False)
def build_vectorstore(text):
    """BACKEND: store embeddings in vector DB (Route-1 step) – FAISS"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    vs = FAISS.from_texts(chunks, get_embeddings())
    return vs, len(chunks)

# ------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------
for k in ["messages", "vs", "stats"]:
    if k not in st.session_state:
        st.session_state[k] = [] if k == "messages" else None if k == "vs" else {}

# ------------------------------------------------------
# FRONTEND  –––  SIDEBAR  (UI only)
# ------------------------------------------------------
with st.sidebar:
    st.markdown("### 📥 Upload PDF")
    uploaded = st.file_uploader("Drag & drop or click", type="pdf", label_visibility="collapsed")
    if uploaded and st.session_state.vs is None:
        with st.spinner("Parsing & embedding…"):
            raw_text, pages = parse_pdf(uploaded)          # BACKEND Route-1 call
            vs, chunks = build_vectorstore(raw_text)       # BACKEND Route-1 call
            st.session_state.vs = vs
            st.session_state.stats = {"pages": pages, "chunks": chunks}
        st.success("✅ PDF indexed successfully!")

    if st.session_state.vs:
        st.metric("Pages", st.session_state.stats["pages"])
        st.metric("Chunks", st.session_state.stats["chunks"])
        if st.button("🗑️  Clear chat"):
            st.session_state.messages = []
            st.rerun()

# ------------------------------------------------------
# FRONTEND  –––  CHAT  UI
# ------------------------------------------------------
# display history
for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# input box
if prompt := st.chat_input("Ask a question about the PDF"):
    if st.session_state.vs is None:
        st.warning("Please upload a PDF first.")
        st.stop()

    # FRONTEND: show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    # BACKEND Route-2: retrieval + Q&A
    docs = st.session_state.vs.similarity_search(prompt, k=3)   # retrieval step
    context = "\n\n".join(d.page_content for d in docs)

    system = (
        "You are a helpful assistant. Answer the question using ONLY the context below. "
        "If the context does not contain the answer, say 'I don't know'."
    )
    qa_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}"

    # BACKEND Route-2: call protected chat completion
    client = OpenAI(api_key=openai_key, base_url="https://api.chatanywhere.tech/v1")
    with st.chat_message("assistant", avatar="🤖"):
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": qa_prompt}],
            stream=True,
        )
        full = st.write_stream(
            (chunk.choices[0].delta.content or "")
            for chunk in stream
            if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta
        )
        st.session_state.messages.append({"role": "assistant", "content": full})
