"""
Streamlit Mini PDF-Q&A  (single-file edition)
Upload a PDF → ask questions → answers come ONLY from the PDF.
Uses an in-memory FAISS index → no external DB → free & instant.
"""
import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import tiktoken

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="PDF-Q&A Bot", page_icon="📄")
st.title("📄 Mini PDF-Q&A Bot")
st.markdown("Upload a PDF, then ask questions **about its content**.")

# ---------- AUTH ----------
try:
    openai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("🔑 Please set OPENAI_API_KEY in Streamlit secrets.")
    st.stop()

# ---------- UTILS ----------
@st.cache_data(show_spinner=False)
def parse_pdf(file) -> str:
    """Return raw text of whole PDF."""
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def get_embeddings():
    """LangChain helper that talks to the same OpenAI key."""
    return OpenAIEmbeddings(client=openai, tiktoken_model_name="cl100k_base")

def build_vectorstore(text: str):
    """Split text → embed → return in-memory FAISS."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = splitter.split_text(text)
    return FAISS.from_texts(chunks, get_embeddings())

# ---------- SESSION ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vs" not in st.session_state:
    st.session_state.vs = None

# ---------- SIDEBAR ----------
with st.sidebar:
    uploaded = st.file_uploader("1. Upload PDF", type="pdf")
    if uploaded and st.session_state.vs is None:
        with st.spinner("Parsing & embedding…"):
            raw_text = parse_pdf(uploaded)
            st.session_state.vs = build_vectorstore(raw_text)
            st.success("PDF indexed!")

# ---------- CHAT ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about the PDF"):
    if st.session_state.vs is None:
        st.warning("Please upload a PDF first.")
        st.stop()

    # 1. show user msg
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. retrieve top-3 chunks
    docs = st.session_state.vs.similarity_search(prompt, k=3)
    context = "\n\n".join(d.page_content for d in docs)

    # 3. build retrieval-augmented prompt
    system = (
        "You are a helpful assistant. Answer the question using ONLY the context below. "
        "If the context does not contain the answer, say 'I don't know'."
    )
    qa_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}"

    # 4. call ChatCompletion
    with st.chat_message("assistant"):
        stream = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": qa_prompt},
            ],
            stream=True,
        )
        full = st.write_stream(chunk.choices[0].delta.content or "" for chunk in stream)
        st.session_state.messages.append({"role": "assistant", "content": full})
