"""
Streamlit Mini PDF-Q&A  (single-file edition)
Upload a PDF → ask questions → answers come ONLY from the PDF.
Uses an in-memory FAISS index → no external DB → free & instant.
"""
import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

# ---------- PAGE ----------
st.set_page_config(page_title="PDF-Q&A Bot", page_icon="📄")
st.title("📄 Mini PDF-Q&A Bot")
st.markdown("Upload a PDF, then ask questions **about its content**.")

# ---------- AUTH ----------
try:
    openai_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    st.error("🔑 Please set OPENAI_API_KEY in Streamlit secrets."); st.stop()

# ---------- UTILS ----------
@st.cache_data(show_spinner=False)
def parse_pdf(file):
    reader = PdfReader(file)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return text, len(reader.pages)

def get_embeddings():
    # use official endpoint for embeddings (chat-anywhere proxy lacks /embeddings)
    return OpenAIEmbeddings(
        openai_api_key=openai_key,
        api_base="https://api.openai.com/v1",
        model="text-embedding-3-small",
    )

@st.cache_data(show_spinner=False)
def build_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    vs = FAISS.from_texts(chunks, get_embeddings())
    return vs, len(chunks)

# ---------- SESSION ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vs" not in st.session_state:
    st.session_state.vs = None
if "stats" not in st.session_state:
    st.session_state.stats = {}

# ---------- SIDEBAR ----------
with st.sidebar:
    uploaded = st.file_uploader("1. Upload PDF", type="pdf")
    if uploaded and st.session_state.vs is None:
        with st.spinner("Parsing & embedding…"):
            raw_text, pages = parse_pdf(uploaded)
            vs, chunks = build_vectorstore(raw_text)
            st.session_state.vs = vs
            st.session_state.stats = {"pages": pages, "chunks": chunks}
        st.success("PDF indexed!")

    if st.session_state.vs:
        st.metric("Pages", st.session_state.stats["pages"])
        st.metric("Chunks", st.session_state.stats["chunks"])
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

# ---------- CHAT ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about the PDF"):
    if st.session_state.vs is None:
        st.warning("Please upload a PDF first."); st.stop()

    # user msg
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # retrieval
    docs = st.session_state.vs.similarity_search(prompt, k=3)
    context = "\n\n".join(d.page_content for d in docs)

    # qa prompt
    system = (
        "You are a helpful assistant. Answer the question using ONLY the context below. "
        "If the context does not contain the answer, say 'I don't know'."
    )
    qa_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}"

    # stream answer
    client = OpenAI(api_key=openai_key)
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": qa_prompt}],
            stream=True,
        )
        full = st.write_stream(chunk.choices[0].delta.content or "" for chunk in stream)
        st.session_state.messages.append({"role": "assistant", "content": full})
