# 📄 PDF-Q&A Bot

**Live demo:** https://simple-chatbot-s2drqmeevnjwhojjxnbav3.streamlit.app/

## 🚀 What it does
1. Upload any PDF (≤ 200 MB).
2. Ask questions in natural language.
3. Get **instant answers** extracted **only from the uploaded document**.

## ✅ Internship Task Checklist
| Requirement | How it’s met |
|-------------|--------------|
| **Backend – Protected API route 1** | Server-side endpoint accepts PDF → extracts text → creates vector embeddings (open-source `all-MiniLM-L6-v2`) → stores in FAISS index. |
| **Backend – Protected API route 2** | Server-side endpoint receives question → retrieves top-3 chunks → sends context + question to OpenAI-compatible chat endpoint → streams answer. |
| **Frontend – Simple UI** | One-page app: drag-and-drop PDF uploader, chat-style question box, live answer bubbles. |
| **Calls protected backend** | All OpenAI calls use Streamlit Secrets (key never exposed to client). |

## 🛠️ Tech Stack
- **Framework**: Streamlit (Python)
- **Text extraction**: PyPDF2
- **Chunking**: LangChain `RecursiveCharacterTextSplitter`
- **Embeddings**: Hugging-Face `sentence-transformers/all-MiniLM-L6-v2` (free, offline)
- **Vector DB**: FAISS (in-memory, zero external infra)
- **Chat**: any OpenAI-compatible API

## ⚙️ 1-Minute Local Run
```bash
git clone https://github.com/krishnabalajiwork/simple-chatbot.git
cd simple-chatbot
pip install -r requirements.txt
streamlit run app.py
