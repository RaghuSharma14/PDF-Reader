# Semantic Document Q&A System — RAG Pipeline

An AI-powered document intelligence application that enables users to upload PDF files and query them using natural language. Built on a Retrieval-Augmented Generation (RAG) architecture using LangChain, FAISS, and Hugging Face models, with a clean Streamlit interface for non-technical users.

🔗 **Live Demo:** [Try it here](https://raghusharma14-pdf-reader-app-62huam.streamlit.app/)

---

## How It Works

1. User uploads a PDF document
2. The document is parsed and split into chunks using PyPDFLoader
3. Each chunk is converted into a vector embedding using Hugging Face sentence transformers
4. Embeddings are stored in a FAISS vector index for fast similarity search
5. When a question is asked, the system retrieves the most relevant chunks
6. A QA model (deepset/roberta-base-squad2) generates an answer from the retrieved context

---

## Features

- Natural language querying over any uploaded PDF
- Semantic similarity search using FAISS vector store
- Special query handling for structured sections (Skills, Education, Projects)
- Deployed as a live web app via Streamlit Cloud
- Runs on CPU — no GPU required

---

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| RAG Framework | LangChain |
| Vector Store | FAISS |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| QA Model | deepset/roberta-base-squad2 |
| PDF Parsing | PyPDF2, PyPDFLoader |
| Language | Python |

---

## Project Structure

```
PDF-Reader/
├── app.py              # Streamlit frontend + QA logic
├── rag_engine.py       # RAG pipeline (vector store + retrieval chain)
├── requirements.txt    # Dependencies
```

---

## Setup & Installation

```bash
git clone https://github.com/RaghuSharma14/PDF-Reader.git
cd PDF-Reader
pip install -r requirements.txt
streamlit run app.py
```

---

## Author

**Raghu Sharma**
B.Tech CSE (AI/ML) — Maharaja Surajmal Institute of Technology, Delhi
raghusharma70876@gmail.com