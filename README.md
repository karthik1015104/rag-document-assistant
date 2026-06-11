# RAG Document Assistant

A production-grade Retrieval-Augmented Generation (RAG) system that enables conversational question-answering over custom PDF documents, with source attribution and page citations.

## Overview

Large Language Models hallucinate when queried on domain-specific knowledge outside their training data. This system solves that by grounding every response in a user-provided PDF knowledge base using semantic retrieval — delivering accurate, cited answers without any model fine-tuning.

## Features

- **PDF ingestion & semantic chunking** — Recursive character splitting with 200-token overlap to preserve context across chunk boundaries
- **FAISS vector indexing** — Sub-second similarity search over embedded document chunks at scale
- **HuggingFace embeddings** — `all-MiniLM-L6-v2` for dense semantic representations, fully local and free
- **Groq LLM (Llama 3.3 70B)** — Fast, accurate answer generation strictly grounded in retrieved context
- **Conversational memory** — Multi-turn dialogue with full chat history maintained across the session
- **Source attribution** — Every answer cites exact page numbers from the source document
- **Streamlit UI** — Clean chat interface with expandable source viewer per response

## Tech Stack

| Component | Tool |
|---|---|
| PDF Parsing | `pypdf`, `langchain-community` |
| Chunking | `RecursiveCharacterTextSplitter` (chunk=1000, overlap=200) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| Vector Store | `FAISS` (Facebook AI Similarity Search) |
| LLM | `Groq API — llama-3.3-70b-versatile` |
| Chain | `LangChain LCEL` (LangChain Expression Language) |
| Memory | `ConversationBufferMemory` via LangChain message history |
| UI | `Streamlit` |

## Project Structure

```
rag-document-assistant/
├── rag_system.ipynb       # Main notebook — PDF ingestion → FAISS indexing → chain → app
├── app.py                 # Streamlit application
├── requirements.txt       # Python dependencies
├── faiss_index/           # Saved FAISS vector index (auto-generated on first run)
│   ├── index.faiss
│   └── index.pkl
└── README.md
```

## How to Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/rag-document-assistant.git
cd rag-document-assistant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your Groq API key
Get a free key at [console.groq.com](https://console.groq.com), then:
```bash
export GROQ_API_KEY=your_groq_api_key_here
```

### 4. Build the FAISS index
Open `rag_system.ipynb` and run Cells 1–4 to ingest your PDF and build the vector index.

### 5. Launch the app
```bash
streamlit run app.py
```

## Results

- Indexed a 604-page metallurgy textbook into **1978 semantic chunks**
- Sub-second FAISS retrieval latency across the full index
- Accurate, citation-grounded answers with page-level source attribution
- Zero hallucination on in-document queries — responses strictly bounded by retrieved context

## Architecture

```
User Query
    ↓
[Streamlit Chat UI]
    ↓
[Conversation Memory] ←→ [LangChain LCEL Chain]
                                ↓              ↓
                      [FAISS Retriever]   [Groq Llama 3.3 70B]
                                ↓
                      [HuggingFace Embeddings]
                                ↑
                      [PDF Ingestion + Semantic Chunking]
                                ↑
                      [User-uploaded PDF files]
```

## Author

Karthik | IIT Madras — Metallurgical & Materials Engineering
