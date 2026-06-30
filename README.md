# AI Document Assistant (RAG)

An AI-powered Document Assistant built with Django that enables semantic question answering over uploaded PDF documents using Retrieval-Augmented Generation (RAG).

## Features

- Upload and manage PDF documents
- Extract text using PyMuPDF
- Chunk documents with overlapping windows
- Generate semantic embeddings using Sentence Transformers
- Store embeddings persistently in SQLite
- Perform fast similarity search with FAISS
- Retrieve relevant context for user queries
- Generate grounded responses using the Gemini API
- Incrementally update and delete indexed documents using content hash-based change detection

## Tech Stack

- Django
- PyMuPDF
- Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS
- SQLite
- Google Gemini API
- NumPy

## Architecture

```
PDF Upload
      │
      ▼
 Text Extraction
      │
      ▼
 Text Chunking
      │
      ▼
 Embedding Generation
      │
      ▼
SQLite Embedding Storage
      │
      ▼
 FAISS Vector Index
      │
      ▼
Semantic Retrieval
      │
      ▼
 Gemini
      │
      ▼
Answer
```

## Running the Project

```bash
git clone <repository-url>
cd <project-folder>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

python manage.py migrate

python manage.py runserver
```

