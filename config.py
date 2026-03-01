"""
Central configuration for CloudSufi Document Q&A.

All tunable parameters live here. Scaling changes — swapping LLM provider,
vector store, or embedding model — require modifying only this file.

Environment variables are read once at import time via load_dotenv().
GROQ_API_KEY is exposed as a typed constant so services import it from
config rather than reading os.environ directly (avoids hidden dependencies).
"""

import os
from dotenv import load_dotenv

# Load .env file into os.environ before reading any values.
# Has no effect if variables are already set in the shell environment,
# which means CI/CD and Docker environments work without a .env file.
load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
# Exposed as a config constant so all services access keys through one module.
# Validation (non-empty check) is handled in main.py for a user-friendly
# Streamlit error rather than a raw Python exception at import time.
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# ── LLM ──────────────────────────────────────────────────────────────────────
# Groq provides free, high-speed inference (~300 tok/s) with LLaMA 3.3 70B.
# Scaling path: swap LLM_MODEL or switch provider — LangChain abstraction
# means generation_service.py requires zero changes.
LLM_MODEL: str = "llama-3.3-70b-versatile"
LLM_TEMPERATURE: float = 0.1   # Low temperature for factual, grounded answers
LLM_MAX_TOKENS: int = 1024

# ── Embeddings ────────────────────────────────────────────────────────────────
# Local sentence-transformer: no API key, no cost, no latency overhead.
# Scaling path: swap to "text-embedding-3-small" (OpenAI) or
# "textembedding-gecko" (GCP Vertex AI) — one constant change here.
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# ── Vector Store ──────────────────────────────────────────────────────────────
# ChromaDB persists to disk, surviving app restarts — unlike FAISS (in-memory).
# Scaling path: point CHROMA_PERSIST_DIR to a ChromaDB HTTP server container,
# or swap to Pinecone via LangChain's vectorstore interface.
CHROMA_PERSIST_DIR: str = "./vectorstore/chroma"
CHROMA_COLLECTION_NAME: str = "child_chunks"

# ── Parent Document Store ─────────────────────────────────────────────────────
# Full parent section Documents are serialized separately from the vector store.
# Scaling path: replace pickle with SQLite (via SQLAlchemy) for concurrent access.
PARENT_STORE_PATH: str = "./vectorstore/parent_store.pkl"

# ── Structure-Aware Chunking ──────────────────────────────────────────────────
# Lines with font size >= (median_body_size * HEADING_SIZE_MULTIPLIER) = heading.
# Multiplier-based threshold generalizes across PDFs with different base font sizes.
HEADING_SIZE_MULTIPLIER: float = 1.2
HEADING_MAX_CHARS: int = 150     # Headings are short; long lines are body text

# Child chunk parameters for RecursiveCharacterTextSplitter
CHILD_CHUNK_SIZE: int = 300
CHILD_CHUNK_OVERLAP: int = 40

# ── Retrieval ─────────────────────────────────────────────────────────────────
# Top-K child chunks retrieved; deduplication reduces these to unique parent sections.
# Scaling path: increase TOP_K_CHILDREN to 50 and add a cross-encoder reranker
# (e.g., cross-encoder/ms-marco-MiniLM-L-6-v2) at 500+ document scale.
TOP_K_CHILDREN: int = 10

# Hybrid retrieval weights for BM25 + dense vector EnsembleRetriever.
# BM25 handles keyword-precise queries; dense handles semantic queries.
# Scaling path: tune based on query distribution in production.
BM25_WEIGHT: float = 0.4

