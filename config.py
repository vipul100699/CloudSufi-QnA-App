"""
Central configuration for CloudSufi Document Q&A.
All tunable parameters are defined here so that scaling changes
(e.g., swapping LLM, vector store, or embedding model) require
modifying only this file — not any business logic.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────
# Groq provides free, high-speed inference (~300 tok/s) with LLaMA 3.3 70B.
# Scaling path: swap LLM_MODEL or switch provider entirely via LangChain abstraction.
LLM_MODEL: str = "llama-3.3-70b-versatile"
LLM_TEMPERATURE: float = 0.1   # Low temperature for factual, grounded answers
LLM_MAX_TOKENS: int = 1024

# ── Embeddings ────────────────────────────────────────────────────────────────
# Local sentence-transformer: no API key, no cost, no latency overhead.
# Scaling path: swap to "text-embedding-3-small" (OpenAI) or
# "textembedding-gecko" (GCP Vertex AI) — one constant change.
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# ── Vector Store ──────────────────────────────────────────────────────────────
# ChromaDB persists to disk, surviving app restarts — unlike FAISS (in-memory).
# Scaling path: switch to ChromaDB HTTP server or Pinecone via LangChain interface.
CHROMA_PERSIST_DIR: str = "./vectorstore/chroma"
CHROMA_COLLECTION_NAME: str = "child_chunks"

# ── Parent Document Store ─────────────────────────────────────────────────────
# Parent section docs (full text) persisted separately for context retrieval.
PARENT_STORE_PATH: str = "./vectorstore/parent_store.pkl"

# ── Structure-Aware Chunking ──────────────────────────────────────────────────
# Text with font size >= (median_body_size * HEADING_SIZE_MULTIPLIER) = heading.
HEADING_SIZE_MULTIPLIER: float = 1.2
HEADING_MAX_CHARS: int = 150     # Headings are short; long lines are body text

# Child chunk parameters for RecursiveCharacterTextSplitter
CHILD_CHUNK_SIZE: int = 300
CHILD_CHUNK_OVERLAP: int = 40

# ── Retrieval ─────────────────────────────────────────────────────────────────
# Top-K child chunks retrieved; deduplication reduces these to unique parent sections.
# Scaling path: increase TOP_K_CHILDREN and add a cross-encoder reranker
# (e.g., cross-encoder/ms-marco-MiniLM-L-6-v2) at 500+ document scale.
TOP_K_CHILDREN: int = 6
