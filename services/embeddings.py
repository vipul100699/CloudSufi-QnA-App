"""
Shared embedding function with lazy initialization and Streamlit-aware caching.

Key design decisions:
  - HuggingFaceEmbeddings is imported INSIDE the function (lazy import), not at
    module level. This prevents the ~3-5s PyTorch + transformers stack initialization
    from blocking Streamlit's UI render on startup.
  - @st.cache_resource replaces @lru_cache for Streamlit compatibility:
      * Persists the model across all user sessions and page reruns.
      * Shows a native Streamlit spinner ("Loading embedding model...") on first load.
      * Thread-safe — Streamlit handles concurrent access automatically.
      * lru_cache is process-scoped but Streamlit-unaware; cache_resource integrates
        with Streamlit's execution model correctly.
"""

import streamlit as st
import config


@st.cache_resource(show_spinner="⏳ Loading embedding model (first run only)...")
def get_embeddings():
    """
    Returns a cached HuggingFace embedding model instance.

    On first call: downloads all-MiniLM-L6-v2 (~80MB) if not cached locally,
    then loads it into memory. Subsequent calls return the cached instance instantly.

    The lazy import of HuggingFaceEmbeddings ensures sentence-transformers and
    PyTorch are only imported when this function is first invoked — not at module
    import time — keeping Streamlit startup fast.
    """
    # Lazy import: defers the heavy sentence-transformers + PyTorch initialization
    # until the embedding model is actually needed (first user interaction),
    # not when Python loads the services package on app startup.
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
