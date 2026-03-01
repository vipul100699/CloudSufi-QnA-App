"""
Shared embedding function with lazy initialization and caching.
Defined once and imported by both ingestion and retrieval services
to ensure the model is loaded only once per process lifecycle.
"""

from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
import config


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Returns a cached HuggingFace embedding model instance.
    lru_cache(maxsize=1) ensures the ~80MB model is loaded only once.
    """
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
