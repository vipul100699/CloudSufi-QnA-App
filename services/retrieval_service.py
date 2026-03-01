"""
Retrieval layer implementing Hybrid Parent-Child retrieval.

Retrieval Strategy — Hybrid BM25 + Dense Vector (Ensemble):
  Problem with pure vector search: semantically overloaded queries (e.g., "machine
  learning libraries") retrieve semantically similar but lexically irrelevant chunks
  (e.g., "MLOps experience") while missing exact keyword matches ("PyTorch, Scikit-learn").

  Solution: Reciprocal Rank Fusion (RRF) ensemble of two retrievers:
    - BM25 (sparse):  Exact keyword match — catches "Machine Learning Libraries: PyTorch..."
    - Dense (vector): Semantic similarity — catches conceptually related chunks
    - Weights: 40% BM25 + 60% Dense. Tunable via config.BM25_WEIGHT.

  Both retrievers operate on CHILD chunks (small, precise units).
  Retrieved child chunks are resolved to their PARENT sections (rich context)
  for LLM generation — the core of Small-to-Big hierarchical retrieval.

Flow:
  1. Load persisted ChromaDB (dense retriever) and pickle store (parent docs + child docs).
  2. Reconstruct BM25 index from persisted child Documents.
  3. Build EnsembleRetriever (BM25 + dense) with RRF fusion.
  4. Invoke ensemble retriever with user query → ranked child chunks.
  5. Resolve each child's parent_id → fetch full parent section.
  6. Deduplicate by parent_id → return context list with citation metadata.
"""

import pickle
from typing import List, Dict

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document

import config
from services.embeddings import get_embeddings


# ── Private Loaders ───────────────────────────────────────────────────────────

def _load_vectorstore() -> Chroma:
    """
    Loads the persisted ChromaDB vector store collection.

    Returns:
        A Chroma instance connected to the persisted collection on disk,
        ready for dense similarity search.
    """
    return Chroma(
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=config.CHROMA_PERSIST_DIR
    )


def _load_stores() -> tuple[Dict, List[Document]]:
    """
    Deserializes both the parent document store and the child document list
    from the single pickle file written during ingestion.

    Returns:
        A tuple of:
          parent_map  : Dict[parent_id → parent Document] for context resolution.
          child_docs  : List of all child Documents for BM25 index reconstruction.

    Note:
        BM25 is a stateless index built from a document list — it does not persist
        to disk like ChromaDB. It must be reconstructed at each retrieval call.
        At the scale of 1–3 PDFs (~50 child chunks), this reconstruction takes
        under 10ms and has no meaningful performance impact.
    """
    with open(config.PARENT_STORE_PATH, "rb") as f:
        data = pickle.load(f)

    # Support both old format (parent_map only) and new format (dict with both stores)
    if isinstance(data, dict) and "parent_map" in data:
        return data["parent_map"], data["child_docs"]
    else:
        # Legacy format: data is parent_map only, no child_docs available for BM25.
        # Fall back gracefully — BM25 will be skipped, pure vector search used.
        return data, []


def _build_ensemble_retriever(
    child_docs: List[Document],
    vectorstore: Chroma
) -> EnsembleRetriever:
    """
    Constructs a hybrid BM25 + dense vector EnsembleRetriever with RRF fusion.

    Why hybrid:
      Pure vector search fails on keyword-precise queries (e.g., "PyTorch, Scikit-learn")
      because specific proper nouns contribute low weight in semantic embedding space.
      BM25 exact-match scoring recovers these cases with high precision.

    Args:
        child_docs  : All child Document chunks — used to build the BM25 sparse index.
        vectorstore : Persisted ChromaDB instance — used as the dense retriever.

    Returns:
        EnsembleRetriever combining BM25 (40%) and dense vector (60%) with RRF fusion.
        Weights are tunable via config.BM25_WEIGHT.
    """
    bm25_retriever = BM25Retriever.from_documents(
        child_docs,
        k=config.TOP_K_CHILDREN
    )
    dense_retriever = vectorstore.as_retriever(
        search_kwargs={"k": config.TOP_K_CHILDREN}
    )
    return EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[config.BM25_WEIGHT, 1.0 - config.BM25_WEIGHT]
    )


# ── Public Retrieval Interface ────────────────────────────────────────────────

def retrieve_context(query: str) -> List[Dict[str, str]]:
    """
    Retrieves relevant context for a user query using Hybrid Parent-Child retrieval.

    Uses a BM25 + dense vector ensemble to retrieve top-K child chunks, then
    resolves each to its parent section for rich LLM context. Results are
    deduplicated so each parent section appears at most once.

    Args:
        query: Natural language question from the user.

    Returns:
        A deduplicated list of context dicts, each containing:
          content : Full parent section text — passed to the LLM as context.
          source  : PDF filename — used for citation.
          page    : Page number where the section begins — used for citation.
          section : Section heading title — used for citation.

    Raises:
        FileNotFoundError: If the vector store or parent store has not been
                           initialized (i.e., no documents have been ingested yet).
    """
    vectorstore = _load_vectorstore()
    parent_map, child_docs = _load_stores()

    # Use hybrid retrieval if child_docs are available (new ingestion format),
    # otherwise fall back to pure vector search (legacy format compatibility).
    if child_docs:
        ensemble = _build_ensemble_retriever(child_docs, vectorstore)
        child_results = ensemble.invoke(query)
    else:
        # Legacy fallback — pure vector search
        child_results = vectorstore.similarity_search(query, k=config.TOP_K_CHILDREN)

    # Resolve child chunks → parent sections, deduplicated by parent_id
    seen_parent_ids = set()
    contexts: List[Dict[str, str]] = []

    for child in child_results:
        parent_id = child.metadata.get("parent_id")

        if not parent_id or parent_id in seen_parent_ids:
            continue

        parent_doc = parent_map.get(parent_id)
        if not parent_doc:
            continue

        seen_parent_ids.add(parent_id)
        contexts.append({
            "content": parent_doc.page_content,
            "source": child.metadata.get("source", "Unknown Document"),
            "page": str(child.metadata.get("page", "N/A")),
            "section": child.metadata.get("section", "Unknown Section")
        })

    return contexts
