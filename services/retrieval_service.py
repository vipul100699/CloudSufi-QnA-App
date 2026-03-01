"""
Retrieval layer implementing Parent-Child (Small-to-Big) retrieval.

Flow:
  1. Embed the user query using the same model as ingestion.
  2. Search ChromaDB for top-K similar CHILD chunks (high precision).
  3. Extract parent_id from each child's metadata.
  4. Fetch the full PARENT section from the pickle store (rich context).
  5. Deduplicate: if multiple children share a parent, return the parent once.
  6. Return a list of context dicts with content + citation metadata.

Why this beats flat retrieval:
  - Child chunks are small → precise semantic match to query.
  - Parent sections are large → full context for the LLM to generate from.
  - Citations are section-level, not arbitrary chunk-level.
"""

import pickle
from typing import List, Dict

from langchain_chroma import Chroma

import config
from services.embeddings import get_embeddings


def _load_vectorstore() -> Chroma:
    """Loads the persisted ChromaDB collection for similarity search."""
    return Chroma(
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=config.CHROMA_PERSIST_DIR
    )


def _load_parent_store() -> Dict:
    """Deserializes the parent document store from disk."""
    with open(config.PARENT_STORE_PATH, "rb") as f:
        return pickle.load(f)


def retrieve_context(query: str) -> List[Dict[str, str]]:
    """
    Retrieves relevant context for a query using hierarchical parent-child retrieval.

    Args:
        query: Natural language question from the user.

    Returns:
        A list of context dicts (deduplicated by parent section), each containing:
          content : Full parent section text — passed to LLM as context.
          source  : PDF filename — used for citation.
          page    : Page number — used for citation.
          section : Section heading — used for citation.
    """
    vectorstore = _load_vectorstore()
    parent_store = _load_parent_store()

    # Step 1: Retrieve top-K child chunks by semantic similarity
    child_results = vectorstore.similarity_search(query, k=config.TOP_K_CHILDREN)

    # Step 2: Resolve child → parent, deduplicating by parent_id
    seen_parent_ids = set()
    contexts: List[Dict[str, str]] = []

    for child in child_results:
        parent_id = child.metadata.get("parent_id")

        if not parent_id or parent_id in seen_parent_ids:
            continue

        parent_doc = parent_store.get(parent_id)
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
