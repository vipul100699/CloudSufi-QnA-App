"""
Shared pytest fixtures for the CloudSufi Document Q&A test suite.

All fixtures are function-scoped by default (isolated per test) unless
explicitly decorated with a broader scope. This prevents cross-test
state contamination — especially important for file-system fixtures
(tmp_path) and mock objects whose call counts are inspected.
"""

import os
import sys
import pickle
import pytest
from unittest.mock import MagicMock
from langchain.schema import Document
from tests.constants import PARENT_ID_A, PARENT_ID_B

# Ensure project root is on sys.path so `import config` and
# `from services.x import y` resolve correctly from any working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



# ── Document fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def parent_doc_a():
    """A parent Document simulating a JD skills section."""
    return Document(
        page_content=(
            "Machine Learning Libraries: Hands-on experience with PyTorch, "
            "Scikit-learn, Hugging Face Transformers, Diffusers, Librosa, "
            "LangGraph, and the OpenAI API."
        ),
        metadata={
            "source": "jd.pdf",
            "page": 1,
            "section": "Qualifications and Skills",
            "parent_id": PARENT_ID_A,
            "chunk_type": "parent",
        },
    )


@pytest.fixture
def parent_doc_b():
    """A parent Document simulating a company overview section."""
    return Document(
        page_content="CLOUDSUFI is a Google Cloud Premier Partner.",
        metadata={
            "source": "about.pdf",
            "page": 1,
            "section": "About Us",
            "parent_id": PARENT_ID_B,
            "chunk_type": "parent",
        },
    )


@pytest.fixture
def child_doc_a():
    """A child chunk derived from parent_doc_a."""
    return Document(
        page_content="PyTorch, Scikit-learn, Hugging Face Transformers.",
        metadata={
            "source": "jd.pdf",
            "page": 1,
            "section": "Qualifications and Skills",
            "parent_id": PARENT_ID_A,
            "chunk_type": "child",
            "child_index": 0,
        },
    )


@pytest.fixture
def child_doc_b():
    """A second child chunk with a different parent_id."""
    return Document(
        page_content="CLOUDSUFI is a Google Cloud Premier Partner.",
        metadata={
            "source": "about.pdf",
            "page": 1,
            "section": "About Us",
            "parent_id": PARENT_ID_B,
            "chunk_type": "child",
            "child_index": 0,
        },
    )


@pytest.fixture
def sample_contexts():
    """Two pre-formatted context dicts as returned by retrieve_context()."""
    return [
        {
            "content": "PyTorch, Scikit-learn, Hugging Face Transformers.",
            "source": "jd.pdf",
            "page": "1",
            "section": "Qualifications and Skills",
        },
        {
            "content": "CLOUDSUFI is a Google Cloud Premier Partner.",
            "source": "about.pdf",
            "page": "1",
            "section": "About Us",
        },
    ]


# ── Pickle store fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def pickle_store(tmp_path, parent_doc_a, child_doc_a):
    """
    Writes a new-format pickle store (dict with parent_map + child_docs)
    to a temp file and returns its path. Mirrors what ingest_pdfs() writes.
    """
    store = {
        "parent_map": {PARENT_ID_A: parent_doc_a},
        "child_docs": [child_doc_a],
    }
    path = tmp_path / "parent_store.pkl"
    with open(path, "wb") as f:
        pickle.dump(store, f)
    return str(path)


@pytest.fixture
def legacy_pickle_store(tmp_path, parent_doc_a):
    """
    Writes a legacy-format pickle store (plain dict, parent_map only)
    to a temp file. Used to test the backward-compatibility fallback.
    """
    path = tmp_path / "parent_store_legacy.pkl"
    with open(path, "wb") as f:
        pickle.dump({PARENT_ID_A: parent_doc_a}, f)
    return str(path)


# ── PDF fixture ───────────────────────────────────────────────────────────────

@pytest.fixture
def sample_pdf(tmp_path):
    """
    Creates a minimal real PDF using PyMuPDF with two distinct font sizes:
      - Heading text at 24pt (well above any threshold)
      - Body text at 10pt (many lines, drives median down)

    The 2.4× size ratio ensures heading detection is reliable regardless
    of minor font metric variations across platforms.
    """
    import fitz

    pdf_path = tmp_path / "test_document.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    # One large heading — must be detected as a section heading
    page.insert_text((50, 80), "Test Section Heading", fontsize=24)

    # Many body lines — their sizes dominate the median calculation,
    # keeping the threshold low enough that only the heading passes
    for i, y in enumerate(range(120, 500, 22)):
        page.insert_text((50, y), f"Body text sentence number {i}.", fontsize=10)

    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


# ── Mock embeddings fixture ───────────────────────────────────────────────────

@pytest.fixture
def mock_embeddings():
    """A lightweight mock HuggingFaceEmbeddings instance."""
    mock = MagicMock()
    mock.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    mock.embed_query.return_value = [0.1, 0.2, 0.3]
    return mock
