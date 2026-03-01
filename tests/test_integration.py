"""
Integration tests — validates the full Ingest → Retrieve → Generate pipeline
with all external APIs mocked (Chroma, HuggingFace, Groq) but all service
code running unmodified. These tests catch wiring bugs that unit tests miss.
"""

import pickle
import pytest
from unittest.mock import patch, MagicMock
from langchain.schema import Document

from services.ingestion_service import ingest_pdfs
from services.retrieval_service import retrieve_context
from services.generation_service import generate_answer


@pytest.fixture
def ingested_store(sample_pdf, tmp_path, monkeypatch):
    """
    Runs the full ingestion pipeline on a real PDF (with Chroma mocked),
    returns the path to the written pickle store for downstream retrieval tests.
    """
    pkl_path = tmp_path / "vs" / "store.pkl"
    monkeypatch.setattr("config.CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr("config.PARENT_STORE_PATH", str(pkl_path))

    with patch("services.ingestion_service.Chroma"), \
         patch("services.ingestion_service.get_embeddings"):
        ingest_pdfs([sample_pdf])

    return str(pkl_path)


class TestFullPipeline:

    def test_ingestion_produces_valid_pickle(self, ingested_store):
        with open(ingested_store, "rb") as f:
            data = pickle.load(f)
        assert "parent_map" in data and "child_docs" in data
        assert len(data["parent_map"]) > 0
        assert len(data["child_docs"]) > 0

    def test_retrieve_returns_contexts_from_ingested_data(self, ingested_store, monkeypatch):
        monkeypatch.setattr("config.PARENT_STORE_PATH", ingested_store)
        mock_vs = MagicMock()

        with open(ingested_store, "rb") as f:
            data = pickle.load(f)

        child_docs = data["child_docs"]
        first_child = child_docs[0]
        mock_vs.similarity_search.return_value = [first_child]

        with patch("services.retrieval_service._load_vectorstore", return_value=mock_vs), \
             patch("services.retrieval_service.BM25Retriever") as mock_bm25, \
             patch("services.retrieval_service.EnsembleRetriever") as mock_ensemble:

            mock_ensemble_instance = MagicMock()
            mock_ensemble_instance.invoke.return_value = [first_child]
            mock_ensemble.return_value = mock_ensemble_instance
            mock_bm25.from_documents.return_value = MagicMock()

            contexts = retrieve_context("test section heading")

        assert len(contexts) > 0
        assert all(k in contexts[0] for k in ["content", "source", "page", "section"])

    def test_generate_returns_answer_from_contexts(self, sample_contexts):
        mock_response = MagicMock()
        mock_response.content = "The answer is grounded [1]."
        with patch("services.generation_service.ChatGroq") as mock_llm:
            mock_llm.return_value.invoke.return_value = mock_response
            result = generate_answer("What is this about?", sample_contexts)
        assert result["answer"] == "The answer is grounded [1]."
        assert "[1]" in result["context_used"]

    def test_empty_contexts_triggers_graceful_no_answer(self):
        result = generate_answer("A question with no relevant docs.", [])
        assert "No relevant content" in result["answer"]
        assert result["context_used"] == ""

    def test_citation_metadata_propagates_end_to_end(self, ingested_store, monkeypatch):
        """
        Full pipeline: ingestion writes pickle → retrieval reads it →
        citation metadata (source, section, page) flows correctly into contexts.
        """
        monkeypatch.setattr("config.PARENT_STORE_PATH", ingested_store)

        with open(ingested_store, "rb") as f:
            data = pickle.load(f)

        child_docs = data["child_docs"]
        mock_vs = MagicMock()

        with patch("services.retrieval_service._load_vectorstore", return_value=mock_vs), \
             patch("services.retrieval_service.BM25Retriever") as mock_bm25, \
             patch("services.retrieval_service.EnsembleRetriever") as mock_ensemble:

            mock_ensemble_instance = MagicMock()
            mock_ensemble_instance.invoke.return_value = child_docs[:2]
            mock_ensemble.return_value = mock_ensemble_instance
            mock_bm25.from_documents.return_value = MagicMock()

            contexts = retrieve_context("body text")

        for ctx in contexts:
            assert ctx["source"].endswith(".pdf")
            assert ctx["page"].isdigit()
            assert len(ctx["section"]) > 0
