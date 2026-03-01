"""
Tests for services/retrieval_service.py — covers store loading (both formats),
ensemble retriever construction, full retrieve_context() pipeline,
deduplication logic, and legacy fallback behaviour.
"""

import pickle
import pytest
from unittest.mock import patch, MagicMock
from langchain.schema import Document

from services.retrieval_service import (
    _load_stores,
    _build_ensemble_retriever,
    retrieve_context,
)
from tests.constants import PARENT_ID_A, PARENT_ID_B


# ── _load_stores ──────────────────────────────────────────────────────────────

class TestLoadStores:
    def test_new_format_returns_parent_map_and_child_docs(self, pickle_store, monkeypatch):
        monkeypatch.setattr("config.PARENT_STORE_PATH", pickle_store)
        parent_map, child_docs = _load_stores()
        assert PARENT_ID_A in parent_map
        assert len(child_docs) == 1

    def test_legacy_format_returns_empty_child_docs(self, legacy_pickle_store, monkeypatch):
        monkeypatch.setattr("config.PARENT_STORE_PATH", legacy_pickle_store)
        parent_map, child_docs = _load_stores()
        assert PARENT_ID_A in parent_map
        assert child_docs == []

    def test_parent_map_values_are_documents(self, pickle_store, monkeypatch):
        monkeypatch.setattr("config.PARENT_STORE_PATH", pickle_store)
        parent_map, _ = _load_stores()
        assert all(isinstance(v, Document) for v in parent_map.values())

    @patch("services.retrieval_service.get_embeddings")
    @patch("services.retrieval_service.Chroma")
    def test_load_vectorstore_returns_chroma_instance(self, mock_chroma, mock_emb):
        """
        Directly exercises _load_vectorstore() — covers retrieval_service.py line 49.
        This line is unreachable via retrieve_context() tests because _load_vectorstore
        is always patched at the boundary in those tests.
        """
        from services.retrieval_service import _load_vectorstore
        import config
        result = _load_vectorstore()
        mock_chroma.assert_called_once_with(
            collection_name=config.CHROMA_COLLECTION_NAME,
            embedding_function=mock_emb.return_value,
            persist_directory=config.CHROMA_PERSIST_DIR,
        )
        assert result == mock_chroma.return_value


# ── _build_ensemble_retriever ─────────────────────────────────────────────────

class TestBuildEnsembleRetriever:
    @patch("services.retrieval_service.EnsembleRetriever")
    @patch("services.retrieval_service.BM25Retriever")
    def test_returns_ensemble_retriever_instance(self, mock_bm25, mock_ensemble, child_doc_a):
        mock_bm25.from_documents.return_value = MagicMock()
        mock_ensemble.return_value = MagicMock()
        mock_vs = MagicMock()
        result = _build_ensemble_retriever([child_doc_a], mock_vs)
        assert result == mock_ensemble.return_value

    @patch("services.retrieval_service.EnsembleRetriever")
    @patch("services.retrieval_service.BM25Retriever")
    def test_weights_sum_to_one(self, mock_bm25, mock_ensemble, child_doc_a):
        mock_bm25.from_documents.return_value = MagicMock()
        mock_vs = MagicMock()
        _build_ensemble_retriever([child_doc_a], mock_vs)
        weights = mock_ensemble.call_args.kwargs["weights"]
        assert pytest.approx(sum(weights), abs=1e-6) == 1.0

    @patch("services.retrieval_service.EnsembleRetriever")
    @patch("services.retrieval_service.BM25Retriever")
    def test_bm25_weight_matches_config(self, mock_bm25, mock_ensemble, child_doc_a):
        import config
        mock_bm25.from_documents.return_value = MagicMock()
        mock_vs = MagicMock()
        _build_ensemble_retriever([child_doc_a], mock_vs)
        weights = mock_ensemble.call_args.kwargs["weights"]
        assert pytest.approx(weights[0], abs=1e-6) == config.BM25_WEIGHT

    @patch("services.retrieval_service.EnsembleRetriever")
    @patch("services.retrieval_service.BM25Retriever")
    def test_bm25_called_with_correct_k(self, mock_bm25, mock_ensemble, child_doc_a):
        import config
        mock_vs = MagicMock()
        _build_ensemble_retriever([child_doc_a], mock_vs)
        assert mock_bm25.from_documents.call_args.kwargs["k"] == config.TOP_K_CHILDREN


# ── retrieve_context ──────────────────────────────────────────────────────────

class TestRetrieveContext:

    def _setup_patches(self, parent_map, child_docs, child_results):
        """Returns a context manager that patches all three internal functions."""
        mock_vs = MagicMock()
        mock_ensemble = MagicMock()
        mock_ensemble.invoke.return_value = child_results

        patches = [
            patch("services.retrieval_service._load_vectorstore", return_value=mock_vs),
            patch("services.retrieval_service._load_stores", return_value=(parent_map, child_docs)),
            patch("services.retrieval_service._build_ensemble_retriever", return_value=mock_ensemble),
        ]
        return patches, mock_vs

    def test_returns_list_of_dicts(self, parent_doc_a, child_doc_a):
        patches, _ = self._setup_patches(
            {PARENT_ID_A: parent_doc_a}, [child_doc_a], [child_doc_a]
        )
        with patches[0], patches[1], patches[2]:
            result = retrieve_context("test query")
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_result_contains_required_keys(self, parent_doc_a, child_doc_a):
        patches, _ = self._setup_patches(
            {PARENT_ID_A: parent_doc_a}, [child_doc_a], [child_doc_a]
        )
        with patches[0], patches[1], patches[2]:
            result = retrieve_context("test query")
        assert {"content", "source", "page", "section"} <= set(result[0].keys())

    def test_deduplicates_children_with_same_parent_id(self, parent_doc_a, child_doc_a):
        """Two child chunks from the same parent → only one parent context returned."""
        child_doc_a_copy = Document(
            page_content="Another child of same parent.",
            metadata={**child_doc_a.metadata, "child_index": 1},
        )
        patches, _ = self._setup_patches(
            {PARENT_ID_A: parent_doc_a},
            [child_doc_a, child_doc_a_copy],
            [child_doc_a, child_doc_a_copy],
        )
        with patches[0], patches[1], patches[2]:
            result = retrieve_context("test query")
        assert len(result) == 1

    def test_returns_parent_content_not_child_content(self, parent_doc_a, child_doc_a):
        patches, _ = self._setup_patches(
            {PARENT_ID_A: parent_doc_a}, [child_doc_a], [child_doc_a]
        )
        with patches[0], patches[1], patches[2]:
            result = retrieve_context("test query")
        assert result[0]["content"] == parent_doc_a.page_content

    def test_skips_child_with_missing_parent_id(self):
        child_no_id = Document(
            page_content="Orphan chunk.", metadata={"source": "x.pdf"}
        )
        patches, _ = self._setup_patches({}, [], [child_no_id])
        with patches[0], patches[1], patches[2]:
            result = retrieve_context("test query")
        assert result == []

    def test_skips_child_whose_parent_not_in_store(self, child_doc_a):
        patches, _ = self._setup_patches({}, [child_doc_a], [child_doc_a])
        with patches[0], patches[1], patches[2]:
            result = retrieve_context("test query")
        assert result == []

    def test_legacy_fallback_uses_similarity_search(self, parent_doc_a, child_doc_a):
        """When child_docs is empty (legacy pickle), vector similarity_search is used."""
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = [child_doc_a]
        with patch("services.retrieval_service._load_vectorstore", return_value=mock_vs), \
             patch("services.retrieval_service._load_stores",
                   return_value=({PARENT_ID_A: parent_doc_a}, [])):
            result = retrieve_context("legacy query")
        mock_vs.similarity_search.assert_called_once()
        assert len(result) == 1

    def test_citation_metadata_matches_child_metadata(self, parent_doc_a, child_doc_a):
        patches, _ = self._setup_patches(
            {PARENT_ID_A: parent_doc_a}, [child_doc_a], [child_doc_a]
        )
        with patches[0], patches[1], patches[2]:
            result = retrieve_context("test query")
        assert result[0]["source"] == child_doc_a.metadata["source"]
        assert result[0]["section"] == child_doc_a.metadata["section"]
        assert result[0]["page"] == str(child_doc_a.metadata["page"])
