"""
Tests for services/ingestion_service.py — covers heading detection,
section extraction, child chunking, pipeline orchestration, and
filesystem utility functions.
"""

import os
import pickle
import pytest
from unittest.mock import patch, MagicMock, call
from langchain.schema import Document

from services.ingestion_service import (
    _compute_heading_threshold,
    _classify_line_as_heading,
    _create_child_chunks,
    extract_structured_sections,
    ingest_pdfs,
    vectorstore_exists,
    clear_vectorstore,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mock_doc(pages_spans):
    """
    Builds a mock fitz.Document from a list of (text, size) tuples per page.
    Each call to __iter__ yields a fresh iterator to support multiple iterations.
    """
    mock_pages = []
    for spans in pages_spans:
        blocks = [
            {
                "type": 0,
                "lines": [{"spans": [{"text": t, "size": s}]}],
            }
            for t, s in spans
        ]
        mock_page = MagicMock()
        mock_page.get_text.return_value = {"blocks": blocks}
        mock_pages.append(mock_page)

    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(side_effect=lambda: iter(mock_pages))
    return mock_doc


# ── _compute_heading_threshold ────────────────────────────────────────────────

class TestComputeHeadingThreshold:
    def test_returns_median_times_multiplier(self):
        # Sizes: [11.0, 11.0, 16.0] → median = 11.0 → threshold = 11.0 * 1.2 = 13.2
        doc = _make_mock_doc([[("Heading", 16.0), ("Body", 11.0), ("Body", 11.0)]])
        threshold = _compute_heading_threshold(doc)
        assert pytest.approx(threshold, abs=0.1) == 13.2

    def test_empty_document_returns_fallback(self):
        doc = _make_mock_doc([[]])
        assert _compute_heading_threshold(doc) == 14.0

    def test_skips_image_blocks(self):
        mock_page = MagicMock()
        mock_page.get_text.return_value = {
            "blocks": [{"type": 1}]  # type=1 is an image block
        }
        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(side_effect=lambda: iter([mock_page]))
        assert _compute_heading_threshold(mock_doc) == 14.0

    def test_ignores_whitespace_only_spans(self):
        doc = _make_mock_doc([[("   ", 20.0), ("Real text", 10.0)]])
        # Only "Real text" at 10.0 is counted → median = 10.0 → 12.0
        threshold = _compute_heading_threshold(doc)
        assert pytest.approx(threshold, abs=0.1) == 12.0


# ── _classify_line_as_heading ─────────────────────────────────────────────────

class TestClassifyLineAsHeading:
    def test_large_font_qualifies_as_heading(self):
        span = {"size": 20.0, "flags": 0}
        assert _classify_line_as_heading("Introduction", span, 14.0) is True

    def test_bold_flag_qualifies_as_heading(self):
        # flags bit 4 (value 16) = bold
        span = {"size": 10.0, "flags": 16}
        assert _classify_line_as_heading("Summary", span, 14.0) is True

    def test_normal_text_not_a_heading(self):
        span = {"size": 10.0, "flags": 0}
        assert _classify_line_as_heading("Regular body text here.", span, 14.0) is False

    def test_single_character_drop_cap_rejected(self):
        """Fix: drop caps like 'T' from CBJ article must not become section headings."""
        span = {"size": 20.0, "flags": 0}
        assert _classify_line_as_heading("T", span, 14.0) is False

    def test_two_character_line_rejected(self):
        span = {"size": 20.0, "flags": 0}
        assert _classify_line_as_heading("To", span, 14.0) is False

    def test_exactly_three_characters_accepted(self):
        """Boundary test: is_meaningful threshold is len >= 3."""
        span = {"size": 20.0, "flags": 0}
        assert _classify_line_as_heading("FAQ", span, 14.0) is True

    def test_long_line_rejected_even_if_large_font(self):
        """Headings must be concise; body paragraphs rendered in large font are not headings."""
        import config
        span = {"size": 20.0, "flags": 0}
        long_text = "A" * (config.HEADING_MAX_CHARS + 1)
        assert _classify_line_as_heading(long_text, span, 14.0) is False

    def test_empty_stripped_line_rejected(self):
        span = {"size": 20.0, "flags": 0}
        assert _classify_line_as_heading("  ", span, 14.0) is False


# ── _create_child_chunks ──────────────────────────────────────────────────────

class TestCreateChildChunks:
    def test_children_inherit_parent_metadata(self, parent_doc_a):
        children = _create_child_chunks([parent_doc_a])
        assert all(c.metadata["source"] == "jd.pdf" for c in children)
        assert all(c.metadata["section"] == "Qualifications and Skills" for c in children)
        assert all(c.metadata["parent_id"] == parent_doc_a.metadata["parent_id"] for c in children)

    def test_children_have_chunk_type_child(self, parent_doc_a):
        children = _create_child_chunks([parent_doc_a])
        assert all(c.metadata["chunk_type"] == "child" for c in children)

    def test_children_have_sequential_child_index(self, parent_doc_a):
        children = _create_child_chunks([parent_doc_a])
        indices = [c.metadata["child_index"] for c in children]
        assert indices == list(range(len(children)))

    def test_child_text_within_chunk_size(self, parent_doc_a):
        import config
        children = _create_child_chunks([parent_doc_a])
        for child in children:
            assert len(child.page_content) <= config.CHILD_CHUNK_SIZE + config.CHILD_CHUNK_OVERLAP

    def test_empty_input_returns_empty_list(self):
        assert _create_child_chunks([]) == []

    def test_multiple_parents_produce_children_for_each(self, parent_doc_a, parent_doc_b):
        children = _create_child_chunks([parent_doc_a, parent_doc_b])
        parent_ids = {c.metadata["parent_id"] for c in children}
        assert parent_doc_a.metadata["parent_id"] in parent_ids
        assert parent_doc_b.metadata["parent_id"] in parent_ids


# ── extract_structured_sections ──────────────────────────────────────────────

class TestExtractStructuredSections:
    def test_returns_list_of_documents(self, sample_pdf):
        sections = extract_structured_sections(sample_pdf)
        assert isinstance(sections, list)
        assert all(isinstance(s, Document) for s in sections)

    def test_sections_have_required_metadata_keys(self, sample_pdf):
        sections = extract_structured_sections(sample_pdf)
        for section in sections:
            assert "source" in section.metadata
            assert "page" in section.metadata
            assert "section" in section.metadata
            assert "parent_id" in section.metadata
            assert section.metadata["chunk_type"] == "parent"

    def test_source_is_filename_not_full_path(self, sample_pdf):
        sections = extract_structured_sections(sample_pdf)
        for section in sections:
            assert "/" not in section.metadata["source"]
            assert "\\" not in section.metadata["source"]

    def test_detects_heading_as_section_boundary(self, sample_pdf):
        sections = extract_structured_sections(sample_pdf)
        section_names = [s.metadata["section"] for s in sections]
        assert "Test Section Heading" in section_names

    def test_each_section_has_non_empty_content(self, sample_pdf):
        sections = extract_structured_sections(sample_pdf)
        assert all(s.page_content.strip() for s in sections)

    def test_adjacent_headings_do_not_create_empty_sections(self, tmp_path):
        import fitz
        pdf_path = tmp_path / "adjacent_headings.pdf"
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 80),  "First Heading",  fontsize=24)
        page.insert_text((50, 120), "Second Heading", fontsize=24)
        page.insert_text((50, 160), "Some body content here.", fontsize=10)
        doc.save(str(pdf_path))
        doc.close()

        sections = extract_structured_sections(str(pdf_path))
        # "First Heading" section has no body → must be excluded
        contents = [s.page_content for s in sections]
        assert all(c.strip() for c in contents), "Empty sections must not be created"


# ── vectorstore_exists ────────────────────────────────────────────────────────

class TestVectorstoreExists:
    def test_returns_true_when_both_stores_present(self, tmp_path, monkeypatch):
        chroma_dir = tmp_path / "chroma"
        chroma_dir.mkdir()
        pkl_path = tmp_path / "parent_store.pkl"
        pkl_path.touch()
        monkeypatch.setattr("config.CHROMA_PERSIST_DIR", str(chroma_dir))
        monkeypatch.setattr("config.PARENT_STORE_PATH", str(pkl_path))
        assert vectorstore_exists() is True

    def test_returns_false_when_chroma_missing(self, tmp_path, monkeypatch):
        pkl_path = tmp_path / "parent_store.pkl"
        pkl_path.touch()
        monkeypatch.setattr("config.CHROMA_PERSIST_DIR", str(tmp_path / "nonexistent"))
        monkeypatch.setattr("config.PARENT_STORE_PATH", str(pkl_path))
        assert vectorstore_exists() is False

    def test_returns_false_when_parent_store_missing(self, tmp_path, monkeypatch):
        chroma_dir = tmp_path / "chroma"
        chroma_dir.mkdir()
        monkeypatch.setattr("config.CHROMA_PERSIST_DIR", str(chroma_dir))
        monkeypatch.setattr("config.PARENT_STORE_PATH", str(tmp_path / "nonexistent.pkl"))
        assert vectorstore_exists() is False

    def test_returns_false_when_neither_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.CHROMA_PERSIST_DIR", str(tmp_path / "no_chroma"))
        monkeypatch.setattr("config.PARENT_STORE_PATH", str(tmp_path / "no_pkl.pkl"))
        assert vectorstore_exists() is False


# ── clear_vectorstore ─────────────────────────────────────────────────────────

class TestClearVectorstore:
    def test_removes_chroma_directory(self, tmp_path, monkeypatch):
        chroma_dir = tmp_path / "chroma"
        chroma_dir.mkdir()
        pkl_path = tmp_path / "store.pkl"
        pkl_path.touch()
        monkeypatch.setattr("config.CHROMA_PERSIST_DIR", str(chroma_dir))
        monkeypatch.setattr("config.PARENT_STORE_PATH", str(pkl_path))
        clear_vectorstore()
        assert not chroma_dir.exists()

    def test_removes_parent_store_pkl(self, tmp_path, monkeypatch):
        chroma_dir = tmp_path / "chroma"
        chroma_dir.mkdir()
        pkl_path = tmp_path / "store.pkl"
        pkl_path.touch()
        monkeypatch.setattr("config.CHROMA_PERSIST_DIR", str(chroma_dir))
        monkeypatch.setattr("config.PARENT_STORE_PATH", str(pkl_path))
        clear_vectorstore()
        assert not pkl_path.exists()

    def test_does_not_raise_when_files_absent(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.CHROMA_PERSIST_DIR", str(tmp_path / "no_chroma"))
        monkeypatch.setattr("config.PARENT_STORE_PATH", str(tmp_path / "no.pkl"))
        clear_vectorstore()  # must not raise


# ── ingest_pdfs ───────────────────────────────────────────────────────────────

class TestIngestPdfs:
    def test_raises_value_error_on_no_extractable_text(self, tmp_path):
        """Empty/image-only PDFs must raise a descriptive ValueError."""
        import fitz
        empty_pdf = tmp_path / "empty.pdf"
        doc = fitz.open()
        doc.new_page()  # blank page — no text
        doc.save(str(empty_pdf))
        doc.close()
        with pytest.raises(ValueError, match="No extractable text"):
            ingest_pdfs([str(empty_pdf)])

    @patch("services.ingestion_service.Chroma")
    @patch("services.ingestion_service.get_embeddings")
    def test_creates_chroma_vectorstore(self, mock_emb, mock_chroma, sample_pdf, tmp_path, monkeypatch):
        monkeypatch.setattr("config.CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        monkeypatch.setattr("config.PARENT_STORE_PATH", str(tmp_path / "vs" / "store.pkl"))
        ingest_pdfs([sample_pdf])
        mock_chroma.from_documents.assert_called_once()

    @patch("services.ingestion_service.Chroma")
    @patch("services.ingestion_service.get_embeddings")
    def test_saves_pickle_in_new_format(self, mock_emb, mock_chroma, sample_pdf, tmp_path, monkeypatch):
        pkl_path = tmp_path / "vs" / "store.pkl"
        monkeypatch.setattr("config.CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        monkeypatch.setattr("config.PARENT_STORE_PATH", str(pkl_path))
        ingest_pdfs([sample_pdf])
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        assert "parent_map" in data
        assert "child_docs" in data

    @patch("services.ingestion_service.Chroma")
    @patch("services.ingestion_service.get_embeddings")
    def test_pickle_parent_map_keyed_by_parent_id(self, mock_emb, mock_chroma, sample_pdf, tmp_path, monkeypatch):
        pkl_path = tmp_path / "vs" / "store.pkl"
        monkeypatch.setattr("config.CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        monkeypatch.setattr("config.PARENT_STORE_PATH", str(pkl_path))
        ingest_pdfs([sample_pdf])
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        for parent_id, doc in data["parent_map"].items():
            assert doc.metadata["parent_id"] == parent_id

    @patch("services.ingestion_service.Chroma")
    @patch("services.ingestion_service.get_embeddings")
    def test_child_docs_are_child_chunk_type(self, mock_emb, mock_chroma, sample_pdf, tmp_path, monkeypatch):
        pkl_path = tmp_path / "vs" / "store.pkl"
        monkeypatch.setattr("config.CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        monkeypatch.setattr("config.PARENT_STORE_PATH", str(pkl_path))
        ingest_pdfs([sample_pdf])
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        assert all(d.metadata["chunk_type"] == "child" for d in data["child_docs"])
