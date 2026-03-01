"""
PDF ingestion using a Structure-Aware Hierarchical Chunking strategy.

Strategy Overview:
  Parent Chunks  → Detected via PyMuPDF font metadata (headings define section boundaries).
                   One Document per section. Rich context for LLM generation.
  Child Chunks   → Parent sections split by RecursiveCharacterTextSplitter.
                   Small, precise units for embedding and vector search.
  Link           → Each child chunk stores its parent's UUID in metadata.

At retrieval time:
  1. Search child chunks for query precision.
  2. Fetch parent chunks via parent_id for rich LLM context.

This avoids the main failure of fixed-size overlap chunking: structure-blind splits
that break sentences mid-thought and produce incoherent embeddings.
"""

import os
import uuid
import pickle
import shutil
from pathlib import Path
from statistics import median
from typing import List, Dict

import fitz  # PyMuPDF
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

import config
from services.embeddings import get_embeddings


# ── Heading Detection ─────────────────────────────────────────────────────────

def _compute_heading_threshold(doc: fitz.Document) -> float:
    """
    Scans all text spans in the document to compute the median body font size.
    Returns the font size threshold above which text is classified as a heading.
    Uses a multiplier instead of a hardcoded value to generalize across PDFs.
    """
    sizes = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:  # skip image blocks
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    if span["text"].strip():
                        sizes.append(round(span["size"], 1))

    if not sizes:
        return 14.0  # Safe fallback

    return median(sizes) * config.HEADING_SIZE_MULTIPLIER


def _classify_line_as_heading(line_text: str, dominant_span: dict, threshold: float) -> bool:
    """
    Returns True if a line qualifies as a section heading.

    Criteria:
      - (Font size >= threshold) OR (Bold flag set in PyMuPDF bitmask)
      - AND line is short (headings are concise; long lines are body text)

    PyMuPDF span["flags"] bitmask:
      bit 0 (1)  = superscript
      bit 1 (2)  = italic
      bit 2 (4)  = serifed
      bit 3 (8)  = monospaced
      bit 4 (16) = bold  ← we check this
    """
    is_large_font = dominant_span["size"] >= threshold
    is_bold = bool(dominant_span["flags"] & 16)
    is_short = len(line_text) <= config.HEADING_MAX_CHARS

    # NEW: Reject single-character lines — these are decorative drop caps,
    # not section headings. A heading must be at least 3 characters.
    is_meaningful = len(line_text.strip()) >= 3

    return (is_large_font or is_bold) and is_short and is_meaningful


# ── Section Extraction ────────────────────────────────────────────────────────

def extract_structured_sections(pdf_path: str) -> List[Document]:
    """
    Extracts sections from a PDF using structural cues (font size, bold flags).
    Returns a list of LangChain Documents — one per detected section (parent chunks).

    Each Document metadata contains:
      source    : PDF filename (for citations)
      page      : Page number where the section begins
      section   : Detected section heading title
      parent_id : UUID linking this parent to its child chunks
      chunk_type: "parent"
    """
    filename = Path(pdf_path).name
    doc = fitz.open(pdf_path)
    threshold = _compute_heading_threshold(doc)

    sections: List[Document] = []
    current_heading: str = "Document Overview"
    current_page: int = 1
    current_lines: List[str] = []

    def _flush_section():
        """Commits accumulated text as a parent Document."""
        text = "\n".join(current_lines).strip()
        if text:
            sections.append(Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "page": current_page,
                    "section": current_heading,
                    "parent_id": str(uuid.uuid4()),
                    "chunk_type": "parent"
                }
            ))

    for page_num, page in enumerate(doc, start=1):
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue

            for line in block["lines"]:
                line_text = " ".join(
                    span["text"] for span in line["spans"]
                ).strip()

                if not line_text:
                    continue

                # Use the largest span's properties to classify the entire line
                dominant_span = max(line["spans"], key=lambda s: s["size"])

                if _classify_line_as_heading(line_text, dominant_span, threshold):
                    _flush_section()
                    current_heading = line_text
                    current_page = page_num
                    current_lines = []
                else:
                    current_lines.append(line_text)

    _flush_section()  # Persist the final section
    doc.close()
    return sections


# ── Child Chunk Creation ──────────────────────────────────────────────────────

def _create_child_chunks(parent_docs: List[Document]) -> List[Document]:
    """
    Splits each parent section Document into smaller child chunks for embedding.

    Each child Document:
      - Contains a small slice of the parent's text (high retrieval precision)
      - Inherits all parent metadata (source, page, section, parent_id)
      - Adds chunk_type="child" and child_index for traceability
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHILD_CHUNK_SIZE,
        chunk_overlap=config.CHILD_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " "]
    )

    children: List[Document] = []
    for parent in parent_docs:
        chunks = splitter.split_text(parent.page_content)
        for idx, chunk_text in enumerate(chunks):
            children.append(Document(
                page_content=chunk_text,
                metadata={
                    **parent.metadata,   # Inherit all parent metadata for citations
                    "chunk_type": "child",
                    "child_index": idx
                }
            ))
    return children


# ── Ingestion Pipeline ────────────────────────────────────────────────────────

def ingest_pdfs(pdf_paths: List[str]) -> None:
    """
    Full ingestion pipeline for a list of PDF file paths.

    Steps:
      1. Extract structured sections (parents) from each PDF via PyMuPDF.
      2. Split parents into child chunks via RecursiveCharacterTextSplitter.
      3. Embed child chunks and store in ChromaDB (persistent vector store).
      4. Serialize parent Documents to a pickle store (indexed by parent_id).

    Args:
        pdf_paths: List of file system paths to PDF documents.

    Raises:
        ValueError: If no text could be extracted from the provided PDFs.
    """
    all_parents: List[Document] = []
    for path in pdf_paths:
        sections = extract_structured_sections(path)
        all_parents.extend(sections)

    if not all_parents:
        raise ValueError(
            "No extractable text found in the provided PDFs. "
            "Ensure files are not scanned images or password-protected."
        )

    child_docs = _create_child_chunks(all_parents)

    # Store child chunk embeddings in ChromaDB
    os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)
    Chroma.from_documents(
        documents=child_docs,
        embedding=get_embeddings(),
        persist_directory=config.CHROMA_PERSIST_DIR,
        collection_name=config.CHROMA_COLLECTION_NAME
    )

    # Persist parent docs keyed by parent_id for retrieval-time context fetch
    os.makedirs(os.path.dirname(config.PARENT_STORE_PATH), exist_ok=True)
    parent_map: Dict[str, Document] = {
        doc.metadata["parent_id"]: doc for doc in all_parents
    }
    # child_docs are persisted alongside parent_map so the BM25 index can be
    # reconstructed at retrieval time without re-parsing the PDFs.
    with open(config.PARENT_STORE_PATH, "wb") as f:
        pickle.dump({"parent_map": parent_map, "child_docs": child_docs}, f)


# ── Utility Functions ─────────────────────────────────────────────────────────

def vectorstore_exists() -> bool:
    """Returns True if a persisted vectorstore and parent store are both present."""
    return (
        os.path.exists(config.CHROMA_PERSIST_DIR)
        and os.path.exists(config.PARENT_STORE_PATH)
    )


def clear_vectorstore() -> None:
    """
    Deletes existing ChromaDB and parent store to allow fresh ingestion.
    Called before re-ingestion when the user uploads new documents.
    """
    if os.path.exists(config.CHROMA_PERSIST_DIR):
        shutil.rmtree(config.CHROMA_PERSIST_DIR)
    if os.path.exists(config.PARENT_STORE_PATH):
        os.remove(config.PARENT_STORE_PATH)
