# CloudSufi Document Q&A

A RAG-based document question-answering system built as a take-home assignment
for the Senior AI/ML Engineer role at CloudSufi.

Upload 1–3 PDF documents and ask natural language questions. Every answer includes
**inline citations** with the exact document name, section heading, and page number.

---

## Quickstart

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — fast Python
  package manager (replaces pip + venv):
  ```bash
  # macOS / Linux
  curl -LsSf https://astral.sh/uv/install.sh | sh

  # Windows (PowerShell)
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"


## Setup and Run (Single Command)

### 1. Clone the repository
git clone https://github.com/vipul100699/CloudSufi-QnA-App.git
cd CloudSufi-QnA-App

### 2. Configure your API key
cp .env.example .env
Open .env and set: GROQ_API_KEY=your_key_here

### 3. Run (single command — uv handles venv + dependency install automatically)
uv run streamlit run main.py

Then, Open http://localhost:8501 in your browser.

### Alternative:
pip install -r requirements.txt
streamlit run main.py

## Run Tests

uv run pytest          # full suite with coverage report
uv run pytest -v       # verbose output  (I got a test coverage of 98.84%)



## Approach

### The Core Problem with Naive Chunking
Fixed-size overlap chunking (the default RAG approach) has two failure modes
that directly affect this use case:

Structure blindness — splits mid-sentence across section boundaries,
producing incoherent embeddings that retrieve the wrong context.

Keyword miss — pure vector search fails on keyword-precise queries
(e.g., "PyTorch, Scikit-learn") because specific proper nouns have low
weight in semantic embedding space, while semantically similar but lexically
irrelevant chunks (e.g., "MLOps experience") rank higher.

Both are solved by the two-layer strategy below.


### Layer 1 — Structure-Aware Hierarchical Chunking
Problem: How do you know where a "section" starts and ends in a PDF?

Solution: PyMuPDF exposes per-span font metadata (size, bold flag). The
ingestion pipeline computes the median body font size across all text spans,
then classifies any line above median × 1.2 as a section heading. This
multiplier-based threshold generalises across PDFs with different base font
sizes rather than using a hardcoded value.

Each detected heading starts a new Parent Chunk — one Document object
per section. Parent chunks are then split into smaller Child Chunks
(300 chars, 40 char overlap) for embedding. Every child stores its parent's
UUID in metadata, enabling the retrieval step to "look up" the full section.

### Layer 2 — Hybrid BM25 + Dense Vector Retrieval (Small-to-Big)
At query time, a BM25Retriever (exact keyword match) and a ChromaDB dense
retriever (semantic similarity) are combined into an EnsembleRetriever via
Reciprocal Rank Fusion (RRF) with weights 40% BM25 / 60% Dense.

BM25 catches keyword-precise queries: "PyTorch", "LangGraph"

Dense catches semantic queries: "what does making data dance mean?"

RRF merges both ranked lists into a single ranked result

The ensemble retrieves child chunks (high precision), then each child's
parent_id is resolved to its full parent section (rich context). Results
are deduplicated by parent_id so the same section never appears twice.

## Architecture

┌─────────────────────────────────────────────────────────────────┐
│                        USER (Browser)                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │  HTTP
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   main.py  (Streamlit UI)                       │
│   File upload ──► Process  │  Chat input ──► Answer display     │
└──────────┬─────────────────┼────────────────────────────────────┘
           │                 │
    INGEST PATH         QUERY PATH
           │                 │
           ▼                 ▼
┌──────────────────┐  ┌──────────────────────────────────────────┐
│ingestion_service │  │          retrieval_service               │
│                  │  │                                          │
│ PyMuPDF          │  │  BM25Retriever  +  ChromaDB dense        │
│  font metadata   │  │       40%       +       60%              │
│       ↓          │  │           EnsembleRetriever              │
│ Parent Chunks    │  │            (RRF fusion)                  │
│  (1/section)     │  │                 ↓                        │
│       ↓          │  │     Top-K=10 child chunks                │
│ Child Chunks     │  │                 ↓                        │
│  (300 chars)     │  │  parent_id → full parent section         │
│       ↓          │  │                 ↓                        │
│ Embeddings ──────┼──┤  Deduplicate → context list             │
│ (MiniLM, local)  │  └────────────────┬─────────────────────────┘
│       ↓          │                   │
│  ChromaDB ───────┘                   ▼
│  (child vecs)    │  ┌──────────────────────────────────────────┐
│  pickle ─────────┘  │         generation_service               │
│  (parent_map        │                                          │
│   + child_docs)     │  System prompt (prompts.py)              │
└──────────────────┘  │  + Numbered context blocks..     │[1][2]
                       │  + User query                            │
                       │          ↓                               │
                       │  Groq LLaMA 3.3 70B                      │
                       │          ↓                               │
                       │  Answer + inline [N] citations           │
                       │  + Sources Used footer                   │
                       └──────────────────────────────────────────┘



## Known Limitations

1. Scanned / image-only PDFs are not supported. The heading detection
pipeline relies on PyMuPDF font metadata, which only exists in
text-layer PDFs. Scanned documents would require OCR (e.g., pytesseract)
as a pre-processing step.

2. Re-uploading clears the previous index. The vectorstore is a single
shared directory. Uploading new documents wipes the existing index.
There is no per-session or per-user isolation.

3. BM25 index is rebuilt on every query. BM25 is a stateless index that
cannot be persisted to disk like ChromaDB. It is reconstructed from the
pickle store at each retrieval call. At 1–3 PDF scale (~50 child chunks)
this takes <10ms. At 100+ document scale this becomes a bottleneck.

4. No streaming responses. The full LLM response is generated before
being displayed. Long answers have a visible delay (~2–4s on Groq free tier).

5. Chat history is session-scoped. Page refresh loses the conversation.
There is no persistent conversation storage.

6. Max 3 PDFs per session is enforced in the UI but not in the ingestion
service directly. The service itself has no hard cap — the UI limit is a
guardrail for this assignment scope.

7. General-purpose embeddings. all-MiniLM-L6-v2 is not fine-tuned for
domain-specific vocabulary (e.g., medical, legal, financial). Retrieval
quality may degrade for highly specialised documents.


## Improvements with More Time

In order of impact:

1. Cross-encoder reranker — Add a cross-encoder/ms-marco-MiniLM-L-6-v2
reranker after the ensemble retriever to re-score the top-K results with
full query-passage attention. Highest single improvement for retrieval
precision at 50+ document scale.

2. Streaming LLM responses — Use llm.stream() with st.write_stream()
for token-by-token display. Eliminates the perceived latency gap.

3. OCR support for scanned PDFs — Integrate pytesseract or pdfplumber
as a fallback when PyMuPDF detects no text layer. Expands supported
document types significantly.

4. Replace pickle with SQLite — pickle is not safe for concurrent
access. Replacing it with SQLite (via SQLAlchemy) would support multi-user
deployments and eliminate the single-file bottleneck.

5. HyDE (Hypothetical Document Embeddings) — Generate a hypothetical
answer to the query, embed it, and use that embedding for retrieval instead
of the raw query. Improves recall for questions phrased differently from
how the answer is written in the document.

6. Persistent BM25 index — Serialise the BM25 index to disk at ingestion
time and load it at retrieval time, removing the rebuild overhead at scale.

7. Per-user session isolation — Use Streamlit's session state with a
per-session vectorstore path, enabling multiple concurrent users without
index collision.

8. FastAPI backend — The services layer is already fully decoupled from
Streamlit. Replacing main.py with a FastAPI app would make the system
deployable as a REST API with zero changes to any service file.




