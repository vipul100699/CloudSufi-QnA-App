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
# Open .env and set: GROQ_API_KEY=your_key_here

### 3. Run (single command — uv handles venv + dependency install automatically)
uv run streamlit run main.py

Then, Open http://localhost:8501 in your browser.

### Alternative:
pip install -r requirements.txt
streamlit run main.py


## Architecture

User Query
    │
    ▼
main.py  (Streamlit UI — fully decoupled from business logic)
    │
    ├─► services/ingestion_service.py
    │       PyMuPDF font metadata extraction
    │       ↓ Structure-Aware Section Detection  → Parent Chunks (1 per section)
    │       ↓ Drop-cap filter (is_meaningful ≥ 3 chars) prevents magazine PDF noise
    │       ↓ RecursiveCharacterTextSplitter     → Child Chunks (300 chars)
    │       ↓ HuggingFace Embeddings (all-MiniLM-L6-v2, local, lazy-loaded)
    │       ↓ ChromaDB (child vectors, persistent) + pickle (parent_map + child_docs)
    │
    ├─► services/retrieval_service.py
    │       Query → Hybrid BM25 (40%) + Dense Vector (60%) EnsembleRetriever
    │       → Top-K=10 child chunks via Reciprocal Rank Fusion
    │       → parent_id lookup → fetch full parent section (rich context)
    │       → deduplicate by parent_id → return contexts with citation metadata
    │
    └─► services/generation_service.py
            Prompt (services/prompts.py) — includes partial-answer awareness
            + Numbered context blocks → Groq LLM (llama-3.3-70b-versatile)
            → Grounded answer with inline [N] citations + Sources Used section

