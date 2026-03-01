# CloudSufi Document Q&A

A RAG-based document question-answering system built as a take-home assignment
for the Senior AI/ML Engineer role at CloudSufi.

Ask natural language questions across 1–3 PDF documents and receive grounded answers
with precise inline citations (document, section, and page number).

---

## Architecture

User Query
    │
    ▼
Streamlit UI (app.py)
    │
    ├─► ingestion_service.py
    │       PyMuPDF font metadata extraction
    │       ↓ Structure-Aware Section Detection (Parent Chunks)
    │       ↓ RecursiveCharacterTextSplitter (Child Chunks)
    │       ↓ HuggingFace Embeddings (all-MiniLM-L6-v2)
    │       ↓ ChromaDB (child vectors) + Pickle Store (parent docs)
    │
    ├─► retrieval_service.py
    │       Query → Embed → ChromaDB similarity search (child chunks)
    │       → parent_id lookup → fetch parent section (rich context)
    │       → deduplicate → return context list with citation metadata
    │
    └─► generation_service.py
            Numbered context blocks + citation instructions → Groq LLM
            (llama-3.3-70b-versatile) → Answer with inline [N] citations


## Quickstart

### 1. Clone and set up environment

```bash
git clone https://github.com/<your-username>/cloudsufi-doc-qa.git
cd cloudsufi-doc-qa
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

