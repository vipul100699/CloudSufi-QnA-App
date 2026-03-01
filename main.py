"""
CloudSufi Document Q&A — Streamlit Application

Entry point for the RAG-based document question-answering system.
Business logic is fully decoupled into the services/ layer — this file
can be replaced with a FastAPI backend with zero changes to ingestion,
retrieval, or generation logic.

Run:
    uv run streamlit run main.py     # recommended (uv manages the venv)
    streamlit run main.py            # alternative (manual venv activation)
"""

import os
import tempfile

import streamlit as st
import config


# ── Page Config ───────────────────────────────────────────────────────────────
# MUST be the first Streamlit call in the script.
# Streamlit raises StreamlitAPIException if any other st.* call precedes it.
# NOTE: Service imports (ingestion, retrieval, generation) are intentionally
# deferred to inside functions below. Top-level service imports trigger
# sentence-transformers + PyTorch loading BEFORE this line executes,
# causing a blank white screen for 5-15s on startup.
st.set_page_config(
    page_title="CloudSufi | Document Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ── Startup Validation ────────────────────────────────────────────────────────
if not config.GROQ_API_KEY:
    st.error(
        "⚠️ **GROQ_API_KEY is not set.** "
        "Copy `.env.example` → `.env`, add your key, and restart the app."
    )
    st.stop()


# ── Session State Initialization ──────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "docs_processed" not in st.session_state:
    # Deferred import: vectorstore_exists() only reads the filesystem — it does
    # not trigger model loading. Safe to call early.
    from services.ingestion_service import vectorstore_exists
    st.session_state.docs_processed = vectorstore_exists()

if "processed_filenames" not in st.session_state:
    st.session_state.processed_filenames = []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 CloudSufi Document Q&A")
    st.caption("Powered by LLaMA 3.3 70B · ChromaDB · LangChain")
    st.divider()

    st.markdown("### 📂 Upload Documents")
    st.caption("Upload 1–3 PDF documents. Ask anything about their content.")

    uploaded_files = st.file_uploader(
        label="Select PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload between 1 and 3 PDF documents."
    )

    if uploaded_files and len(uploaded_files) > 3:
        st.warning("⚠️ Maximum 3 documents supported. Only the first 3 will be used.")
        uploaded_files = uploaded_files[:3]

    process_clicked = st.button(
        label="⚙️ Process Documents",
        use_container_width=True,
        disabled=not uploaded_files,
        type="primary"
    )

    if process_clicked and uploaded_files:
        # Deferred imports: these trigger sentence-transformers + ChromaDB loading.
        # Importing here (inside the button handler) means they only load when the
        # user explicitly clicks "Process Documents" — not on every page render.
        from services.ingestion_service import ingest_pdfs, clear_vectorstore

        with st.spinner("🔍 Parsing structure, chunking, and indexing..."):
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_paths = []
                    for f in uploaded_files:
                        tmp_path = os.path.join(tmpdir, f.name)
                        with open(tmp_path, "wb") as out:
                            out.write(f.read())
                        tmp_paths.append(tmp_path)

                    clear_vectorstore()
                    ingest_pdfs(tmp_paths)

                st.session_state.docs_processed = True
                st.session_state.processed_filenames = [f.name for f in uploaded_files]
                st.session_state.chat_history = []
                st.success("✅ Documents indexed successfully!")

            except ValueError as ve:
                st.error(f"❌ {str(ve)}")
                st.session_state.docs_processed = False
            except Exception as e:
                st.error(f"❌ Unexpected error during processing: {str(e)}")
                st.session_state.docs_processed = False

    if st.session_state.docs_processed and st.session_state.processed_filenames:
        st.divider()
        st.markdown("**📑 Indexed Documents**")
        for fname in st.session_state.processed_filenames:
            st.markdown(f"- `{fname}`")

    st.divider()
    st.markdown("**⚙️ Stack**")
    st.caption(f"🤖 LLM: `{config.LLM_MODEL}`")
    st.caption(f"🔢 Embeddings: `{config.EMBEDDING_MODEL}`")
    st.caption(f"🗄️ Vector Store: `ChromaDB (persistent)`")
    st.caption(f"✂️ Chunking: `Structure-Aware + Hierarchical`")
    st.caption(f"🔍 Top-K Children: `{config.TOP_K_CHILDREN}`")

    if st.session_state.chat_history:
        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ── Main Panel ────────────────────────────────────────────────────────────────
st.title("📄 CloudSufi Document Q&A")
st.markdown(
    "Ask natural language questions about your uploaded documents. "
    "Every answer includes **inline citations** with document name, section, and page number."
)
st.divider()

if not st.session_state.docs_processed:
    st.info(
        "👈 **Get started:** Upload 1–3 PDF documents in the sidebar "
        "and click **Process Documents**."
    )
    st.markdown(
        "> 💡 **Suggested documents:** The CloudSufi Case Study, the JD PDF, "
        "and the [CloudSufi CBJ article](https://calbizjournal.com/wp-content/"
        "uploads/2021/10/cbj-cloudsufi-article.pdf) make an excellent demo set."
    )
    st.stop()


# ── Chat Interface ────────────────────────────────────────────────────────────
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("context_used"):
            with st.expander("📚 View Retrieved Context", expanded=False):
                st.code(message["context_used"], language="markdown")

user_query = st.chat_input(
    placeholder="e.g. What are the key responsibilities of the AI/ML Engineer role?",
    disabled=not st.session_state.docs_processed
)

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_query
    })

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                # Deferred imports: retrieval and generation services are only
                # imported when the user submits a query. By this point,
                # get_embeddings() has already been warm-started by the
                # ingestion step, so there is no additional loading delay.
                from services.retrieval_service import retrieve_context
                from services.generation_service import generate_answer

                contexts = retrieve_context(user_query)
                result = generate_answer(user_query, contexts)

                st.markdown(result["answer"])

                if result["context_used"]:
                    with st.expander("📚 View Retrieved Context", expanded=False):
                        st.code(result["context_used"], language="markdown")

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "context_used": result["context_used"]
                })

            except Exception as e:
                error_msg = f"❌ An error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg,
                    "context_used": ""
                })
