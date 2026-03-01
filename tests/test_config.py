"""
Tests for config.py — verifies all constants are correctly typed, valued,
and that environment variable injection works as expected.
"""

import os
import importlib
import pytest


def test_groq_api_key_reads_from_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-xyz")
    import config
    importlib.reload(config)
    assert config.GROQ_API_KEY == "test-key-xyz"


def test_groq_api_key_defaults_to_empty_string(monkeypatch):
    from unittest.mock import patch
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with patch("dotenv.load_dotenv"):
        import config
        importlib.reload(config)
    assert config.GROQ_API_KEY == ""


def test_llm_model_is_llama():
    import config
    assert "llama" in config.LLM_MODEL.lower()


def test_llm_temperature_is_low_for_factual_responses():
    import config
    assert 0.0 <= config.LLM_TEMPERATURE <= 0.3


def test_llm_max_tokens_is_positive():
    import config
    assert config.LLM_MAX_TOKENS > 0


def test_embedding_model_is_minilm():
    import config
    assert config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"


def test_top_k_children_is_ten():
    """Validates the retrieval recall improvement fix — was 6, now 10."""
    import config
    assert config.TOP_K_CHILDREN == 10


def test_bm25_weight_is_between_zero_and_one():
    import config
    assert 0.0 < config.BM25_WEIGHT < 1.0


def test_chroma_and_parent_store_paths_configured():
    import config
    assert config.CHROMA_PERSIST_DIR != ""
    assert config.PARENT_STORE_PATH.endswith(".pkl")


def test_heading_detection_constants_sane():
    import config
    assert config.HEADING_SIZE_MULTIPLIER > 1.0
    assert config.HEADING_MAX_CHARS > 0
    assert config.CHILD_CHUNK_SIZE > config.CHILD_CHUNK_OVERLAP
