"""
Tests for services/embeddings.py — verifies lazy import behaviour,
correct model configuration, and that the function is callable.

Note on @st.cache_resource:
  The root conftest.py replaces st.cache_resource with a passthrough decorator
  before this module is imported. get_embeddings() is therefore the plain
  original function, and HuggingFaceEmbeddings is mocked to avoid the ~80MB
  model download during test runs.
"""

from unittest.mock import patch, MagicMock
import config


def test_get_embeddings_returns_huggingface_instance():
    with patch("langchain_huggingface.HuggingFaceEmbeddings") as MockHF:
        MockHF.return_value = MagicMock()
        from services.embeddings import get_embeddings
        result = get_embeddings()
        assert result is MockHF.return_value


def test_get_embeddings_uses_configured_model_name():
    with patch("langchain_huggingface.HuggingFaceEmbeddings") as MockHF:
        MockHF.return_value = MagicMock()
        from services.embeddings import get_embeddings
        get_embeddings()
        call_kwargs = MockHF.call_args.kwargs
        assert call_kwargs["model_name"] == config.EMBEDDING_MODEL


def test_get_embeddings_uses_cpu_device():
    with patch("langchain_huggingface.HuggingFaceEmbeddings") as MockHF:
        MockHF.return_value = MagicMock()
        from services.embeddings import get_embeddings
        get_embeddings()
        assert MockHF.call_args.kwargs["model_kwargs"]["device"] == "cpu"


def test_get_embeddings_enables_normalization():
    with patch("langchain_huggingface.HuggingFaceEmbeddings") as MockHF:
        MockHF.return_value = MagicMock()
        from services.embeddings import get_embeddings
        get_embeddings()
        assert MockHF.call_args.kwargs["encode_kwargs"]["normalize_embeddings"] is True
