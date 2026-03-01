"""
Tests for services/generation_service.py — covers context block formatting,
empty-context handling, LLM invocation parameters, and response structure.
"""

import pytest
from unittest.mock import patch, MagicMock

from services.generation_service import _format_context_blocks, generate_answer


# ── _format_context_blocks ────────────────────────────────────────────────────

class TestFormatContextBlocks:
    def test_first_block_labelled_one(self, sample_contexts):
        result = _format_context_blocks(sample_contexts)
        assert "[1]" in result

    def test_second_block_labelled_two(self, sample_contexts):
        result = _format_context_blocks(sample_contexts)
        assert "[2]" in result

    def test_includes_source_filename(self, sample_contexts):
        result = _format_context_blocks(sample_contexts)
        assert "jd.pdf" in result
        assert "about.pdf" in result

    def test_includes_section_name(self, sample_contexts):
        result = _format_context_blocks(sample_contexts)
        assert "Qualifications and Skills" in result

    def test_includes_page_number(self, sample_contexts):
        result = _format_context_blocks(sample_contexts)
        assert "Page: 1" in result

    def test_blocks_separated_by_divider(self, sample_contexts):
        result = _format_context_blocks(sample_contexts)
        assert "---" in result

    def test_empty_contexts_returns_minimal_string(self):
        result = _format_context_blocks([])
        assert isinstance(result, str)


# ── generate_answer ───────────────────────────────────────────────────────────

class TestGenerateAnswer:
    def test_no_contexts_returns_no_relevant_content_message(self):
        result = generate_answer("Any question?", [])
        assert "No relevant content" in result["answer"]
        assert result["context_used"] == ""

    def test_no_contexts_does_not_call_llm(self):
        with patch("services.generation_service.ChatGroq") as mock_llm:
            generate_answer("Any question?", [])
            mock_llm.assert_not_called()

    def test_returns_dict_with_answer_key(self, sample_contexts):
        mock_response = MagicMock()
        mock_response.content = "The answer is PyTorch [1]."
        with patch("services.generation_service.ChatGroq") as mock_llm:
            mock_llm.return_value.invoke.return_value = mock_response
            result = generate_answer("What libraries?", sample_contexts)
        assert "answer" in result

    def test_returns_dict_with_context_used_key(self, sample_contexts):
        mock_response = MagicMock()
        mock_response.content = "Answer."
        with patch("services.generation_service.ChatGroq") as mock_llm:
            mock_llm.return_value.invoke.return_value = mock_response
            result = generate_answer("Question?", sample_contexts)
        assert "context_used" in result
        assert result["context_used"] != ""

    def test_llm_invoked_with_system_and_human_messages(self, sample_contexts):
        from langchain_core.messages import SystemMessage, HumanMessage
        mock_response = MagicMock()
        mock_response.content = "Answer."
        with patch("services.generation_service.ChatGroq") as mock_llm:
            mock_llm.return_value.invoke.return_value = mock_response
            generate_answer("Question?", sample_contexts)
            invoke_args = mock_llm.return_value.invoke.call_args[0][0]
        assert any(isinstance(m, SystemMessage) for m in invoke_args)
        assert any(isinstance(m, HumanMessage) for m in invoke_args)

    def test_llm_instantiated_with_correct_model(self, sample_contexts):
        import config
        mock_response = MagicMock()
        mock_response.content = "Answer."
        with patch("services.generation_service.ChatGroq") as mock_llm:
            mock_llm.return_value.invoke.return_value = mock_response
            generate_answer("Question?", sample_contexts)
            assert mock_llm.call_args.kwargs["model"] == config.LLM_MODEL

    def test_llm_instantiated_with_correct_temperature(self, sample_contexts):
        import config
        mock_response = MagicMock()
        mock_response.content = "Answer."
        with patch("services.generation_service.ChatGroq") as mock_llm:
            mock_llm.return_value.invoke.return_value = mock_response
            generate_answer("Question?", sample_contexts)
            assert mock_llm.call_args.kwargs["temperature"] == config.LLM_TEMPERATURE

    def test_answer_content_equals_llm_response(self, sample_contexts):
        mock_response = MagicMock()
        mock_response.content = "PyTorch and Scikit-learn are required [1]."
        with patch("services.generation_service.ChatGroq") as mock_llm:
            mock_llm.return_value.invoke.return_value = mock_response
            result = generate_answer("Libraries?", sample_contexts)
        assert result["answer"] == "PyTorch and Scikit-learn are required [1]."
