"""
Tests for services/prompts.py — validates the system prompt content, structure,
and the presence of all 7 required instruction clauses.
"""

from services.prompts import DOCUMENT_QA_SYSTEM_PROMPT


def test_prompt_is_non_empty_string():
    assert isinstance(DOCUMENT_QA_SYSTEM_PROMPT, str)
    assert len(DOCUMENT_QA_SYSTEM_PROMPT.strip()) > 100


def test_prompt_contains_grounding_instruction():
    """LLM must be told to use ONLY retrieved excerpts — no outside knowledge."""
    assert "ONLY" in DOCUMENT_QA_SYSTEM_PROMPT


def test_prompt_contains_inline_citation_instruction():
    """Every factual claim must be followed by an inline [N] citation."""
    assert "[N]" in DOCUMENT_QA_SYSTEM_PROMPT or "cite" in DOCUMENT_QA_SYSTEM_PROMPT.lower()


def test_prompt_contains_no_outside_knowledge_instruction():
    assert "outside" in DOCUMENT_QA_SYSTEM_PROMPT.lower()


def test_prompt_contains_sources_used_section_instruction():
    """Requires a structured 'Sources Used' footer in every answer."""
    assert "Sources Used" in DOCUMENT_QA_SYSTEM_PROMPT


def test_prompt_contains_insufficient_info_response():
    """Defines the exact fallback string when context is inadequate."""
    assert "do not contain" in DOCUMENT_QA_SYSTEM_PROMPT.lower()


def test_prompt_contains_partial_answer_awareness():
    """
    Instruction 7: LLM must flag when retrieved sections may be incomplete.
    This is the fix applied after the ML libraries retrieval failure diagnosis.
    """
    assert "partially" in DOCUMENT_QA_SYSTEM_PROMPT.lower() or \
           "may not include" in DOCUMENT_QA_SYSTEM_PROMPT.lower()
