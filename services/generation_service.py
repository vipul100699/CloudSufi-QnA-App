"""
Generation layer: prompt construction, LLM invocation, and citation formatting.

Design decisions:
  - System prompt is imported from services/prompts.py — single source of truth.
  - Context blocks are numbered [1], [2], etc., each labelled with citation metadata.
  - LLM is instructed to cite inline using [N] notation immediately after each claim.
  - GROQ_API_KEY is passed explicitly from config — no hidden os.environ reads.
  - Temperature=0.1 ensures deterministic, factual responses (not creative generation).
"""

from typing import List, Dict

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

import config
from services.prompts import DOCUMENT_QA_SYSTEM_PROMPT


def _format_context_blocks(contexts: List[Dict[str, str]]) -> str:
    """
    Formats retrieved contexts into numbered citation blocks for the prompt.
    Each block is clearly delimited and labelled for the LLM to reference.
    """
    blocks = []
    for i, ctx in enumerate(contexts, start=1):
        block = (
            f"[{i}] Source: {ctx['source']} | "
            f"Section: {ctx['section']} | "
            f"Page: {ctx['page']}\n"
            f"{ctx['content']}"
        )
        blocks.append(block)
    return "\n\n" + ("\n\n---\n\n".join(blocks)) + "\n"


def generate_answer(query: str, contexts: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Generates a grounded, cited answer for the user's query.

    Args:
        query    : The user's natural language question.
        contexts : List of retrieved context dicts from retrieval_service.

    Returns:
        Dict with:
          answer      : LLM response string with inline [N] citations.
          context_used: Formatted context block (rendered as expandable in UI).
    """
    if not contexts:
        return {
            "answer": (
                "No relevant content was found in the uploaded documents for your query. "
                "Try rephrasing your question or uploading additional documents."
            ),
            "context_used": ""
        }

    context_block = _format_context_blocks(contexts)
    human_content = (
        f"Document Excerpts:{context_block}\n"
        f"Question: {query}"
    )

    llm = ChatGroq(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
        api_key=config.GROQ_API_KEY  # Explicitly sourced from config, not os.environ
    )

    response = llm.invoke([
        SystemMessage(content=DOCUMENT_QA_SYSTEM_PROMPT),
        HumanMessage(content=human_content)
    ])

    return {
        "answer": response.content,
        "context_used": context_block.strip()
    }
