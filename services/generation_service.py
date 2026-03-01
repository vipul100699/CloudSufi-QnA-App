"""
Generation layer: prompt construction, LLM invocation, and citation formatting.

Design decisions:
  - System prompt enforces strict grounding: LLM must ONLY use provided context.
  - Context blocks are numbered [1], [2], etc. and each includes citation metadata.
  - LLM is instructed to cite inline using [N] notation after each claim.
  - A "Sources" section is appended to every answer for easy reference.
  - Temperature=0.1 ensures deterministic, factual responses (not creative generation).
"""

from typing import List, Dict

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

import config


# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a precise, grounded document assistant for CloudSufi.

Your task is to answer questions using ONLY the numbered document excerpts provided below.

Instructions:
1. Cite every factual claim inline using [N] immediately after the sentence, 
   where N is the number of the source excerpt.
2. If the answer draws from multiple sources, cite all relevant ones: e.g., [1][3].
3. If the provided excerpts do not contain sufficient information to answer the question,
   respond exactly with: "The provided documents do not contain enough information to answer this question."
4. Do NOT use any knowledge from outside the provided excerpts.
5. Structure your answer clearly. Use bullet points for lists or comparisons.
6. After your main answer, include a "**Sources Used:**" section listing each citation:
   [N] <filename> | Section: <section> | Page: <page>
"""


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
          answer      : LLM response string with inline citations.
          context_used: Formatted context block (shown as expandable in UI).
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

    # GROQ_API_KEY is auto-read from environment (loaded via dotenv in config.py)
    llm = ChatGroq(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS
    )

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_content)
    ])

    return {
        "answer": response.content,
        "context_used": context_block.strip()
    }
