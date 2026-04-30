"""Groq LLM wrapper for RAG generation.

Wraps Llama 3.3 70B via Groq's API using LangChain's ChatGroq class.
This is the only place in the codebase where Groq is used — Phases 1-4
use Gemini (Google AI) and local sentence-transformers.

Model independence:
    Llama 3.3 70B (Meta/Groq) is a completely different model family
    from Gemini (Google AI). This ensures RAGAS faithfulness scores
    compare Llama-generated answers against Gemini-generated reference
    answers — a genuine cross-model evaluation.

The same LLM instance is also used as the RAGAS judge in Phase 6,
which is independent of both the KB model (Gemini 3 Flash) and the
golden dataset model (Gemini 2.5 Flash).
"""

from __future__ import annotations

import getpass
import logging
import os
from typing import Optional

from langchain_groq import ChatGroq

from config import RAG_MAX_TOKENS, RAG_MODEL_NAME, RAG_TEMPERATURE

logger = logging.getLogger(__name__)


def get_rag_llm(api_key: Optional[str] = None) -> ChatGroq:
    """Return a configured ChatGroq instance for RAG generation.

    Resolves the Groq API key in this priority order:
        1. Explicit ``api_key`` argument (for testing)
        2. ``GROQ_API_KEY`` environment variable (.env)
        3. Interactive getpass prompt as final fallback

    Args:
        api_key: Optional explicit API key override.

    Returns:
        Configured ChatGroq instance wrapping Llama 3.3 70B,
        ready for use in LCEL chains or direct .invoke() calls.

    Raises:
        ValueError: If no API key can be resolved from any source.

    Example:
        >>> llm = get_rag_llm()
        >>> response = llm.invoke("Explain peak demand patterns.")
        >>> print(response.content)
    """
    if api_key is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if api_key:
            logger.info("Groq API key loaded from environment variable.")

    if not api_key:
        logger.info(
            "Groq API key not found in environment. "
            "Prompting for manual entry."
        )
        api_key = getpass.getpass("Enter your Groq API key: ")

    if not api_key:
        raise ValueError(
            "No Groq API key found. "
            "Add GROQ_API_KEY to your .env file."
        )

    llm = ChatGroq(
        model=RAG_MODEL_NAME,
        temperature=RAG_TEMPERATURE,
        max_tokens=RAG_MAX_TOKENS,
        api_key=api_key,
    )

    logger.info(
        "ChatGroq configured: model='%s', temperature=%.1f, "
        "max_tokens=%d.",
        RAG_MODEL_NAME, RAG_TEMPERATURE, RAG_MAX_TOKENS,
    )
    return llm
