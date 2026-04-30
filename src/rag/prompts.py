"""RAG generation prompt templates.

Defines the prompt template used to instruct Llama 3.3 70B to generate
energy demand insights grounded in retrieved KB context. The prompt
enforces factual grounding — the model must base its answer strictly
on the provided context and must not hallucinate.

The template uses LangChain's ChatPromptTemplate with two variables:
    {context}  — formatted retrieved KB summaries
    {question} — the user's natural language query

The format_docs() utility converts a list of Document objects into
the formatted context string expected by the template.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# ─────────────────────────────────────────────────────────────────────────────
# System instructions for the RAG generation LLM
# ─────────────────────────────────────────────────────────────────────────────

RAG_SYSTEM_MESSAGE: str = (
    "You are an expert energy systems analyst providing data-driven "
    "demand insights to utility managers and energy planners.\n\n"
    "Rules:\n"
    "1. Base your answer STRICTLY on the provided context summaries.\n"
    "2. Use specific numbers, dates, and zone identifiers from the context.\n"
    "3. If the context does not contain enough information to fully answer "
    "the question, explicitly state what information is missing.\n"
    "4. Do NOT hallucinate or introduce facts not present in the context.\n"
    "5. Do NOT speculate about causes unless the context supports it.\n"
    "6. Write 3-5 sentences in clear, stakeholder-friendly language.\n"
    "7. When comparing values, state both numbers and the direction of "
    "the difference (higher/lower, increase/decrease)."
)

RAG_HUMAN_MESSAGE: str = (
    "Context (retrieved from the Energy Knowledge Base):\n"
    "---\n"
    "{context}\n"
    "---\n\n"
    "Question: {question}\n\n"
    "Provide a factual, evidence-grounded answer based strictly on the "
    "context above."
)

# ─────────────────────────────────────────────────────────────────────────────
# LangChain ChatPromptTemplate
# ─────────────────────────────────────────────────────────────────────────────

RAG_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_MESSAGE),
    ("human", RAG_HUMAN_MESSAGE),
])


# ─────────────────────────────────────────────────────────────────────────────
# Document formatter
# ─────────────────────────────────────────────────────────────────────────────

def format_docs(docs: List[Document]) -> str:
    """Format a list of retrieved Documents into a single context string.

    Each document is labelled with its source row_id and dataset/granularity
    metadata so the LLM can reference specific summaries in its answer.
    This labelling also helps RAGAS faithfulness evaluation trace which
    context chunks were used.

    Args:
        docs: List of Document objects from any retriever's retrieve() method.

    Returns:
        Formatted string with each document separated by blank lines.

    Example:
        >>> context = format_docs(retrieved_docs)
        >>> print(context[:100])
        [gefcom_daily_4_2004-01-01] (gefcom/daily):
        On January 1st 2004, Zone 4 saw an average...
    """
    parts = []
    for doc in docs:
        source      = doc.metadata.get("source", "unknown")
        dataset     = doc.metadata.get("dataset", "")
        granularity = doc.metadata.get("granularity", "")
        label       = f"[{source}] ({dataset}/{granularity}):"
        parts.append(f"{label}\n{doc.page_content}")

    return "\n\n".join(parts)
