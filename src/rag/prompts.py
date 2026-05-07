"""RAG generation prompt templates.

Defines all prompt templates used across EXP_01 through EXP_09.
All experiments import from this single file.

TEMPLATES AVAILABLE
-------------------
RAG_PROMPT          — original baseline prompt (EXP_01, 02, 03 baseline runs)
GROUNDED_RAG_PROMPT — improved prompt with uncertainty signalling + focus
                      constraint (EXP_04 and all v2 improved runs)

The two prompts share the same system rules (1–7). GROUNDED_RAG_PROMPT
adds rule 8 (uncertainty signalling) and tightens the human message to
discourage broad energy reports in favour of focused answers.

WHAT CHANGED vs original prompts.py
-------------------------------------
1. Rule 8 added to GROUNDED_RAG_SYSTEM — model must say
   "The available summaries do not contain sufficient data..."
   when evidence is insufficient, rather than hallucinating a completion.
2. GROUNDED_RAG_HUMAN_MESSAGE tightens the closing instruction to
   explicitly anchor the answer to retrieved summaries only.
3. format_docs() now prefixes each document with a numbered [Summary N]
   label in addition to the row_id, improving RAGAS faithfulness
   traceability (the LLM can reference [Summary 1] in its answer).
4. RAG_PROMPT and its original format_docs() behaviour are fully
   preserved — no existing imports break.

USAGE
-----
    # Existing experiments (unchanged imports work as before)
    from src.rag.prompts import RAG_PROMPT, format_docs

    # Improved runs (EXP_04 onwards / v2 runs)
    from src.rag.prompts import GROUNDED_RAG_PROMPT, format_docs
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


# ─────────────────────────────────────────────────────────────────────────────
# Shared system rules (rules 1–7, identical in both prompts)
# ─────────────────────────────────────────────────────────────────────────────

_SHARED_RULES: str = (
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


# ─────────────────────────────────────────────────────────────────────────────
# RAG_PROMPT — original baseline (used by EXP_01/02/03 baseline runs)
# ─────────────────────────────────────────────────────────────────────────────

RAG_SYSTEM_MESSAGE: str = (
    "You are an expert energy systems analyst providing data-driven "
    "demand insights to utility managers and energy planners.\n\n"
    + _SHARED_RULES
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

RAG_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_MESSAGE),
    ("human",  RAG_HUMAN_MESSAGE),
])


# ─────────────────────────────────────────────────────────────────────────────
# GROUNDED_RAG_PROMPT — improved (EXP_04 onwards, all v2 runs)
#
# Adds rule 8: explicit uncertainty signalling when evidence is insufficient.
# Tightens the human message to discourage broad energy reports.
# ─────────────────────────────────────────────────────────────────────────────

GROUNDED_RAG_SYSTEM: str = (
    "You are an expert energy systems analyst providing data-driven "
    "demand insights to utility managers and energy planners.\n\n"
    + _SHARED_RULES + "\n"
    "8. If the retrieved summaries do not contain sufficient evidence to "
    "fully answer the question, explicitly state: \"The available summaries "
    "do not contain sufficient data to fully answer this question.\" "
    "Then answer only what the evidence supports. "
    "Do NOT use general knowledge about energy systems or GEFCom that is "
    "not present in the retrieved summaries."
)

GROUNDED_RAG_HUMAN: str = (
    "Retrieved summaries:\n"
    "{context}\n\n"
    "Question: {question}\n\n"
    "Answer the specific question asked, based only on the retrieved "
    "summaries above. Do not produce a broad energy report."
)

GROUNDED_RAG_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages([
    ("system", GROUNDED_RAG_SYSTEM),
    ("human",  GROUNDED_RAG_HUMAN),
])


# ─────────────────────────────────────────────────────────────────────────────
# format_docs — shared document formatter
#
# Used by all experiments. Adds [Summary N] index labels for RAGAS
# faithfulness traceability while preserving the original row_id labelling.
# ─────────────────────────────────────────────────────────────────────────────

def format_docs(docs: List[Document]) -> str:
    """Format a list of retrieved Documents into a single context string.

    Each document is labelled with a sequential [Summary N] index, its
    source row_id, and dataset/granularity metadata. The numeric index
    helps the LLM reference specific summaries (e.g. "According to
    Summary 2...") and improves RAGAS faithfulness tracing.

    Args:
        docs: List of Document objects from any retriever's retrieve() method.

    Returns:
        Formatted string with each document separated by blank lines.
        Returns "No summaries retrieved." if docs is empty.

    Example:
        >>> context = format_docs(retrieved_docs)
        >>> print(context[:120])
        [Summary 1] gefcom_daily_4_2004-01-01 (gefcom/daily):
        On January 1st 2004, Zone 4 saw an average hourly load of...
    """
    if not docs:
        return "No summaries retrieved."

    parts = []
    for i, doc in enumerate(docs, 1):
        source      = doc.metadata.get("source", doc.metadata.get("row_id", "unknown"))
        dataset     = doc.metadata.get("dataset", "")
        granularity = doc.metadata.get("granularity", "")
        label       = f"[Summary {i}] {source} ({dataset}/{granularity}):"
        parts.append(f"{label}\n{doc.page_content}")

    return "\n\n".join(parts)