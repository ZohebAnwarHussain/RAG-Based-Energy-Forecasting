"""LCEL chain construction and batch RAG answer generation.

Two main functions:

    build_rag_chain()       Constructs a LangChain Expression Language (LCEL)
                            chain that retrieves context and generates answers.

    generate_rag_answers()  Runs all golden dataset queries through a specified
                            retriever, generates RAG answers via the chain,
                            and saves results to CSV for evaluation.

LCEL chain pattern:
    {"context": retriever | format_docs, "question": passthrough}
        → prompt
        → llm
        → StrOutputParser

This is the standard LangChain RAG chain — recognisable to any reviewer.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from tqdm.auto import tqdm

from config import REQUEST_DELAY_SECONDS
from src.rag.prompts import RAG_PROMPT, format_docs
from src.utils.timestamps import get_timestamp

logger = logging.getLogger(__name__)


def build_rag_chain(
    retriever: Any,
    llm: ChatGroq,
) -> Any:
    """Build a standard LCEL RAG chain.

    The chain pattern:
        1. Retrieve relevant documents using the provided retriever
        2. Format documents into a context string
        3. Inject context + question into the prompt template
        4. Send to Llama 3.3 70B via Groq
        5. Parse the output string

    Args:
        retriever: A LangChain-compatible retriever (from any pipeline's
            as_langchain_retriever() method).
        llm: Configured ChatGroq instance from get_rag_llm().

    Returns:
        LCEL chain that accepts a question string and returns an
        answer string.

    Example:
        >>> chain = build_rag_chain(dense_retriever.as_langchain_retriever(), llm)
        >>> answer = chain.invoke("What was peak winter demand in Zone 4?")
        >>> print(answer)
    """
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    logger.info("LCEL RAG chain built successfully.")
    return chain


def generate_rag_answers(
    golden_df: pd.DataFrame,
    retrieval_results_df: pd.DataFrame,
    retrievers: Dict[str, Any],
    llm: ChatGroq,
    output_path: Path,
    request_delay: float = REQUEST_DELAY_SECONDS,
) -> pd.DataFrame:
    """Generate RAG answers for all golden queries using retrieved context.

    For each row in retrieval_results_df, retrieves documents using the
    specified pipeline, formats them as context, and generates an answer
    via Llama 3.3 70B. Results are appended incrementally to output_path
    for crash recovery.

    Args:
        golden_df: Combined golden dataset DataFrame containing user_query,
            reference_answer, and evaluation fields.
        retrieval_results_df: Retrieval results from Phase 4 containing
            golden_id, pipeline, and retrieved_ids per query.
        retrievers: Dict mapping pipeline name to retriever instance.
            Example: {"dense": DenseRetriever, "hybrid": HybridRetriever, ...}
        llm: Configured ChatGroq instance from get_rag_llm().
        output_path: CSV path for incremental output. Created with headers
            if it does not exist.
        request_delay: Seconds to sleep between Groq API calls.
            Default REQUEST_DELAY_SECONDS.

    Returns:
        Complete results DataFrame loaded from output_path.
    """
    RAG_COLUMNS = [
        "golden_id",
        "pipeline",
        "user_query",
        "retrieved_context",
        "rag_answer",
        "reference_answer",
        "generated_at",
    ]

    # Resume support
    if output_path.exists():
        existing      = pd.read_csv(output_path)
        completed_keys = set(
            existing["golden_id"].astype(str) + "_" + existing["pipeline"]
        )
        logger.info(
            "Resuming RAG generation: %d already completed.",
            len(completed_keys),
        )
    else:
        existing       = pd.DataFrame(columns=RAG_COLUMNS)
        completed_keys: set = set()
        existing.to_csv(output_path, index=False)

    # Build golden_id → reference_answer lookup
    ref_lookup = dict(
        zip(
            golden_df["golden_id"].astype(str),
            golden_df["reference_answer"],
        )
    )
    query_lookup = dict(
        zip(
            golden_df["golden_id"].astype(str),
            golden_df["user_query"],
        )
    )

    pending = []
    for _, row in retrieval_results_df.iterrows():
        key = f"{row['golden_id']}_{row['pipeline']}"
        if key not in completed_keys:
            pending.append(row)

    logger.info("%d RAG generations pending.", len(pending))

    for row in tqdm(pending, desc="Generating RAG answers"):
        golden_id     = str(row["golden_id"])
        pipeline_name = row["pipeline"]
        query         = query_lookup.get(golden_id, row.get("query", ""))

        # Retrieve documents using the specified pipeline
        retriever = retrievers.get(pipeline_name)
        if retriever is None:
            logger.warning(
                "Retriever '%s' not found — skipping.", pipeline_name
            )
            continue

        docs = retriever.retrieve(query)
        context_str = format_docs(docs)

        # Generate RAG answer via Groq
        try:
            prompt_value = RAG_PROMPT.format_messages(
                context=context_str, question=query
            )
            response = llm.invoke(prompt_value)
            rag_answer = response.content.strip()

            logger.info(
                "Generated RAG answer for golden_id=%s, pipeline=%s.",
                golden_id, pipeline_name,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "RAG generation failed for golden_id=%s, pipeline=%s: %s",
                golden_id, pipeline_name, str(exc),
            )
            rag_answer = f"ERROR: {str(exc)}"

        record = {
            "golden_id":         golden_id,
            "pipeline":          pipeline_name,
            "user_query":        query,
            "retrieved_context": context_str[:2000],  # Truncate for CSV
            "rag_answer":        rag_answer,
            "reference_answer":  ref_lookup.get(golden_id, ""),
            "generated_at":      get_timestamp(),
        }
        pd.DataFrame([record]).to_csv(
            output_path, mode="a", header=False, index=False
        )

        time.sleep(request_delay)

    result_df = pd.read_csv(output_path)
    logger.info(
        "RAG generation complete: %d total answers saved to %s.",
        len(result_df), output_path,
    )
    return result_df
