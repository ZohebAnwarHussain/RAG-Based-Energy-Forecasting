"""Gemini 2.5 Flash client and golden dataset generation functions.

Three components:

    configure_gemini_golden()     Initialise and return a configured client.
                                  Uses the same key resolution chain as the
                                  KB pipeline (env → Colab secrets → getpass).

    call_gemini_golden()          Send one query + KB context to Gemini 2.5
                                  Flash and return the reference answer.
                                  Implements exponential back-off retry.

    generate_golden_dataset()     Iterate over a query list, select KB context
                                  per query, call Gemini to generate a
                                  reference answer, and append each result to
                                  the output CSV immediately. Supports seamless
                                  resume via golden_id deduplication.

    build_combined_golden_dataset() Merge all three per-source golden CSVs
                                  into the combined master golden dataset.

Model: gemini-2.5-flash (stable production release)
Reasoning: Different Gemini generation from KB model (gemini-3-flash-preview)
           reduces style-matching contamination in faithfulness evaluation.
"""

from __future__ import annotations

import getpass
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from google import genai
from google.genai import types
from tqdm.auto import tqdm

from config import (
    GOLDEN_MAX_TOKENS,
    GOLDEN_MODEL_NAME,
    GOLDEN_TEMPERATURE,
    MAX_RETRIES,
    REQUEST_DELAY_SECONDS,
    RETRY_BACKOFF_SECONDS,
)
from src.utils.timestamps import get_timestamp

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Golden Dataset CSV Schema
# ─────────────────────────────────────────────────────────────────────────────

GOLDEN_CSV_COLUMNS: List[str] = [
    "golden_id",
    "dataset_source",
    "query_type",
    "difficulty_level",
    "query_scope",
    "granularity_target",
    "user_query",
    "expected_summary_ids",
    "expected_primary_summary_id",
    "expected_context_summary",
    "reference_answer",
    "answer_must_include",
    "answer_must_not_include",
    "retrieval_strategy_target",
    "retrieval_notes",
    "evaluation_notes",
    "generated_at",
    "generated_by",
]

# ─────────────────────────────────────────────────────────────────────────────
# System prompt used for every reference answer generation call
# ─────────────────────────────────────────────────────────────────────────────

GOLDEN_SYSTEM_PROMPT: str = (
    "You are an expert energy systems analyst and RAG evaluation specialist.\n"
    "Your task is to generate high-quality reference answers for a golden "
    "evaluation dataset.\n\n"
    "You will be given:\n"
    "1. A natural language query about energy consumption patterns\n"
    "2. Relevant knowledge base summaries retrieved from historical energy data\n\n"
    "Rules for generating reference answers:\n"
    "- Base your answer STRICTLY on the provided knowledge base summaries\n"
    "- Do NOT introduce facts, numbers, or claims not present in the summaries\n"
    "- Be factual, precise and use specific numbers where available\n"
    "- Write 3-5 sentences in clear, stakeholder-friendly language\n"
    "- If summaries lack enough information, explicitly state what is missing\n"
    "- Do NOT hallucinate or speculate beyond the provided evidence"
)


# ─────────────────────────────────────────────────────────────────────────────
# Gemini Client Configuration
# ─────────────────────────────────────────────────────────────────────────────

def configure_gemini_golden(api_key: Optional[str] = None) -> genai.Client:
    """Initialise and return the Gemini client for golden dataset generation.

    Resolves the API key using the following priority order:
        1. Explicit ``api_key`` argument (for testing)
        2. ``GEMINI_API_KEY`` environment variable (.env)
        3. Google Colab userdata secrets (Colab only)
        4. Interactive getpass prompt as final fallback

    Args:
        api_key: Optional explicit API key override.

    Returns:
        Configured genai.Client instance ready for calls to
        ``GOLDEN_MODEL_NAME`` (gemini-2.5-flash).

    Raises:
        ValueError: If no API key can be resolved from any source.
    """
    if api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            logger.info(
                "Gemini API key loaded from environment variable."
            )

    if not api_key:
        try:
            from google.colab import userdata  # noqa: PLC0415
            api_key = userdata.get("GEMINI_API_KEY")
            if api_key:
                logger.info("Gemini API key loaded from Colab Secrets.")
        except (ImportError, Exception):  # noqa: BLE001
            pass

    if not api_key:
        logger.info(
            "API key not found in environment. Prompting for manual entry."
        )
        api_key = getpass.getpass("Enter your Gemini API key: ")

    if not api_key:
        raise ValueError(
            "No Gemini API key found. "
            "Add GEMINI_API_KEY to your .env file or Colab Secrets."
        )

    client = genai.Client(api_key=api_key)
    logger.info(
        "Gemini golden client configured for model '%s'.", GOLDEN_MODEL_NAME
    )
    return client


# Generation config used for every reference answer call
GOLDEN_GENERATION_CONFIG = types.GenerateContentConfig(
    temperature=GOLDEN_TEMPERATURE,
    max_output_tokens=GOLDEN_MAX_TOKENS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Reference Answer Generation
# ─────────────────────────────────────────────────────────────────────────────

def _build_reference_prompt(query: str, kb_context: str) -> str:
    """Combine system instructions, KB context, and query into one prompt.

    Args:
        query: Natural language query string.
        kb_context: Concatenated KB summary text used as evidence.

    Returns:
        Formatted prompt string ready for Gemini 2.5 Flash.
    """
    return (
        f"{GOLDEN_SYSTEM_PROMPT}\n\n"
        f"Knowledge Base Summaries (use ONLY these as evidence):\n"
        f"---\n{kb_context}\n---\n\n"
        f"Query: {query}\n\n"
        f"Generate a factual reference answer based strictly on the "
        f"above summaries."
    )


def call_gemini_golden(
    client: genai.Client,
    query: str,
    kb_context: str,
    query_id: str,
    max_retries: int = MAX_RETRIES,
    retry_backoff: float = RETRY_BACKOFF_SECONDS,
) -> Optional[str]:
    """Call Gemini 2.5 Flash to generate one reference answer.

    Sends the combined system prompt, KB context, and user query to
    Gemini 2.5 Flash. Implements exponential back-off retry for
    transient errors (e.g. 503 during free-tier high-load periods).

    Args:
        client: Configured genai.Client from configure_gemini_golden().
        query: Natural language query string.
        kb_context: Concatenated KB summary text used as evidence.
        query_id: Unique query identifier for logging.
        max_retries: Maximum retry attempts before permanent failure.
        retry_backoff: Base wait seconds before first retry. Doubles
            on subsequent attempts.

    Returns:
        Generated reference answer text (stripped) on success,
        or None if all retry attempts are exhausted.
    """
    prompt = _build_reference_prompt(query, kb_context)

    for attempt in range(1, max_retries + 1):
        t0 = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=GOLDEN_MODEL_NAME,
                contents=prompt,
                config=GOLDEN_GENERATION_CONFIG,
            )
            latency = time.perf_counter() - t0
            logger.info(
                "Generated reference answer for '%s' in %.2fs.",
                query_id, latency,
            )
            return response.text.strip()

        except Exception as exc:  # noqa: BLE001
            latency = time.perf_counter() - t0
            logger.warning(
                "Attempt %d/%d failed for '%s': %s",
                attempt, max_retries, query_id, str(exc),
            )
            if attempt < max_retries:
                wait = retry_backoff * (2 ** (attempt - 1))
                logger.info("Waiting %.0f s before retry...", wait)
                time.sleep(wait)
            else:
                logger.error(
                    "Permanently failed for query '%s'.", query_id
                )

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Generation and Assembly
# ─────────────────────────────────────────────────────────────────────────────

def generate_golden_dataset(
    client: genai.Client,
    query_list: List[Dict[str, Any]],
    kb_df: pd.DataFrame,
    dataset_source: str,
    output_path: Path,
    context_selector_fn: Any,
    request_delay: float = REQUEST_DELAY_SECONDS,
) -> pd.DataFrame:
    """Generate the golden dataset for one query list.

    For each query in ``query_list``, selects relevant KB context using
    ``context_selector_fn``, calls Gemini 2.5 Flash to generate a
    reference answer, and appends the complete record to ``output_path``
    immediately. Supports resume — already-generated golden_ids are
    detected and skipped automatically.

    Args:
        client: Configured genai.Client instance.
        query_list: List of query metadata dicts from the query bank.
        kb_df: KB DataFrame for context selection. For single-dataset
            queries this should be the dataset-filtered KB. For
            cross-scale queries pass the full master KB.
        dataset_source: Dataset tag for golden_id construction.
            One of 'gefcom', 'household', 'cross_scale'.
        output_path: CSV path for incremental output. Created with
            headers if it does not exist.
        context_selector_fn: Callable that takes (query_meta, kb_df)
            and returns (context_str, all_ids, primary_id). Pass either
            select_kb_context or select_cross_scale_context.
        request_delay: Seconds to sleep between API calls.

    Returns:
        Complete golden dataset DataFrame loaded from output_path,
        including previously generated and newly generated entries.
    """
    # ── Resume support ────────────────────────────────────────────────────────
    if output_path.exists():
        existing      = pd.read_csv(output_path)
        completed_ids = set(existing["golden_id"].astype(str).tolist())
        logger.info(
            "Resuming '%s': %d already completed.",
            dataset_source, len(completed_ids),
        )
    else:
        existing      = pd.DataFrame(columns=GOLDEN_CSV_COLUMNS)
        completed_ids: set = set()
        existing.to_csv(output_path, index=False)

    # Assign stable golden_ids to all queries
    for i, q in enumerate(query_list):
        if "golden_id" not in q:
            q["golden_id"] = (
                f"{dataset_source}_{q.get('query_type', 'q')}_{i + 1:03d}"
            )

    pending = [
        q for q in query_list
        if q["golden_id"] not in completed_ids
    ]
    logger.info(
        "%d queries pending for '%s'.", len(pending), dataset_source
    )

    for q in tqdm(pending, desc=f"Generating {dataset_source} answers"):
        query_id = q["golden_id"]

        context_str, all_ids, primary_id = context_selector_fn(q, kb_df)

        reference_answer = call_gemini_golden(
            client=client,
            query=q["user_query"],
            kb_context=context_str,
            query_id=query_id,
        )

        if reference_answer is not None:
            record = {
                "golden_id":                   query_id,
                "dataset_source":              dataset_source,
                "query_type":                  q.get("query_type", ""),
                "difficulty_level":            q.get("difficulty_level", "medium"),
                "query_scope":                 q.get("query_scope", ""),
                "granularity_target":          q.get("granularity_target", "mixed"),
                "user_query":                  q["user_query"],
                "expected_summary_ids":        json.dumps(all_ids),
                "expected_primary_summary_id": primary_id,
                "expected_context_summary":    context_str,
                "reference_answer":            reference_answer,
                "answer_must_include":         json.dumps(
                    q.get("answer_must_include", [])
                ),
                "answer_must_not_include":     json.dumps(
                    q.get("answer_must_not_include", [])
                ),
                "retrieval_strategy_target":   q.get(
                    "retrieval_strategy_target", "all"
                ),
                "retrieval_notes":             q.get("retrieval_notes", ""),
                "evaluation_notes":            q.get("evaluation_notes", ""),
                "generated_at":                get_timestamp(),
                "generated_by":                GOLDEN_MODEL_NAME,
            }
            pd.DataFrame([record]).to_csv(
                output_path, mode="a", header=False, index=False
            )

        time.sleep(request_delay)

    result_df = pd.read_csv(output_path)
    logger.info(
        "Golden dataset '%s' complete: %d entries.",
        dataset_source, len(result_df),
    )
    return result_df


def build_combined_golden_dataset(
    golden_dfs: List[pd.DataFrame],
    output_path: Path,
) -> pd.DataFrame:
    """Merge all per-source golden datasets into the combined master CSV.

    Concatenates the three individual golden datasets, deduplicates by
    the original string golden_id, and reassigns clean sequential
    integer golden_ids (1, 2, 3...) for downstream evaluation pipelines.

    Args:
        golden_dfs: List of per-source DataFrames
            (gefcom, household, cross_scale).
        output_path: Destination path for combined_golden_dataset.csv.

    Returns:
        Combined DataFrame with sequential integer golden_ids.
        Written to output_path.

    Example:
        >>> combined = build_combined_golden_dataset(
        ...     [gefcom_df, household_df, cross_scale_df],
        ...     PATHS["golden_dataset"] / "combined_golden_dataset.csv",
        ... )
        >>> len(combined)
        50
    """
    valid = [df for df in golden_dfs if df is not None and not df.empty]
    if not valid:
        logger.warning("No golden DataFrames to merge.")
        return pd.DataFrame()

    combined = pd.concat(valid, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["golden_id"]
    ).reset_index(drop=True)

    # Reassign clean sequential integer IDs
    combined["golden_id"] = range(1, len(combined) + 1)
    combined = combined[GOLDEN_CSV_COLUMNS]
    combined.to_csv(output_path, index=False)

    logger.info(
        "Combined golden dataset saved: %d total queries → %s",
        len(combined), output_path,
    )
    return combined
