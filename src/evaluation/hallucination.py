"""Hallucination detection via keyword guards.

Lightweight complement to RAGAS faithfulness scoring. Uses two fields
from the golden dataset:

    answer_must_include
        List of terms/facts that a correct answer must contain.
        Missing terms suggest retrieval failure or insufficient grounding.

    answer_must_not_include
        List of forbidden terms that indicate hallucination or domain
        bleed (e.g. household terms in a GEFCom answer, or speculative
        phrases like "I think" or "probably").

This is not a substitute for RAGAS faithfulness — it is a fast binary
check that catches obvious failures before the more expensive LLM-based
RAGAS evaluation runs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def check_hallucination(
    answer: str,
    must_include: List[str],
    must_not_include: List[str],
) -> Dict[str, Any]:
    """Check a single RAG answer against keyword inclusion/exclusion rules.

    Args:
        answer: RAG-generated answer text to check.
        must_include: List of terms that must appear in the answer.
            Case-insensitive matching.
        must_not_include: List of forbidden terms that must NOT appear.
            Case-insensitive matching.

    Returns:
        Dict with keys:
            missing_terms:   List of must_include terms not found
            forbidden_terms: List of must_not_include terms found
            include_pass:    True if all must_include terms are present
            exclude_pass:    True if no must_not_include terms are found
            overall_pass:    True if both checks pass

    Example:
        >>> result = check_hallucination(
        ...     "Zone 4 had peak demand of 800 MW in winter",
        ...     must_include=["MW", "peak", "Zone 4"],
        ...     must_not_include=["household", "I think"],
        ... )
        >>> result["overall_pass"]
        True
    """
    answer_lower = answer.lower()

    missing_terms = [
        term for term in must_include
        if term.lower() not in answer_lower
    ]
    forbidden_found = [
        term for term in must_not_include
        if term.lower() in answer_lower
    ]

    include_pass = len(missing_terms) == 0
    exclude_pass = len(forbidden_found) == 0

    return {
        "missing_terms":   missing_terms,
        "forbidden_terms": forbidden_found,
        "include_pass":    include_pass,
        "exclude_pass":    exclude_pass,
        "overall_pass":    include_pass and exclude_pass,
    }


def compute_hallucination_rate(
    rag_results_df: pd.DataFrame,
    golden_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Compute hallucination checks for all RAG answers.

    Merges RAG answers with golden dataset to access the must_include
    and must_not_include fields, then runs check_hallucination() on
    each answer.

    Args:
        rag_results_df: DataFrame from Phase 5 with columns:
            golden_id, pipeline, rag_answer.
        golden_df: Combined golden dataset with columns:
            golden_id, answer_must_include (JSON), answer_must_not_include (JSON).
        output_path: CSV path to save hallucination check results.

    Returns:
        DataFrame with hallucination check results per answer, including
        missing_terms, forbidden_terms, include_pass, exclude_pass,
        overall_pass.
    """
    logger.info(
        "Running hallucination checks on %d RAG answers.",
        len(rag_results_df),
    )

    # Build golden lookup
    golden_lookup: Dict[str, Dict[str, List[str]]] = {}
    for _, row in golden_df.iterrows():
        gid = str(row["golden_id"])
        golden_lookup[gid] = {
            "must_include": json.loads(
                row.get("answer_must_include", "[]")
            ),
            "must_not_include": json.loads(
                row.get("answer_must_not_include", "[]")
            ),
        }

    results: List[Dict[str, Any]] = []

    for _, row in rag_results_df.iterrows():
        gid    = str(row["golden_id"])
        answer = str(row.get("rag_answer", ""))
        golden = golden_lookup.get(gid, {})

        check = check_hallucination(
            answer=answer,
            must_include=golden.get("must_include", []),
            must_not_include=golden.get("must_not_include", []),
        )

        results.append({
            "golden_id":       gid,
            "pipeline":        row["pipeline"],
            "include_pass":    check["include_pass"],
            "exclude_pass":    check["exclude_pass"],
            "overall_pass":    check["overall_pass"],
            "missing_terms":   json.dumps(check["missing_terms"]),
            "forbidden_terms": json.dumps(check["forbidden_terms"]),
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    # Log summary
    print("\n" + "=" * 60)
    print("  HALLUCINATION CHECK BY PIPELINE")
    print("=" * 60)
    for pipeline in results_df["pipeline"].unique():
        subset = results_df[results_df["pipeline"] == pipeline]
        include_rate = subset["include_pass"].mean() * 100
        exclude_rate = subset["exclude_pass"].mean() * 100
        overall_rate = subset["overall_pass"].mean() * 100
        print(
            f"  {pipeline:<15} "
            f"include_pass={include_rate:5.1f}%  "
            f"exclude_pass={exclude_rate:5.1f}%  "
            f"overall_pass={overall_rate:5.1f}%"
        )
    print("=" * 60)

    logger.info("Hallucination checks saved to %s.", output_path)
    return results_df
