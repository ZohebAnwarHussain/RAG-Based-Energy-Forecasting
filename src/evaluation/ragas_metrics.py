"""RAGAS-based evaluation metrics.

Uses the RAGAS library to compute LLM-judged quality metrics for
RAG-generated answers. The LLM judge is Llama 3.3 70B via Groq —
independent of both the KB model (Gemini 3 Flash) and the golden
dataset model (Gemini 2.5 Flash).

Metrics computed:

    faithfulness
        Does the answer contain only claims supported by the retrieved
        context? High faithfulness = low hallucination.

    answer_relevancy
        Is the answer relevant to the question asked? Measures whether
        the answer addresses the query rather than going off-topic.

    context_precision
        Are the retrieved chunks relevant to the question? Measures
        whether the retriever returned useful context.

    context_recall
        Were the expected chunks (from golden dataset) actually
        retrieved? Measures retrieval completeness.

Note: RAGAS evaluation requires API calls (LLM judge) and may take
several minutes for 93 query-pipeline combinations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def run_ragas_evaluation(
    rag_results_df: pd.DataFrame,
    golden_df: pd.DataFrame,
    llm: object,
    embeddings: object,
    output_path: Path,
) -> Optional[pd.DataFrame]:
    """Run RAGAS evaluation on all RAG answers.

    Constructs a HuggingFace Dataset from the RAG results and golden
    dataset, then runs RAGAS evaluate() with four metrics.

    Args:
        rag_results_df: DataFrame from Phase 5 with columns:
            golden_id, pipeline, user_query, rag_answer,
            retrieved_context, reference_answer.
        golden_df: Combined golden dataset DataFrame.
        llm: LLM instance for RAGAS judge (ChatGroq from get_rag_llm()).
        embeddings: Embedding model for answer_relevancy metric.
        output_path: CSV path to save RAGAS scores.

    Returns:
        DataFrame with RAGAS scores per answer, or None if RAGAS
        evaluation fails. Scores are saved to output_path regardless.

    Note:
        RAGAS API may change between versions. If import errors occur,
        check the installed ragas version with: pip show ragas
    """
    logger.info(
        "Starting RAGAS evaluation on %d RAG answers.",
        len(rag_results_df),
    )

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        logger.error(
            "RAGAS import failed: %s. "
            "Install with: pip install ragas datasets",
            str(exc),
        )
        return None

    # Build golden lookup for reference answers and expected contexts
    golden_lookup = {}
    for _, row in golden_df.iterrows():
        gid = str(row["golden_id"])
        golden_lookup[gid] = {
            "reference": str(row.get("reference_answer", "")),
            "expected_ids": json.loads(
                row.get("expected_summary_ids", "[]")
            ),
        }

    # Prepare RAGAS-compatible data
    ragas_data = {
        "question":        [],
        "answer":          [],
        "contexts":        [],
        "ground_truth":    [],
    }

    # Track metadata for output
    meta_golden_ids = []
    meta_pipelines  = []

    for _, row in rag_results_df.iterrows():
        gid = str(row["golden_id"])
        golden = golden_lookup.get(gid, {})

        # Split retrieved context into list of strings (RAGAS expects list)
        context_str = str(row.get("retrieved_context", ""))
        contexts = [
            chunk.strip()
            for chunk in context_str.split("\n\n")
            if chunk.strip()
        ]
        if not contexts:
            contexts = [context_str]

        ragas_data["question"].append(str(row["user_query"]))
        ragas_data["answer"].append(str(row["rag_answer"]))
        ragas_data["contexts"].append(contexts)
        ragas_data["ground_truth"].append(
            golden.get("reference", "")
        )

        meta_golden_ids.append(gid)
        meta_pipelines.append(row["pipeline"])

    dataset = Dataset.from_dict(ragas_data)

    # Run RAGAS evaluation
    try:
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=llm,
            embeddings=embeddings,
        )

        scores_df = result.to_pandas()
        scores_df["golden_id"] = meta_golden_ids
        scores_df["pipeline"]  = meta_pipelines

        scores_df.to_csv(output_path, index=False)

        # Log pipeline-level averages
        metric_cols = [
            "faithfulness", "answer_relevancy",
            "context_precision", "context_recall",
        ]
        available_cols = [
            c for c in metric_cols if c in scores_df.columns
        ]

        print("\n" + "=" * 65)
        print("  RAGAS METRICS BY PIPELINE")
        print("=" * 65)
        if available_cols:
            summary = scores_df.groupby("pipeline")[available_cols].mean()
            print(summary.round(4).to_string())
        else:
            print("  No RAGAS metric columns found in output.")
        print("=" * 65)

        logger.info("RAGAS evaluation saved to %s.", output_path)
        return scores_df

    except Exception as exc:  # noqa: BLE001
        logger.error("RAGAS evaluation failed: %s", str(exc))

        # Save partial results if available
        partial = pd.DataFrame({
            "golden_id": meta_golden_ids,
            "pipeline":  meta_pipelines,
            "error":     str(exc),
        })
        partial.to_csv(output_path, index=False)
        logger.info("Partial results saved to %s.", output_path)

        return None
