"""
src/analysis/aggregators.py
============================
Pure-computation layer: takes the raw DataFrames from metrics_loader
and returns clean summary tables used by both the notebook and
report_builder.

No plotting here — only pandas / numpy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

RETRIEVAL_COLS = ["recall_at_k", "precision_at_k", "mrr", "ndcg"]


def pipeline_retrieval_summary(retrieval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a tidy summary table of retrieval metrics per pipeline,
    with an additional composite_score column (equal-weight average).
    """
    cols = [c for c in RETRIEVAL_COLS if c in retrieval_df.columns]
    summary = retrieval_df[["pipeline"] + cols].copy()
    summary["composite_score"] = summary[cols].mean(axis=1).round(4)
    summary = summary.sort_values("composite_score", ascending=False).reset_index(drop=True)
    summary["rank"] = summary.index + 1
    return summary


def best_pipeline_retrieval(retrieval_df: pd.DataFrame) -> str:
    """Return the pipeline name with the highest composite retrieval score."""
    s = pipeline_retrieval_summary(retrieval_df)
    return str(s.iloc[0]["pipeline"])


def retrieval_improvement_vs_baseline(retrieval_df: pd.DataFrame,
                                       baseline: str = "dense") -> pd.DataFrame:
    """
    For each pipeline compute % improvement over *baseline* on each metric.
    """
    cols = [c for c in RETRIEVAL_COLS if c in retrieval_df.columns]
    base_row = retrieval_df[retrieval_df["pipeline"] == baseline]
    if base_row.empty:
        return pd.DataFrame()

    base_vals = base_row[cols].values[0]
    rows = []
    for _, row in retrieval_df.iterrows():
        entry = {"pipeline": row["pipeline"]}
        for col, base in zip(cols, base_vals):
            if base and not np.isnan(base) and base != 0:
                entry[f"{col}_pct_vs_{baseline}"] = round(
                    (row[col] - base) / base * 100, 1
                )
            else:
                entry[f"{col}_pct_vs_{baseline}"] = float("nan")
        rows.append(entry)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Hallucination
# ---------------------------------------------------------------------------

HALLUC_COLS = ["include_pass", "exclude_pass", "overall_pass"]


def pipeline_hallucination_summary(halluc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hallucination pass-rates per pipeline.
    Works whether the input is per-query (long) or already aggregated.
    """
    if "pipeline" not in halluc_df.columns:
        raise ValueError("hallucination DataFrame must have a 'pipeline' column")

    cols = [c for c in HALLUC_COLS if c in halluc_df.columns]

    if halluc_df.shape[0] <= 3:
        # Already aggregated (one row per pipeline)
        return halluc_df[["pipeline"] + cols].copy()

    # Per-query format — need to aggregate
    agg_funcs = {c: "mean" for c in cols}
    summary = halluc_df.groupby("pipeline").agg(agg_funcs).reset_index()
    summary[cols] = summary[cols].round(4)
    return summary


def hallucination_risk_score(halluc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a single risk score per pipeline:
        risk = 1 - overall_pass_rate   (lower is better)
    """
    summary = pipeline_hallucination_summary(halluc_df)
    if "overall_pass" in summary.columns:
        summary["risk_score"] = (1 - summary["overall_pass"]).round(4)
    return summary.sort_values("risk_score")


# ---------------------------------------------------------------------------
# RAGAS
# ---------------------------------------------------------------------------

RAGAS_COLS = ["faithfulness", "answer_relevancy",
              "context_precision", "context_recall"]


def ragas_summary(ragas_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate RAGAS scores per pipeline; NaN allowed.
    If the input is per-query, group and take the mean.
    """
    cols = [c for c in RAGAS_COLS if c in ragas_df.columns]

    if "pipeline" not in ragas_df.columns:
        raise ValueError("RAGAS DataFrame must have a 'pipeline' column")

    if ragas_df.shape[0] <= 3:
        return ragas_df[["pipeline"] + cols].copy()

    agg = ragas_df.groupby("pipeline")[cols].mean().reset_index()
    agg[cols] = agg[cols].round(4)
    return agg


def ragas_coverage(ragas_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each RAGAS metric, report what fraction of pipelines have a score.
    Useful for showing how partial the evaluation is.
    """
    cols = [c for c in RAGAS_COLS if c in ragas_df.columns]
    records = []
    for col in cols:
        n_total = len(ragas_df)
        n_valid = ragas_df[col].notna().sum()
        records.append(
            {"metric": col,
             "n_valid": n_valid,
             "n_total": n_total,
             "coverage_pct": round(n_valid / n_total * 100, 1) if n_total else 0}
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Answer Quality (from RAG answers)
# ---------------------------------------------------------------------------

def answer_length_stats(rag_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-pipeline descriptive stats on answer word count.
    """
    if rag_df.empty or "answer" not in rag_df.columns:
        return pd.DataFrame()

    if "answer_length" not in rag_df.columns:
        rag_df = rag_df.copy()
        rag_df["answer_length"] = rag_df["answer"].astype(str).apply(
            lambda x: len(x.split())
        )

    if "pipeline" not in rag_df.columns:
        return rag_df[["answer_length"]].describe().T

    return (
        rag_df.groupby("pipeline")["answer_length"]
        .describe()[["count", "mean", "std", "min", "max"]]
        .round(1)
        .reset_index()
    )


def query_source_distribution(golden_df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of queries by dataset_source and granularity."""
    if golden_df.empty:
        return pd.DataFrame()
    cols = [c for c in ["dataset_source", "granularity"] if c in golden_df.columns]
    if not cols:
        return pd.DataFrame()
    return golden_df.groupby(cols).size().reset_index(name="count")


# ---------------------------------------------------------------------------
# Unified score table (for final comparison)
# ---------------------------------------------------------------------------

def unified_pipeline_scorecard(
    retrieval_df: pd.DataFrame,
    halluc_df: pd.DataFrame,
    ragas_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge all metrics into a single comparison table.

    Columns: pipeline | composite_retrieval | overall_pass | risk_score
             | answer_relevancy (if available) | overall_rank
    """
    # Retrieval composite
    ret_summary = pipeline_retrieval_summary(retrieval_df)[
        ["pipeline", "composite_score"]
    ].rename(columns={"composite_score": "composite_retrieval"})

    # Hallucination
    hal_summary = hallucination_risk_score(halluc_df)[
        ["pipeline", "overall_pass", "risk_score"]
    ]

    scorecard = ret_summary.merge(hal_summary, on="pipeline", how="outer")

    # RAGAS (optional / partial)
    if ragas_df is not None and not ragas_df.empty:
        rag_summary = ragas_summary(ragas_df)[
            ["pipeline"] + [c for c in ["answer_relevancy"] if c in ragas_df.columns]
        ]
        scorecard = scorecard.merge(rag_summary, on="pipeline", how="left")

    # Rank pipelines: higher retrieval + lower risk = better
    # Normalise both to [0,1] then combine
    if "composite_retrieval" in scorecard.columns:
        r_max = scorecard["composite_retrieval"].max()
        scorecard["_ret_norm"] = scorecard["composite_retrieval"] / r_max if r_max else 0

    if "risk_score" in scorecard.columns:
        scorecard["_risk_norm"] = 1 - scorecard["risk_score"]   # invert: lower risk = higher score

    norm_cols = [c for c in ["_ret_norm", "_risk_norm"] if c in scorecard.columns]
    if norm_cols:
        scorecard["overall_score"] = scorecard[norm_cols].mean(axis=1).round(4)
        scorecard = scorecard.drop(columns=norm_cols)
        scorecard = scorecard.sort_values("overall_score", ascending=False)
        scorecard["overall_rank"] = range(1, len(scorecard) + 1)

    return scorecard.reset_index(drop=True)
