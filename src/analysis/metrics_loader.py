"""
src/analysis/metrics_loader.py
================================
Loads all evaluation artifacts produced by Notebooks 04, 05, and 06
into clean pandas DataFrames ready for analysis.

Handles:
  - Retrieval metrics  (CSV / JSON from notebook 04 / 06)
  - Hallucination checks (CSV / JSON from notebook 06)
  - RAGAS metrics       (CSV / JSON from notebook 06, partial NaN allowed)
  - RAG answers         (JSON from notebook 05)
  - Golden dataset      (JSON from notebook 02)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_file(directory: Path, patterns: list[str]) -> Optional[Path]:
    """Return the first file in *directory* matching any glob pattern."""
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[-1]          # most recent if multiple
    return None


def _load_json_or_csv(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Accept list-of-dicts or dict-of-lists
    if isinstance(data, list):
        return pd.DataFrame(data)
    return pd.DataFrame.from_dict(data)


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_retrieval_metrics(outputs_dir: Path) -> pd.DataFrame:
    """
    Load per-pipeline retrieval metrics.

    Expected columns (at minimum):
        pipeline, recall_at_k, precision_at_k, mrr, ndcg
    """
    retrieval_dir = outputs_dir / "retrieval_results"
    path = _find_file(
        retrieval_dir,
        ["retrieval_metrics*.csv", "retrieval_metrics*.json",
         "*retrieval*metrics*.csv", "*retrieval*metrics*.json"]
    )

    if path is None:
        # Fallback: build from the known pilot results so the notebook
        # still runs even if the CSV was not saved explicitly.
        logger.warning(
            "retrieval_metrics file not found in %s — using hard-coded pilot values.",
            retrieval_dir,
        )
        return pd.DataFrame(
            {
                "pipeline":      ["dense", "hybrid", "hierarchical"],
                "recall_at_k":   [0.060,   0.035,    0.081],
                "precision_at_k":[0.060,   0.035,    0.080],
                "mrr":           [0.091,   0.044,    0.159],
                "ndcg":          [0.051,   0.032,    0.075],
                "n_queries":     [50,      23,       20],
            }
        )

    df = _load_json_or_csv(path)
    logger.info("Loaded retrieval metrics from %s  shape=%s", path, df.shape)
    return df


def load_hallucination_results(outputs_dir: Path) -> pd.DataFrame:
    """
    Load per-query hallucination check results.

    Expected columns (at minimum):
        pipeline, query_id, include_pass, exclude_pass, overall_pass
    """
    eval_dir = outputs_dir / "evaluation"
    path = _find_file(
        eval_dir,
        ["hallucination*.csv", "hallucination*.json",
         "*halluc*.csv", "*halluc*.json"]
    )

    if path is None:
        logger.warning(
            "hallucination file not found in %s — using hard-coded pilot values.",
            eval_dir,
        )
        return pd.DataFrame(
            {
                "pipeline":     ["dense", "hybrid", "hierarchical"],
                "include_pass": [0.60,    0.565,    0.65],
                "exclude_pass": [0.98,    0.957,    0.95],
                "overall_pass": [0.58,    0.522,    0.60],
                "n_queries":    [50,      23,       20],
            }
        )

    df = _load_json_or_csv(path)
    logger.info("Loaded hallucination results from %s  shape=%s", path, df.shape)
    return df


def load_ragas_metrics(outputs_dir: Path) -> pd.DataFrame:
    """
    Load RAGAS metrics.  NaN values are expected for partial runs.

    Expected columns (subset of):
        pipeline, query_id, faithfulness, answer_relevancy,
        context_precision, context_recall
    """
    eval_dir = outputs_dir / "evaluation"
    path = _find_file(
        eval_dir,
        ["ragas*.csv", "ragas*.json", "*ragas*.csv", "*ragas*.json"]
    )

    if path is None:
        logger.warning(
            "RAGAS file not found in %s — using partial pilot values.", eval_dir
        )
        # Only the one score that succeeded (dense answer_relevancy = 0.822)
        return pd.DataFrame(
            {
                "pipeline":         ["dense", "hybrid", "hierarchical"],
                "faithfulness":     [float("nan"), float("nan"), float("nan")],
                "answer_relevancy": [0.822,        float("nan"), float("nan")],
                "context_precision":[float("nan"), float("nan"), float("nan")],
                "context_recall":   [float("nan"), float("nan"), float("nan")],
            }
        )

    df = _load_json_or_csv(path)
    logger.info("Loaded RAGAS metrics from %s  shape=%s", path, df.shape)
    return df


def load_rag_answers(outputs_dir: Path) -> pd.DataFrame:
    """
    Load the 93 RAG answers generated in Notebook 05.

    Expected columns (at minimum):
        query_id, pipeline, question, answer, context_docs
    """
    rag_dir = outputs_dir / "rag_results"
    path = _find_file(
        rag_dir,
        ["rag_answers*.json", "rag_answers*.csv",
         "*answers*.json", "*answers*.csv",
         "rag_results*.json", "rag_results*.csv"]
    )

    if path is None:
        logger.warning("RAG answers file not found in %s.", rag_dir)
        return pd.DataFrame(
            columns=["query_id", "pipeline", "question", "answer", "answer_length"]
        )

    df = _load_json_or_csv(path)

    # Derive answer_length if not present
    if "answer_length" not in df.columns and "answer" in df.columns:
        df["answer_length"] = df["answer"].astype(str).apply(lambda x: len(x.split()))

    logger.info("Loaded RAG answers from %s  shape=%s", path, df.shape)
    return df


def load_golden_dataset(outputs_dir: Path) -> pd.DataFrame:
    """
    Load the 50-query golden dataset produced in Notebook 02.

    Expected columns (at minimum):
        query_id, question, ground_truth, dataset_source, granularity
    """
    gd_dir = outputs_dir / "golden_dataset"
    path = _find_file(
        gd_dir,
        ["golden_dataset*.json", "golden_dataset*.csv",
         "*golden*.json", "*golden*.csv"]
    )

    if path is None:
        logger.warning("Golden dataset file not found in %s.", gd_dir)
        return pd.DataFrame(
            columns=["query_id", "question", "ground_truth",
                     "dataset_source", "granularity"]
        )

    df = _load_json_or_csv(path)
    logger.info("Loaded golden dataset from %s  shape=%s", path, df.shape)
    return df


# ---------------------------------------------------------------------------
# Convenience: load everything at once
# ---------------------------------------------------------------------------

def load_all(outputs_dir: str | Path) -> dict[str, pd.DataFrame]:
    """
    Load every evaluation artefact and return as a named dict.

    Usage
    -----
    >>> from src.analysis.metrics_loader import load_all
    >>> data = load_all("outputs")
    >>> data["retrieval"].head()
    """
    outputs_dir = Path(outputs_dir)

    return {
        "retrieval":     load_retrieval_metrics(outputs_dir),
        "hallucination": load_hallucination_results(outputs_dir),
        "ragas":         load_ragas_metrics(outputs_dir),
        "rag_answers":   load_rag_answers(outputs_dir),
        "golden":        load_golden_dataset(outputs_dir),
    }
