"""
experiments/runner.py
======================
Shared experiment execution engine.

Every experiment module exposes a run() function with this signature:

    def run(queries, retriever, llm_client, config) -> ExperimentResult

This runner handles:
  - Iterating over queries
  - Timing each call (latency)
  - Persisting results to outputs/experiments/{exp_id}/
  - Returning a clean ExperimentResult dataclass
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    query_id:       str
    question:       str
    pipeline:       str
    top_k:          int
    answer:         str
    retrieved_docs: list[dict] = field(default_factory=list)
    metrics:        dict       = field(default_factory=dict)
    latency_sec:    float      = 0.0
    error:          str        = ""


@dataclass
class ExperimentResult:
    exp_id:          str
    pipeline:        str
    top_k:           int
    query_results:   list[QueryResult] = field(default_factory=list)
    agg_metrics:     dict              = field(default_factory=dict)
    total_queries:   int               = 0
    total_errors:    int               = 0
    total_time_sec:  float             = 0.0

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for qr in self.query_results:
            row = {
                "exp_id":      self.exp_id,
                "query_id":    qr.query_id,
                "question":    qr.question,
                "pipeline":    qr.pipeline,
                "top_k":       qr.top_k,
                "answer":      qr.answer,
                "latency_sec": qr.latency_sec,
                "error":       qr.error,
            }
            row.update(qr.metrics)
            rows.append(row)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_experiment(
    exp_id:       str,
    pipeline:     str,
    top_k:        int,
    queries:      list[dict],
    generate_fn:  Callable[[dict, list[dict], int], dict],
    outputs_dir:  str | Path = "outputs/experiments",
    log_every:    int = 5,
) -> ExperimentResult:
    """
    Run one experiment (one pipeline × one K value) over all queries.

    Parameters
    ----------
    exp_id       : e.g. 'EXP_01_NO_RAG_LLM'
    pipeline     : e.g. 'dense', 'hybrid', 'no_rag'
    top_k        : number of docs to retrieve (0 for no-RAG)
    queries      : list of dicts with at least 'query_id' and 'question'
    generate_fn  : callable(query_dict, retrieved_docs, top_k) → dict
                   must return {'answer': str, 'retrieved_docs': list,
                                'metrics': dict}
    outputs_dir  : where to save results
    log_every    : log progress every N queries

    Returns
    -------
    ExperimentResult
    """
    outputs_dir = Path(outputs_dir) / exp_id / f"k{top_k}"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    result = ExperimentResult(
        exp_id=exp_id,
        pipeline=pipeline,
        top_k=top_k,
    )

    t_start = time.time()

    for i, query in enumerate(queries):
        query_id = query.get("query_id", f"q{i:03d}")
        question = query.get("question", query.get("user_query", query.get("query", "")))

        if i > 0 and i % log_every == 0:
            elapsed = time.time() - t_start
            logger.info(
                "[%s | k=%d] %d/%d queries done  (%.1fs elapsed)",
                exp_id, top_k, i, len(queries), elapsed,
            )

        t0 = time.time()
        try:
            out = generate_fn(query, [], top_k)
            latency = time.time() - t0

            qr = QueryResult(
                query_id=query_id,
                question=question,
                pipeline=pipeline,
                top_k=top_k,
                answer=out.get("answer", ""),
                retrieved_docs=out.get("retrieved_docs", []),
                metrics=out.get("metrics", {}),
                latency_sec=round(latency, 3),
            )

        except Exception as exc:
            latency = time.time() - t0
            logger.error("[%s] query %s failed: %s", exp_id, query_id, exc)
            qr = QueryResult(
                query_id=query_id,
                question=question,
                pipeline=pipeline,
                top_k=top_k,
                answer="",
                latency_sec=round(latency, 3),
                error=str(exc),
            )
            result.total_errors += 1

        result.query_results.append(qr)

    result.total_queries  = len(queries)
    result.total_time_sec = round(time.time() - t_start, 2)

    # Persist individual query results
    _save_results(result, outputs_dir)

    logger.info(
        "[%s | k=%d] Complete — %d queries, %d errors, %.1fs",
        exp_id, top_k, result.total_queries,
        result.total_errors, result.total_time_sec,
    )
    return result


# ---------------------------------------------------------------------------
# Multi-K runner (runs same experiment at K=3, K=5, K=10)
# ---------------------------------------------------------------------------

def run_experiment_multi_k(
    exp_id:      str,
    pipeline:    str,
    k_values:    list[int],
    queries:     list[dict],
    generate_fn: Callable,
    outputs_dir: str | Path = "outputs/experiments",
) -> list[ExperimentResult]:
    """
    Run the same experiment across multiple K values.
    Returns one ExperimentResult per K value.
    """
    results = []
    for k in k_values:
        logger.info("─── Running %s at K=%d ───", exp_id, k)
        res = run_experiment(
            exp_id=exp_id,
            pipeline=pipeline,
            top_k=k,
            queries=queries,
            generate_fn=generate_fn,
            outputs_dir=outputs_dir,
        )
        results.append(res)
    return results


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _save_results(result: ExperimentResult, output_dir: Path) -> None:
    """Save query-level JSON and aggregate CSV."""
    # Full JSON (all fields)
    json_path = output_dir / "query_results.json"
    rows = [asdict(qr) for qr in result.query_results]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    # Flat CSV
    df = result.to_dataframe()
    df.to_csv(output_dir / "query_results.csv", index=False)

    # Aggregate metrics
    agg_path = output_dir / "agg_metrics.json"
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(result.agg_metrics, f, indent=2, ensure_ascii=False)


def load_experiment_result(exp_dir: str | Path) -> pd.DataFrame:
    """Load a previously saved experiment result as a DataFrame."""
    exp_dir = Path(exp_dir)
    csv_path = exp_dir / "query_results.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    json_path = exp_dir / "query_results.json"
    if json_path.exists():
        with open(json_path) as f:
            return pd.DataFrame(json.load(f))
    raise FileNotFoundError(f"No results found in {exp_dir}")
