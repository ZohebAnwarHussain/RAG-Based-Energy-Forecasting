"""
experiments/exp_10_final_comparison.py
========================================
EXP_10_FINAL_COMPARISON — Rank all RAG methods using common metrics.

No new generation. Loads agg_metrics.json and ragas_scores.csv from
EXP_01 through EXP_09 and produces:
  - Table 1: Overall RAG Performance (all experiments)
  - Table 2: Retrieval Quality (EXP_02–09)
  - Table 3: RAGAS Scores (EXP_01–09)
  - Table 4: Novelty 1 — Attribution Metrics (EXP_05–07)
  - Table 5: Novelty 2 — Difficulty Metrics (EXP_08–09)
  - Table 6: Overall Ranking by composite score
  - ragas_scores_merged.csv — merged RAGAS scores across all experiments

Usage (in notebook)
--------------------
    from experiments.exp_10_final_comparison import run_exp_10
    tables = run_exp_10(outputs_dir=EXP_OUTPUTS_DIR, k=5)
    print(tables["table1_overall"])
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Experiment registry — all experiments and their K values
# ---------------------------------------------------------------------------

ALL_EXPERIMENTS = [
    "EXP_01_NO_RAG_LLM",
    "EXP_02_DENSE_RAG",
    "EXP_03_HYBRID_RAG",
    "EXP_04_HIERARCHICAL_RAG",
    "EXP_05_DENSE_RAG_ATTRIBUTION",
    "EXP_06_HYBRID_RAG_ATTRIBUTION",
    "EXP_07_HIERARCHICAL_RAG_ATTRIBUTION",
    "EXP_08_QUERY_DIFFICULTY_DENSE_HYBRID",
    "EXP_09_QUERY_DIFFICULTY_HIERARCHICAL",
]

NO_RAG_EXPERIMENTS   = {"EXP_01_NO_RAG_LLM"}
ATTRIBUTION_EXPS     = {
    "EXP_05_DENSE_RAG_ATTRIBUTION",
    "EXP_06_HYBRID_RAG_ATTRIBUTION",
    "EXP_07_HIERARCHICAL_RAG_ATTRIBUTION",
}
DIFFICULTY_EXPS      = {
    "EXP_08_QUERY_DIFFICULTY_DENSE_HYBRID",
    "EXP_09_QUERY_DIFFICULTY_HIERARCHICAL",
}

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_agg_metrics(outputs_dir: Path, exp_id: str, k: int) -> dict | None:
    path = outputs_dir / exp_id / f"k{k}" / "agg_metrics.json"
    if not path.exists():
        logger.warning("agg_metrics.json not found: %s", path)
        return None
    with open(path) as f:
        return json.load(f)


def _load_ragas_scores(outputs_dir: Path, exp_id: str, k: int) -> pd.DataFrame | None:
    path = outputs_dir / exp_id / f"k{k}" / "ragas_scores.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["exp_id"] = exp_id
    df["k"]      = k
    return df


def _load_query_results(outputs_dir: Path, exp_id: str, k: int) -> pd.DataFrame | None:
    path = outputs_dir / exp_id / f"k{k}" / "query_results.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["exp_id"] = exp_id
    df["k"]      = k
    return df


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def _build_table1(agg_rows: list[dict]) -> pd.DataFrame:
    """Table 1: Overall RAG Performance."""
    cols = [
        "exp_id", "top_k", "pipeline", "run_mode",
        "pct_useful", "avg_answer_relevance", "avg_semantic_similarity",
        "avg_faithfulness", "avg_hallucination_rate",
        "avg_insight_clarity", "avg_latency_sec",
        "n_queries", "n_valid", "n_errors",
    ]
    rows = []
    for r in agg_rows:
        rows.append({c: r.get(c) for c in cols})
    df = pd.DataFrame(rows)
    df = df.sort_values(["top_k", "exp_id"]).reset_index(drop=True)
    return df


def _build_table2(agg_rows: list[dict]) -> pd.DataFrame:
    """Table 2: Retrieval Quality — EXP_02 onwards only."""
    cols = [
        "exp_id", "top_k", "pipeline", "run_mode",
        "avg_recall_at_k", "avg_precision_at_k",
        "avg_mrr", "avg_ndcg_at_k",
        "avg_relevant_available", "avg_relevant_retrieved",
    ]
    rows = []
    for r in agg_rows:
        if r.get("exp_id") in NO_RAG_EXPERIMENTS:
            continue
        rows.append({c: r.get(c) for c in cols})
    df = pd.DataFrame(rows)
    df = df.sort_values(["top_k", "exp_id"]).reset_index(drop=True)
    return df


def _build_table3(ragas_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Table 3: RAGAS Scores averaged per experiment × K."""
    if not ragas_dfs:
        return pd.DataFrame()
    combined = pd.concat(ragas_dfs, ignore_index=True)
    ragas_cols = ["faithfulness", "answer_relevancy",
                  "context_precision", "context_recall"]
    existing = [c for c in ragas_cols if c in combined.columns]
    agg = (
        combined.groupby(["exp_id", "k"])[existing]
        .agg(["mean", "count"])
        .round(4)
    )
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg.reset_index()


def _build_table4(agg_rows: list[dict]) -> pd.DataFrame:
    """Table 4: Novelty 1 — Attribution Metrics (EXP_05–07)."""
    cols = [
        "exp_id", "top_k", "pipeline",
        "avg_attribution_coverage",
        "avg_citation_accuracy",
        "avg_unsupported_claim_rate",
        "avg_total_claims",
        "avg_claims_with_citation",
        "avg_correct_citations",
        "avg_spurious_citations",
    ]
    rows = []
    for r in agg_rows:
        if r.get("exp_id") not in ATTRIBUTION_EXPS:
            continue
        rows.append({c: r.get(c) for c in cols})
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    df = df.sort_values(["top_k", "exp_id"]).reset_index(drop=True)
    return df


def _build_table5(agg_rows: list[dict]) -> pd.DataFrame:
    """Table 5: Novelty 2 — Difficulty Metrics (EXP_08–09)."""
    cols = [
        "exp_id", "top_k", "pipeline",
        "avg_coverage_score", "avg_consistency_score", "avg_difficulty_score",
        "n_easy", "n_medium", "n_hard",
        "cautious_response_accuracy",
    ]
    rows = []
    for r in agg_rows:
        if r.get("exp_id") not in DIFFICULTY_EXPS:
            continue
        rows.append({c: r.get(c) for c in cols})
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    df = df.sort_values(["top_k", "exp_id"]).reset_index(drop=True)
    return df


def _build_table6(agg_rows: list[dict], ragas_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Table 6: Overall Ranking by composite score.

    Composite score (equal weights, higher = better):
        0.25 × avg_faithfulness          (RAGAS)
        0.20 × (1 - avg_hallucination_rate)
        0.20 × avg_answer_relevance
        0.20 × avg_recall_at_k
        0.15 × avg_semantic_similarity

    No-RAG experiment: hallucination term dominates, faithfulness = 0.
    Attribution/difficulty experiments: same formula — novelty metrics
    are tracked separately in Table 4/5, not in the composite rank.
    """
    # Merge RAGAS faithfulness into agg_rows
    ragas_faith: dict[tuple, float] = {}
    for df in ragas_dfs:
        for _, row in df.iterrows():
            key = (str(row.get("exp_id", "")), int(row.get("k", 0)))
            if "faithfulness" in df.columns:
                vals = df.loc[df["exp_id"] == row["exp_id"], "faithfulness"].dropna()
                if len(vals) > 0:
                    ragas_faith[key] = float(vals.mean())

    rows = []
    for r in agg_rows:
        exp_id = r.get("exp_id", "")
        k      = r.get("top_k", 0)
        key    = (exp_id, int(k))

        faith    = ragas_faith.get(key) or r.get("avg_faithfulness") or 0.0
        halluc   = r.get("avg_hallucination_rate") or 1.0
        ans_rel  = r.get("avg_answer_relevance")   or 0.0
        recall   = r.get("avg_recall_at_k")        or 0.0
        sem_sim  = r.get("avg_semantic_similarity") or 0.0

        composite = (
            0.25 * faith +
            0.20 * (1.0 - halluc) +
            0.20 * ans_rel +
            0.20 * recall +
            0.15 * sem_sim
        )

        rows.append({
            "exp_id":            exp_id,
            "top_k":             k,
            "pipeline":          r.get("pipeline", ""),
            "run_mode":          r.get("run_mode", "baseline"),
            "faithfulness":      round(faith, 4),
            "1_minus_halluc":    round(1.0 - halluc, 4),
            "answer_relevance":  round(ans_rel, 4),
            "recall_at_k":       round(recall, 4),
            "semantic_sim":      round(sem_sim, 4),
            "composite_score":   round(composite, 4),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------

def _save_table(df: pd.DataFrame, path: Path, name: str) -> None:
    df.to_csv(path, index=False)
    logger.info("%s saved: %s", name, path)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_exp_10(
    outputs_dir: str | Path = "outputs/experiments",
    k_values:    list[int] | None = None,
    save:        bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Load all experiment results and produce comparison tables.

    Parameters
    ----------
    outputs_dir : root experiments directory
    k_values    : K values to include. Defaults to [3, 5, 10].
    save        : if True, save all tables as CSV in outputs_dir/EXP_10/

    Returns
    -------
    dict with keys:
        table1_overall, table2_retrieval, table3_ragas,
        table4_attribution, table5_difficulty, table6_ranking,
        ragas_merged
    """
    if k_values is None:
        k_values = [3, 5, 10]

    outputs_dir = Path(outputs_dir)
    exp10_dir   = outputs_dir / "EXP_10_FINAL_COMPARISON"
    exp10_dir.mkdir(parents=True, exist_ok=True)

    agg_rows:  list[dict]         = []
    ragas_dfs: list[pd.DataFrame] = []
    qr_dfs:    list[pd.DataFrame] = []

    # ── Load all experiments at all K values ─────────────────────────────────
    for exp_id in ALL_EXPERIMENTS:
        k_list = [0] if exp_id == "EXP_01_NO_RAG_LLM" else k_values
        for k in k_list:
            agg = _load_agg_metrics(outputs_dir, exp_id, k)
            if agg:
                agg["top_k"] = k   # ensure top_k is always set
                agg_rows.append(agg)
                logger.info("Loaded agg_metrics: %s k=%d", exp_id, k)
            else:
                logger.warning("Missing agg_metrics: %s k=%d", exp_id, k)

            ragas = _load_ragas_scores(outputs_dir, exp_id, k)
            if ragas is not None:
                ragas_dfs.append(ragas)

            qr = _load_query_results(outputs_dir, exp_id, k)
            if qr is not None:
                qr_dfs.append(qr)

    if not agg_rows:
        logger.error("No agg_metrics loaded. Run experiments first.")
        return {}

    # ── Build tables ──────────────────────────────────────────────────────────
    logger.info("Building comparison tables from %d experiment-K rows...",
                len(agg_rows))

    t1 = _build_table1(agg_rows)
    t2 = _build_table2(agg_rows)
    t3 = _build_table3(ragas_dfs)
    t4 = _build_table4(agg_rows)
    t5 = _build_table5(agg_rows)
    t6 = _build_table6(agg_rows, ragas_dfs)

    ragas_merged = (
        pd.concat(ragas_dfs, ignore_index=True) if ragas_dfs
        else pd.DataFrame()
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    if save:
        _save_table(t1, exp10_dir / "table1_overall.csv",       "Table 1")
        _save_table(t2, exp10_dir / "table2_retrieval.csv",     "Table 2")
        _save_table(t3, exp10_dir / "table3_ragas.csv",         "Table 3")
        _save_table(t4, exp10_dir / "table4_attribution.csv",   "Table 4")
        _save_table(t5, exp10_dir / "table5_difficulty.csv",    "Table 5")
        _save_table(t6, exp10_dir / "table6_ranking.csv",       "Table 6")
        if not ragas_merged.empty:
            _save_table(ragas_merged,
                        exp10_dir / "ragas_scores_merged.csv",  "RAGAS merged")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  EXP_10 FINAL COMPARISON — COMPLETE")
    print("="*70)
    print(f"\n  Experiments loaded : {len(set(r['exp_id'] for r in agg_rows))}")
    print(f"  Total rows (exp×K) : {len(agg_rows)}")
    print(f"  RAGAS rows loaded  : {len(ragas_merged)}")
    print(f"\n  Table 1  — Overall RAG Performance     : {len(t1)} rows")
    print(f"  Table 2  — Retrieval Quality            : {len(t2)} rows")
    print(f"  Table 3  — RAGAS Scores                 : {len(t3)} rows")
    print(f"  Table 4  — Novelty 1 Attribution        : {len(t4)} rows")
    print(f"  Table 5  — Novelty 2 Difficulty         : {len(t5)} rows")
    print(f"  Table 6  — Overall Ranking              : {len(t6)} rows")
    print()
    if not t6.empty:
        print("  TOP 5 ARCHITECTURES BY COMPOSITE SCORE:")
        print(t6[["rank", "exp_id", "top_k", "composite_score"]]
              .head(5).to_string(index=False))
    print("="*70 + "\n")

    return {
        "table1_overall":    t1,
        "table2_retrieval":  t2,
        "table3_ragas":      t3,
        "table4_attribution":t4,
        "table5_difficulty": t5,
        "table6_ranking":    t6,
        "ragas_merged":      ragas_merged,
    }
