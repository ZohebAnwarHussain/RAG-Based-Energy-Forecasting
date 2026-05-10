"""
experiments/results_printer.py
================================
Pretty-print experiment results in tabular format after RAGAS is run.

USAGE (in notebook)
-------------------
    from experiments.results_printer import print_experiment_table

    # Single experiment at one K value
    print_experiment_table("EXP_02_DENSE_RAG", k=5)

    # All K values for one experiment
    print_experiment_table("EXP_03_HYBRID_RAG", k="all")

    # All experiments, all K values — master comparison table
    print_all_experiments()

    # Specific list of experiments
    print_all_experiments(
        exp_ids=["EXP_02_DENSE_RAG", "EXP_03_HYBRID_RAG"],
        k_values=[3, 5, 10]
    )

OUTPUT
------
Produces a rich IPython-rendered HTML table inside Jupyter, or a plain
pandas DataFrame with all metrics if called outside a notebook.

Columns in the table (matching your thesis Table 1 + Table 2 layout):
  EXP / Pipeline / K
  Correct / Useful Insights
  Answer Relevance (custom)
  RAGAS Answer Relevancy
  Semantic Similarity
  Faithfulness Score         ← from ragas_scores.csv
  Context Precision          ← from ragas_scores.csv
  Context Recall             ← from ragas_scores.csv
  Hallucination Rate (%)
  Avg Latency / Query (s)
  Relevant Summaries Available
  Relevant Summaries Retrieved
  Recall @ K
  Precision @ K
  MRR
  nDCG @ K
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Experiment registry — all known experiments and their expected K values
# ---------------------------------------------------------------------------

ALL_EXP_IDS = [
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

# EXP_01 uses K=0 (no retrieval); all others use K=3,5,10
K_MAP = {
    "EXP_01_NO_RAG_LLM":                     [0],
    "EXP_05_DENSE_RAG_ATTRIBUTION":          [5],
    "EXP_06_HYBRID_RAG_ATTRIBUTION":         [5],
    "EXP_07_HIERARCHICAL_RAG_ATTRIBUTION":   [5],
    "EXP_08_QUERY_DIFFICULTY_DENSE_HYBRID":  [5],
    "EXP_09_QUERY_DIFFICULTY_HIERARCHICAL":  [5],
}
DEFAULT_K_VALUES = [3, 5, 10]

# Short display names for the table (keeps table compact)
EXP_SHORT_NAMES = {
    "EXP_01_NO_RAG_LLM":                     "EXP_01 No-RAG",
    "EXP_02_DENSE_RAG":                      "EXP_02 Dense",
    "EXP_03_HYBRID_RAG":                     "EXP_03 Hybrid",
    "EXP_04_HIERARCHICAL_RAG":               "EXP_04 Hierarchical",
    "EXP_05_DENSE_RAG_ATTRIBUTION":          "EXP_05 Dense+Attr",
    "EXP_06_HYBRID_RAG_ATTRIBUTION":         "EXP_06 Hybrid+Attr",
    "EXP_07_HIERARCHICAL_RAG_ATTRIBUTION":   "EXP_07 Hier+Attr",
    "EXP_08_QUERY_DIFFICULTY_DENSE_HYBRID":  "EXP_08 Difficulty",
    "EXP_09_QUERY_DIFFICULTY_HIERARCHICAL":  "EXP_09 Diff+Hier",
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_agg(outputs_dir: Path, exp_id: str, k: int) -> dict | None:
    path = outputs_dir / exp_id / f"k{k}" / "agg_metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_ragas(outputs_dir: Path, exp_id: str, k: int) -> pd.DataFrame | None:
    path = outputs_dir / exp_id / f"k{k}" / "ragas_scores.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _safe(val, fmt=".4f", pct=False, missing="—") -> str:
    """Format a metric value. Returns '—' if None/NaN."""
    if val is None:
        return missing
    try:
        v = float(val)
        if np.isnan(v):
            return missing
        if pct:
            return f"{v * 100:.1f}%"
        return format(v, fmt)
    except (TypeError, ValueError):
        return str(val) if val else missing



# ---------------------------------------------------------------------------
# Row builder — one row per experiment × K
# ---------------------------------------------------------------------------

def _build_row(
    outputs_dir: Path,
    exp_id: str,
    k: int,
    obs_overrides: dict | None = None,
) -> dict | None:
    """
    Load agg_metrics.json + ragas_scores.csv for one experiment × K
    and return a flat dict of all display metrics.
    Returns None if no data found.
    """
    agg = _load_agg(outputs_dir, exp_id, k)
    if agg is None:
        return None

    ragas_df = _load_ragas(outputs_dir, exp_id, k)

    # Pull RAGAS metrics from ragas_scores.csv (computed over non-null rows)
    def _ragas_mean(col: str):
        if ragas_df is None or col not in ragas_df.columns:
            return None
        vals = ragas_df[col].dropna()
        return float(vals.mean()) if len(vals) > 0 else None

    def _ragas_valid(col: str):
        if ragas_df is None or col not in ragas_df.columns:
            return 0
        return int(ragas_df[col].notna().sum())

    faithfulness       = _ragas_mean("faithfulness")
    answer_relevancy   = _ragas_mean("answer_relevancy")
    context_precision  = _ragas_mean("context_precision")
    context_recall     = _ragas_mean("context_recall")
    n_ragas_valid      = _ragas_valid("faithfulness")
    n_total            = agg.get("n_queries", 50)

    row = {
        # Identity
        "exp_id":       exp_id,
        "exp_short":    EXP_SHORT_NAMES.get(exp_id, exp_id),
        "pipeline":     agg.get("pipeline", ""),
        "run_mode":     agg.get("run_mode", "baseline"),
        "k":            k,

        # Generation quality
        "n_valid":             agg.get("n_valid", 0),
        "n_queries":           n_total,
        "pct_useful":          agg.get("pct_useful"),
        "avg_answer_relevance": agg.get("avg_answer_relevance"),
        "ragas_answer_relevancy": answer_relevancy,
        "avg_semantic_similarity": agg.get("avg_semantic_similarity"),

        # RAGAS grounding
        "faithfulness":       faithfulness,
        "context_precision":  context_precision,
        "context_recall":     context_recall,
        "n_ragas_valid":      n_ragas_valid,

        # Hallucination
        "hallucination_rate": agg.get("avg_hallucination_rate"),

        # Latency
        "avg_latency_sec":    agg.get("avg_latency_sec"),

        # Retrieval
        "avg_relevant_available": agg.get("avg_relevant_available"),
        "avg_relevant_retrieved": agg.get("avg_relevant_retrieved"),
        "recall_at_k":            agg.get("avg_recall_at_k"),
        "precision_at_k":         agg.get("avg_precision_at_k"),
        "mrr":                    agg.get("avg_mrr"),
        "ndcg_at_k":              agg.get("avg_ndcg_at_k"),

        # Novelty 1 -- Attribution (EXP_05/06/07)
        "attribution_coverage":   agg.get("avg_attribution_coverage"),
        "citation_accuracy":      agg.get("avg_citation_accuracy"),
        "unsupported_claim_rate": agg.get("avg_unsupported_claim_rate"),

        # Novelty 2 -- Difficulty (EXP_08/09)
        "coverage_score":              agg.get("avg_coverage_score"),
        "consistency_score":           agg.get("avg_consistency_score"),
        "difficulty_score":            agg.get("avg_difficulty_score"),
        "n_easy":                      agg.get("n_easy"),
        "n_medium":                    agg.get("n_medium"),
        "n_hard":                      agg.get("n_hard"),
        "cautious_response_accuracy":  agg.get("cautious_response_accuracy"),

        # Hierarchical extras (EXP_04/07/09)
        "avg_n_children":     agg.get("avg_n_children"),
        "avg_n_parents_added": agg.get("avg_n_parents_added"),
    }

    # Observation
    obs_key = f"{exp_id}_k{k}"
    if obs_overrides and obs_key in obs_overrides:
        row["observation"] = obs_overrides[obs_key]

    return row


# ---------------------------------------------------------------------------
# DataFrame builder
# ---------------------------------------------------------------------------

def build_results_dataframe(
    outputs_dir: str | Path = "outputs/experiments",
    exp_ids: list[str] | None = None,
    k_values: list[int] | None = None,
    obs_overrides: dict | None = None,
) -> pd.DataFrame:
    """
    Build a flat DataFrame of all experiment × K results.

    Parameters
    ----------
    outputs_dir   : root experiments output directory
    exp_ids       : list of experiment IDs to include. Defaults to all.
    k_values      : K values to include. Defaults to [0,3,5,10].
    obs_overrides : dict mapping "EXP_ID_kN" → custom observation string.
                    e.g. {"EXP_03_HYBRID_RAG_k5": "My custom note"}

    Returns
    -------
    pd.DataFrame with one row per experiment × K.
    Missing experiments (not yet run) are silently skipped.
    """
    outputs_dir = Path(outputs_dir)
    if exp_ids is None:
        exp_ids = ALL_EXP_IDS
    rows = []
    for exp_id in exp_ids:
        ks = k_values if k_values is not None else K_MAP.get(exp_id, DEFAULT_K_VALUES)
        for k in ks:
            row = _build_row(outputs_dir, exp_id, k, obs_overrides)
            if row is not None:
                rows.append(row)

    if not rows:
        print("No experiment results found. Check outputs_dir and run experiments first.")
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# HTML renderer (Jupyter)
# ---------------------------------------------------------------------------

_TABLE_CSS = """
<style>
.exp-table {
    border-collapse: collapse;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 12px;
    width: 100%;
    margin: 12px 0;
}
.exp-table thead tr {
    background: #1a1a2e;
    color: #ffffff;
}
.exp-table thead th {
    padding: 8px 10px;
    text-align: center;
    font-weight: 600;
    border: 1px solid #333;
    white-space: nowrap;
}
.exp-table thead th.left { text-align: left; }
.exp-table tbody tr:nth-child(even) { background: #f4f6fb; }
.exp-table tbody tr:nth-child(odd)  { background: #ffffff; }
.exp-table tbody tr:hover           { background: #e8edf8; }
.exp-table tbody td {
    padding: 6px 10px;
    border: 1px solid #d0d7e8;
    text-align: center;
    vertical-align: middle;
}
.exp-table tbody td.left { text-align: left; }
.exp-table tbody td.obs  {
    text-align: left;
    font-size: 11px;
    color: #444;
    max-width: 280px;
    white-space: normal;
    line-height: 1.4;
}
.tag-pending {
    background: #fff3cd; color: #856404;
    padding: 1px 6px; border-radius: 4px; font-size: 10px;
}
.tag-good    {
    background: #d4edda; color: #155724;
    padding: 1px 6px; border-radius: 4px; font-size: 10px;
    font-weight: 600;
}
.tag-warn    {
    background: #ffeeba; color: #856404;
    padding: 1px 6px; border-radius: 4px; font-size: 10px;
}
.tag-bad     {
    background: #f8d7da; color: #721c24;
    padding: 1px 6px; border-radius: 4px; font-size: 10px;
}
.sep-row td {
    background: #e8edf8 !important;
    font-weight: 600;
    font-size: 11px;
    color: #1a1a2e;
    text-align: left;
    padding: 4px 10px;
}
</style>
"""


def _colour_cell(val, col: str) -> str:
    """Wrap a value string in a coloured badge based on metric + threshold."""
    if val == "—":
        return f'<span class="tag-pending">—</span>'

    try:
        v = float(val.replace("%", "")) / (100 if "%" in val else 1)
    except ValueError:
        return val

    if col == "faithfulness":
        cls = "tag-good" if v >= 0.30 else ("tag-warn" if v >= 0.10 else "tag-bad")
    elif col == "hallucination_rate":
        cls = "tag-good" if v <= 0.15 else ("tag-warn" if v <= 0.25 else "tag-bad")
    elif col in ("recall_at_k", "precision_at_k", "mrr", "ndcg_at_k"):
        cls = "tag-good" if v >= 0.15 else ("tag-warn" if v >= 0.05 else "tag-bad")
    elif col == "context_precision":
        cls = "tag-good" if v >= 0.10 else ("tag-warn" if v > 0.00 else "tag-bad")
    elif col == "context_recall":
        cls = "tag-good" if v >= 0.20 else ("tag-warn" if v >= 0.05 else "tag-bad")
    elif col == "pct_useful":
        cls = "tag-good" if v >= 0.95 else ("tag-warn" if v >= 0.80 else "tag-bad")
    elif col == "attribution_coverage":
        cls = "tag-good" if v >= 0.85 else ("tag-warn" if v >= 0.60 else "tag-bad")
    elif col == "citation_accuracy":
        cls = "tag-good" if v >= 0.90 else ("tag-warn" if v >= 0.70 else "tag-bad")
    elif col == "unsupported_claim_rate":
        cls = "tag-good" if v <= 0.15 else ("tag-warn" if v <= 0.30 else "tag-bad")
    elif col == "cautious_response_accuracy":
        cls = "tag-good" if v >= 0.80 else ("tag-warn" if v >= 0.50 else "tag-bad")
    elif col in ("coverage_score", "consistency_score"):
        cls = "tag-good" if v >= 0.70 else ("tag-warn" if v >= 0.40 else "tag-bad")
    else:
        return val

    return f'<span class="{cls}">{val}</span>'


def _render_html_table(df: pd.DataFrame, title: str = "") -> str:
    """Render the results DataFrame as a styled HTML table."""
    if df.empty:
        return "<p>No data available.</p>"

    is_attribution = any(
        "ATTRIBUTION" in str(row.get("exp_id", ""))
        for _, row in df.iterrows()
    )
    is_difficulty = any(
        "DIFFICULTY" in str(row.get("exp_id", ""))
        for _, row in df.iterrows()
    )
    is_hierarchical = any(
        "HIERARCHICAL" in str(row.get("exp_id", ""))
        for _, row in df.iterrows()
    )

    headers = [
        ("EXP / Pipeline",          "left",   None),
        ("K",                        "center", None),
        ("Correct / Useful",         "center", "pct_useful"),
        ("Answer Relevance",         "center", None),
        ("RAGAS Ans. Relevancy",     "center", None),
        ("Semantic Similarity",      "center", None),
        ("Faithfulness",             "center", "faithfulness"),
        ("Context Precision",        "center", "context_precision"),
        ("Context Recall",           "center", "context_recall"),
        ("Hallucination %",          "center", "hallucination_rate"),
        ("Avg Latency (s)",          "center", None),
        ("Rel. Available",           "center", None),
        ("Rel. Retrieved",           "center", None),
        ("Recall@K",                 "center", "recall_at_k"),
        ("Precision@K",              "center", "precision_at_k"),
        ("MRR",                      "center", None),
        ("nDCG@K",                   "center", None),
    ]

    if is_hierarchical:
        headers += [
            ("Children",  "center", None),
            ("Parents",   "center", None),
        ]

    if is_attribution:
        headers += [
            ("Attr. Coverage",     "center", "attribution_coverage"),
            ("Citation Acc.",      "center", "citation_accuracy"),
            ("Unsupported %",      "center", "unsupported_claim_rate"),
        ]

    if is_difficulty:
        headers += [
            ("Coverage Sc.",   "center", "coverage_score"),
            ("Consistency Sc.","center", "consistency_score"),
            ("Easy/Med/Hard",  "center", None),
            ("Cautious Acc.",  "center", "cautious_response_accuracy"),
        ]


    html = [_TABLE_CSS]
    if title:
        html.append(f"<h3 style='font-family:Arial;color:#1a1a2e;margin:8px 0'>{title}</h3>")

    html.append('<table class="exp-table"><thead><tr>')
    for label, align, _ in headers:
        cls = "left" if align == "left" else ""
        html.append(f'<th class="{cls}">{label}</th>')
    html.append("</tr></thead><tbody>")

    last_exp = None
    for _, row in df.iterrows():
        # Separator row when experiment changes
        if row["exp_id"] != last_exp:
            html.append(f'<tr class="sep-row"><td colspan="{len(headers)}">'
                        f'{EXP_SHORT_NAMES.get(row["exp_id"], row["exp_id"])}'
                        f' &mdash; {row["pipeline"]}'
                        f'</td></tr>')
            last_exp = row["exp_id"]

        # RAGAS coverage note
        n_valid = row.get("n_ragas_valid", 0)
        n_total = row.get("n_queries", 50)
        ragas_note = f" <small style='color:#888'>({n_valid}/{n_total})</small>" if n_valid else ""

        # Useful insights: n_valid / n_total + pct
        n_useful = int(round((row.get("pct_useful") or 0) * n_total))
        useful_str = f"{n_useful}/{n_total} ({_safe(row.get('pct_useful'), pct=True)})"

        cells = [
            (f"{row['exp_short']} <small style='color:#888;font-size:10px'>"
             f"[{row.get('run_mode','baseline')}]</small>",         "left",  None),
            (str(row["k"]) if row["k"] > 0 else "—",               "center", None),
            (useful_str,                                             "center", "pct_useful"),
            (_safe(row.get("avg_answer_relevance")),                 "center", None),
            (_safe(row.get("ragas_answer_relevancy")) + ragas_note, "center", None),
            (_safe(row.get("avg_semantic_similarity")),              "center", None),
            (_safe(row.get("faithfulness")) + ragas_note,           "center", "faithfulness"),
            (_safe(row.get("context_precision")) + ragas_note,      "center", "context_precision"),
            (_safe(row.get("context_recall")) + ragas_note,         "center", "context_recall"),
            (_safe(row.get("hallucination_rate"), pct=True),        "center", "hallucination_rate"),
            (_safe(row.get("avg_latency_sec"), fmt=".3f") + "s",    "center", None),
            (_safe(row.get("avg_relevant_available"), fmt=".2f"),   "center", None),
            (_safe(row.get("avg_relevant_retrieved"), fmt=".2f"),   "center", None),
            (_safe(row.get("recall_at_k")),                         "center", "recall_at_k"),
            (_safe(row.get("precision_at_k")),                      "center", "precision_at_k"),
            (_safe(row.get("mrr")),                                  "center", None),
            (_safe(row.get("ndcg_at_k")),                           "center", None),
        ]

        if is_hierarchical:
            cells += [
                (_safe(row.get("avg_n_children"),      fmt=".1f"), "center", None),
                (_safe(row.get("avg_n_parents_added"), fmt=".1f"), "center", None),
            ]

        if is_attribution:
            cells += [
                (_safe(row.get("attribution_coverage"),   pct=True), "center", "attribution_coverage"),
                (_safe(row.get("citation_accuracy"),       pct=True), "center", "citation_accuracy"),
                (_safe(row.get("unsupported_claim_rate"),  pct=True), "center", "unsupported_claim_rate"),
            ]

        if is_difficulty:
            easy   = row.get("n_easy")   or 0
            medium = row.get("n_medium") or 0
            hard   = row.get("n_hard")   or 0
            diff_str = f"{easy}/{medium}/{hard}" if (easy or medium or hard) else "—"
            cells += [
                (_safe(row.get("coverage_score")),              "center", "coverage_score"),
                (_safe(row.get("consistency_score")),           "center", "consistency_score"),
                (diff_str,                                      "center", None),
                (_safe(row.get("cautious_response_accuracy")), "center", "cautious_response_accuracy"),
            ]


        html.append("<tr>")
        for val_str, align, col_key in cells:
            css = "obs" if align == "obs" else ("left" if align == "left" else "")
            display = _colour_cell(val_str, col_key) if col_key else val_str
            html.append(f'<td class="{css}">{display}</td>')
        html.append("</tr>")

    html.append("</tbody></table>")
    return "\n".join(html)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def print_experiment_table(
    exp_id: str,
    k: int | str = "all",
    outputs_dir: str | Path = "outputs/experiments",
    obs_overrides: dict | None = None,
    return_df: bool = False,
) -> pd.DataFrame | None:
    """
    Print results table for one experiment.

    Parameters
    ----------
    exp_id        : e.g. "EXP_03_HYBRID_RAG"
    k             : specific K value, or "all" for all K values
    outputs_dir   : root experiments output directory
    obs_overrides : dict of {"EXP_ID_kN": "custom observation string"}
    return_df     : if True, return the raw DataFrame in addition to printing

    Example
    -------
        print_experiment_table("EXP_03_HYBRID_RAG", k=5)
        print_experiment_table("EXP_02_DENSE_RAG", k="all")
    """
    k_values = K_MAP.get(exp_id, DEFAULT_K_VALUES) if k == "all" else [k]
    df = build_results_dataframe(
        outputs_dir=outputs_dir,
        exp_ids=[exp_id],
        k_values=k_values,
        obs_overrides=obs_overrides,
    )
    if df.empty:
        print(f"No results found for {exp_id}. Check outputs_dir.")
        return df if return_df else None

    _display_or_print(df, title=f"Results: {EXP_SHORT_NAMES.get(exp_id, exp_id)}")
    return df if return_df else None


def print_all_experiments(
    outputs_dir: str | Path = "outputs/experiments",
    exp_ids: list[str] | None = None,
    k_values: list[int] | None = None,
    obs_overrides: dict | None = None,
    split_by_k: bool = False,
    return_df: bool = False,
) -> pd.DataFrame | None:
    """
    Print master comparison table across all experiments.

    Parameters
    ----------
    outputs_dir   : root experiments output directory
    exp_ids       : subset of experiments to include. Defaults to all.
    k_values      : K values to include. Defaults to [3, 5, 10].
    obs_overrides : dict of {"EXP_ID_kN": "custom observation string"}
    split_by_k    : if True, print one table per K value instead of combined
    return_df     : if True, return the raw DataFrame

    Example
    -------
        # After running all experiments:
        print_all_experiments()

        # Just K=5 comparison:
        print_all_experiments(k_values=[5])

        # Split into separate tables per K:
        print_all_experiments(split_by_k=True)
    """
    df = build_results_dataframe(
        outputs_dir=outputs_dir,
        exp_ids=exp_ids,
        k_values=k_values,
        obs_overrides=obs_overrides,
    )
    if df.empty:
        print("No results found. Check outputs_dir and ensure experiments have been run.")
        return df if return_df else None

    if split_by_k:
        for k_val in sorted(df["k"].unique()):
            sub = df[df["k"] == k_val].reset_index(drop=True)
            _display_or_print(sub, title=f"All Experiments — K = {k_val}")
    else:
        _display_or_print(df, title="All Experiments — Full Comparison")

    return df if return_df else None


def print_summary_stats(
    outputs_dir: str | Path = "outputs/experiments",
    exp_ids: list[str] | None = None,
    k_values: list[int] | None = None,
) -> None:
    """
    Print a quick coverage summary showing which experiments have been
    run and which RAGAS scores are complete vs pending.

    Useful for checking status before running print_all_experiments().

    Example
    -------
        print_summary_stats()
    """
    df = build_results_dataframe(
        outputs_dir=outputs_dir,
        exp_ids=exp_ids,
        k_values=k_values,
    )
    if df.empty:
        print("No experiments found.")
        return

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT COVERAGE SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Experiment':<40} {'K':>4}  {'Gen':>5}  {'RAGAS':>8}")
    print(f"  {'-'*55}")
    for _, row in df.iterrows():
        gen_status   = "OK " if row.get("n_valid", 0) >= int(row.get("n_queries", 200) * 0.9) else "WARN"
        n_valid      = row.get("n_ragas_valid", 0)
        n_total      = row.get("n_queries", 50)
        ragas_status = "OK " if n_valid >= n_total * 0.9 else (
            f"PART {n_valid}/{n_total}" if n_valid > 0 else "PENDING"
        )
        k_str = str(row["k"]) if row["k"] > 0 else " —"
        print(f"  {row['exp_short']:<40} {k_str:>4}  {gen_status}     {ragas_status}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Display helper — Jupyter vs plain terminal
# ---------------------------------------------------------------------------

def _display_or_print(df: pd.DataFrame, title: str = "") -> None:
    """Render HTML in Jupyter; fall back to pandas print in terminal."""
    try:
        from IPython.display import display, HTML
        html = _render_html_table(df, title=title)
        display(HTML(html))
    except ImportError:
        # Plain terminal fallback
        display_cols = [
            "exp_short", "k",
            "pct_useful", "avg_answer_relevance", "avg_semantic_similarity",
            "faithfulness", "context_precision", "context_recall",
            "hallucination_rate", "avg_latency_sec",
            "avg_relevant_available", "avg_relevant_retrieved",
            "recall_at_k", "precision_at_k", "mrr", "ndcg_at_k",
            "attribution_coverage", "citation_accuracy", "unsupported_claim_rate",
            "coverage_score", "consistency_score", "cautious_response_accuracy",
            "avg_n_children", "avg_n_parents_added",
        ]
        available = [c for c in display_cols if c in df.columns]
        print(f"\n{title}")
        print(df[available].to_string(index=False))
