"""
src/analysis/plot_helpers.py
==============================
All plotting functions for Notebook 07.

Design choices
--------------
* Every function is self-contained: takes a DataFrame, returns a
  matplotlib Figure.  The notebook just calls plt.show() / savefig().
* Consistent colour palette so all charts feel like one report.
* Handles NaN gracefully (RAGAS partial results).
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared palette & style
# ---------------------------------------------------------------------------

PIPELINE_COLORS = {
    "dense":        "#2563EB",   # blue
    "hybrid":       "#16A34A",   # green
    "hierarchical": "#DC2626",   # red
}
DEFAULT_COLOR = "#6B7280"

FIGSIZE_WIDE    = (12, 5)
FIGSIZE_SQUARE  = (7, 6)
FIGSIZE_TALL    = (8, 9)

FONT_TITLE  = {"fontsize": 13, "fontweight": "bold", "pad": 10}
FONT_LABEL  = {"fontsize": 11}
FONT_TICK   = {"labelsize": 10}

plt.rcParams.update(
    {
        "figure.facecolor":  "white",
        "axes.facecolor":    "#F9FAFB",
        "axes.edgecolor":    "#E5E7EB",
        "axes.grid":         True,
        "grid.color":        "#E5E7EB",
        "grid.linestyle":    "--",
        "grid.linewidth":    0.6,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.family":       "DejaVu Sans",
    }
)


def _pipeline_colors(pipelines: list[str]) -> list[str]:
    return [PIPELINE_COLORS.get(p, DEFAULT_COLOR) for p in pipelines]


# ---------------------------------------------------------------------------
# 1. Retrieval metrics grouped bar chart
# ---------------------------------------------------------------------------

def plot_retrieval_metrics(retrieval_df: pd.DataFrame,
                            metrics: Optional[list[str]] = None,
                            title: str = "Retrieval Metrics by Pipeline") -> plt.Figure:
    """
    Grouped bar chart: one group per metric, one bar per pipeline.
    """
    if metrics is None:
        metrics = [c for c in ["recall_at_k", "precision_at_k", "mrr", "ndcg"]
                   if c in retrieval_df.columns]

    pipelines = retrieval_df["pipeline"].tolist()
    n_pipelines = len(pipelines)
    n_metrics   = len(metrics)
    x = np.arange(n_metrics)
    width = 0.25

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    for i, (_, row) in enumerate(retrieval_df.iterrows()):
        offset = (i - n_pipelines / 2 + 0.5) * width
        vals   = [row.get(m, 0) for m in metrics]
        color  = PIPELINE_COLORS.get(str(row["pipeline"]), DEFAULT_COLOR)
        bars   = ax.bar(x + offset, vals, width, label=row["pipeline"],
                        color=color, alpha=0.88, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_title(title, **FONT_TITLE)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], **FONT_TICK)
    ax.set_ylabel("Score", **FONT_LABEL)
    ax.set_ylim(0, max(retrieval_df[metrics].max().max() * 1.25, 0.25))
    ax.legend(title="Pipeline", framealpha=0.9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Radar / spider chart for retrieval (holistic view)
# ---------------------------------------------------------------------------

def plot_retrieval_radar(retrieval_df: pd.DataFrame,
                          metrics: Optional[list[str]] = None,
                          title: str = "Retrieval Profile per Pipeline") -> plt.Figure:
    """
    Radar chart comparing pipelines across retrieval metrics.
    """
    if metrics is None:
        metrics = [c for c in ["recall_at_k", "precision_at_k", "mrr", "ndcg"]
                   if c in retrieval_df.columns]

    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]                   # close the polygon

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE,
                            subplot_kw=dict(polar=True))

    for _, row in retrieval_df.iterrows():
        vals = [row.get(m, 0) for m in metrics]
        vals += vals[:1]
        color = PIPELINE_COLORS.get(str(row["pipeline"]), DEFAULT_COLOR)
        ax.plot(angles, vals, "o-", linewidth=2, color=color,
                label=row["pipeline"])
        ax.fill(angles, vals, alpha=0.12, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]),
                      [m.replace("_", "\n") for m in metrics],
                      fontsize=10)
    ax.set_title(title, **FONT_TITLE, y=1.12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), framealpha=0.9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Hallucination stacked bar
# ---------------------------------------------------------------------------

def plot_hallucination_bars(halluc_df: pd.DataFrame,
                             title: str = "Hallucination Check Pass Rates") -> plt.Figure:
    """
    Horizontal grouped bar chart showing include / exclude / overall pass %.
    """
    check_cols = [c for c in ["include_pass", "exclude_pass", "overall_pass"]
                  if c in halluc_df.columns]

    pipelines = halluc_df["pipeline"].tolist()
    n = len(pipelines)
    y = np.arange(n)
    height = 0.25

    fig, ax = plt.subplots(figsize=(10, max(4, n * 1.8)))

    label_map = {
        "include_pass":  "Include-pass (grounding)",
        "exclude_pass":  "Exclude-pass (hallucination)",
        "overall_pass":  "Overall pass",
    }
    hatch_map = {"include_pass": "", "exclude_pass": "//", "overall_pass": "xx"}
    alpha_map  = {"include_pass": 0.85, "exclude_pass": 0.75, "overall_pass": 1.0}

    for j, col in enumerate(check_cols):
        offset = (j - len(check_cols) / 2 + 0.5) * height
        vals   = halluc_df[col].tolist()
        colors = _pipeline_colors(pipelines)
        bars   = ax.barh(y + offset, vals, height,
                         label=label_map.get(col, col),
                         color=colors,
                         alpha=alpha_map.get(col, 0.85),
                         hatch=hatch_map.get(col, ""),
                         zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(v + 0.008, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1%}", va="center", fontsize=9)

    ax.set_title(title, **FONT_TITLE)
    ax.set_yticks(y)
    ax.set_yticklabels(pipelines, **FONT_TICK)
    ax.set_xlabel("Pass Rate", **FONT_LABEL)
    ax.set_xlim(0, 1.15)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. RAGAS heatmap (handles NaN)
# ---------------------------------------------------------------------------

def plot_ragas_heatmap(ragas_df: pd.DataFrame,
                        title: str = "RAGAS Metrics (partial — rate-limit constrained)") -> plt.Figure:
    """
    Heatmap of RAGAS scores per pipeline × metric.
    NaN cells are shown in light grey with 'N/A' text.
    """
    metric_cols = [c for c in
                   ["faithfulness", "answer_relevancy",
                    "context_precision", "context_recall"]
                   if c in ragas_df.columns]

    matrix = ragas_df.set_index("pipeline")[metric_cols]

    fig, ax = plt.subplots(figsize=(max(7, len(metric_cols) * 2), max(4, len(matrix) * 1.5)))

    # Background grid for NaN
    nan_mask = matrix.isna()
    data_for_plot = matrix.copy().fillna(0)

    cmap = plt.cm.get_cmap("Blues")
    im = ax.imshow(data_for_plot.values, cmap=cmap, aspect="auto",
                   vmin=0, vmax=1)

    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="Score (0–1)")

    for i in range(len(matrix)):
        for j in range(len(metric_cols)):
            val = matrix.iloc[i, j]
            if pd.isna(val):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=True, color="#E5E7EB", zorder=2))
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=11, color="#9CA3AF", zorder=3)
            else:
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=11,
                        color="white" if val > 0.6 else "black",
                        fontweight="bold", zorder=3)

    ax.set_xticks(range(len(metric_cols)))
    ax.set_xticklabels([c.replace("_", "\n") for c in metric_cols], fontsize=10)
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels(matrix.index.tolist(), fontsize=10)
    ax.set_title(title, **FONT_TITLE)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Answer length distribution (box + strip)
# ---------------------------------------------------------------------------

def plot_answer_length_dist(rag_df: pd.DataFrame,
                             title: str = "Answer Word-Count Distribution by Pipeline") -> plt.Figure:
    """
    Box plots of answer word counts per pipeline.
    Falls back to a simple bar if per-query data is unavailable.
    """
    if rag_df.empty:
        fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
        ax.text(0.5, 0.5, "No RAG answer data available",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title(title, **FONT_TITLE)
        return fig

    if "answer_length" not in rag_df.columns:
        rag_df = rag_df.copy()
        rag_df["answer_length"] = rag_df["answer"].astype(str).apply(
            lambda x: len(x.split())
        )

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)

    if "pipeline" in rag_df.columns:
        pipelines = sorted(rag_df["pipeline"].unique())
        data = [rag_df[rag_df["pipeline"] == p]["answer_length"].dropna().values
                for p in pipelines]
        colors = _pipeline_colors(pipelines)

        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_xticklabels(pipelines, **FONT_TICK)
        ax.set_xlabel("Pipeline", **FONT_LABEL)
    else:
        ax.hist(rag_df["answer_length"].dropna(), bins=20, color="#2563EB", alpha=0.8)
        ax.set_xlabel("Word Count", **FONT_LABEL)

    ax.set_ylabel("Word Count", **FONT_LABEL)
    ax.set_title(title, **FONT_TITLE)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Unified scorecard bar chart
# ---------------------------------------------------------------------------

def plot_scorecard(scorecard_df: pd.DataFrame,
                   title: str = "Overall Pipeline Scorecard") -> plt.Figure:
    """
    Horizontal bar chart showing the overall_score per pipeline,
    with labelled sub-scores.
    """
    if scorecard_df.empty or "overall_score" not in scorecard_df.columns:
        fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
        ax.text(0.5, 0.5, "Scorecard data unavailable",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    pipelines = scorecard_df["pipeline"].tolist()
    scores    = scorecard_df["overall_score"].tolist()
    colors    = _pipeline_colors(pipelines)

    fig, ax = plt.subplots(figsize=(9, max(4, len(pipelines) * 1.8)))
    bars = ax.barh(pipelines, scores, color=colors, alpha=0.88, zorder=3, height=0.5)

    for bar, score, (_, row) in zip(bars, scores, scorecard_df.iterrows()):
        ax.text(score + 0.01, bar.get_y() + bar.get_height() / 2,
                f"  {score:.3f}", va="center", fontsize=11, fontweight="bold")

    ax.set_title(title, **FONT_TITLE)
    ax.set_xlabel("Overall Score (normalised)", **FONT_LABEL)
    ax.set_xlim(0, 1.2)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Query distribution (golden dataset)
# ---------------------------------------------------------------------------

def plot_query_distribution(golden_df: pd.DataFrame,
                             title: str = "Golden Dataset — Query Distribution") -> plt.Figure:
    """
    Stacked bar showing query count by dataset_source × granularity.
    """
    if golden_df.empty:
        fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
        ax.text(0.5, 0.5, "Golden dataset not loaded",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    group_col   = "dataset_source" if "dataset_source" in golden_df.columns else None
    gran_col    = "granularity"     if "granularity" in golden_df.columns else None

    if group_col is None and gran_col is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
        ax.text(0.5, 0.5, "No 'dataset_source' or 'granularity' column found",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    if group_col and gran_col:
        pivot = (golden_df.groupby([group_col, gran_col])
                          .size()
                          .unstack(fill_value=0))
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        pivot.plot(kind="bar", ax=ax, colormap="Set2", alpha=0.9, zorder=3)
        ax.set_xlabel("Dataset Source", **FONT_LABEL)
        ax.set_xticklabels(pivot.index, rotation=0, **FONT_TICK)
        ax.legend(title="Granularity", framealpha=0.9)
    else:
        col = group_col or gran_col
        counts = golden_df[col].value_counts()
        fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
        counts.plot(kind="bar", ax=ax, color="#2563EB", alpha=0.88, zorder=3)
        ax.set_xlabel(col, **FONT_LABEL)
        ax.set_xticklabels(counts.index, rotation=45, ha="right", **FONT_TICK)

    ax.set_ylabel("Number of Queries", **FONT_LABEL)
    ax.set_title(title, **FONT_TITLE)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 8. Save helper
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    """Save figure to *path* (PNG/PDF/SVG)."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
