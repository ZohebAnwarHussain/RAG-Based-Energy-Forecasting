"""
src/experiments/difficulty.py
================================
Novelty 2 — Query Difficulty Prediction

Pipeline
--------
1. Run retrieval to get top-k docs + similarity scores
2. compute_coverage_score()    — fraction of docs above similarity threshold
3. compute_consistency_score() — how similar the retrieved docs are to each other
4. compute_difficulty_score()  — combine into one score
5. label_difficulty()          — Easy / Medium / Hard
6. build_difficulty_prompt_prefix() — text block injected into the LLM prompt
7. evaluate_caution()          — check if Hard answers contain hedging language
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Thresholds (can be overridden)
# ---------------------------------------------------------------------------

DEFAULT_SIMILARITY_THRESHOLD = 0.50   # cosine similarity
EASY_THRESHOLD   = 0.65               # difficulty_score < this → Easy
HARD_THRESHOLD   = 0.40               # difficulty_score < this → Hard (else Medium)

# Hedging phrases that indicate a cautious response
_CAUTION_PHRASES = [
    "limited evidence", "insufficient data", "uncertain", "not enough",
    "caution", "may not", "could not", "unreliable", "low confidence",
    "exercise caution", "interpret carefully", "should be treated",
    "results may vary", "data is sparse", "weak coverage",
]

# ---------------------------------------------------------------------------
# 1. Coverage score
# ---------------------------------------------------------------------------

def compute_coverage_score(
    similarity_scores: list[float],
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> float:
    """
    Fraction of retrieved docs whose similarity score exceeds *threshold*.

    A high coverage score means the KB has strong evidence for this query.

    Parameters
    ----------
    similarity_scores : cosine similarity scores for the top-k retrieved docs
    threshold         : minimum similarity to count as 'relevant'

    Returns
    -------
    float in [0, 1]
    """
    if not similarity_scores:
        return 0.0
    above = sum(1 for s in similarity_scores if s >= threshold)
    return round(above / len(similarity_scores), 4)


# ---------------------------------------------------------------------------
# 2. Consistency score
# ---------------------------------------------------------------------------

def compute_consistency_score(similarity_scores: list[float]) -> float:
    """
    Measure how consistent (low variance) the similarity scores are.

    High consistency → retrieved docs are similarly relevant → stable evidence.
    Low consistency  → mixed evidence → harder to generate a reliable insight.

    Returns
    -------
    float in [0, 1]  (1 = perfectly consistent, 0 = maximum variance)
    """
    if len(similarity_scores) < 2:
        return 1.0

    arr = np.array(similarity_scores, dtype=float)
    std = float(np.std(arr))

    # Normalise: std of uniform [0,1] scores is ~0.289 at worst
    MAX_STD = 0.5
    consistency = max(0.0, 1.0 - (std / MAX_STD))
    return round(consistency, 4)


# ---------------------------------------------------------------------------
# 3. Difficulty score
# ---------------------------------------------------------------------------

def compute_difficulty_score(
    coverage_score: float,
    consistency_score: float,
    coverage_weight: float = 0.6,
    consistency_weight: float = 0.4,
) -> float:
    """
    Combine coverage and consistency into a single difficulty score.

        difficulty_score = coverage_weight  × coverage_score
                         + consistency_weight × consistency_score

    Higher score → more evidence → easier query.

    Returns
    -------
    float in [0, 1]
    """
    score = coverage_weight * coverage_score + consistency_weight * consistency_score
    return round(float(score), 4)


# ---------------------------------------------------------------------------
# 4. Difficulty label
# ---------------------------------------------------------------------------

def label_difficulty(
    difficulty_score: float,
    easy_threshold: float = EASY_THRESHOLD,
    hard_threshold: float = HARD_THRESHOLD,
) -> str:
    """
    Map difficulty_score to a human-readable label.

        score >= easy_threshold  → 'Easy'
        score >= hard_threshold  → 'Medium'
        score <  hard_threshold  → 'Hard'
    """
    if difficulty_score >= easy_threshold:
        return "Easy"
    elif difficulty_score >= hard_threshold:
        return "Medium"
    else:
        return "Hard"


def classify_query(
    similarity_scores: list[float],
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> dict[str, Any]:
    """
    Full difficulty classification pipeline for one query.

    Parameters
    ----------
    similarity_scores : list of cosine similarities from retrieval

    Returns
    -------
    dict with keys:
        coverage_score, consistency_score, difficulty_score, difficulty_label
    """
    coverage    = compute_coverage_score(similarity_scores, threshold)
    consistency = compute_consistency_score(similarity_scores)
    difficulty  = compute_difficulty_score(coverage, consistency)
    label       = label_difficulty(difficulty)

    return {
        "coverage_score":    coverage,
        "consistency_score": consistency,
        "difficulty_score":  difficulty,
        "difficulty_label":  label,
    }


# ---------------------------------------------------------------------------
# 5. Prompt prefix builder
# ---------------------------------------------------------------------------

def build_difficulty_prompt_prefix(classification: dict[str, Any]) -> str:
    """
    Return a short text block injected at the top of the LLM prompt
    to condition the model's confidence level.

    Parameters
    ----------
    classification : output of classify_query()
    """
    label       = classification.get("difficulty_label", "Unknown")
    coverage    = classification.get("coverage_score", 0)
    consistency = classification.get("consistency_score", 0)

    prefix_map = {
        "Easy": (
            f"[QUERY DIFFICULTY: EASY | "
            f"Coverage: {coverage:.2f} | Consistency: {consistency:.2f}]\n"
            "Strong historical evidence is available for this query. "
            "You may generate a confident and detailed energy demand insight.\n\n"
        ),
        "Medium": (
            f"[QUERY DIFFICULTY: MEDIUM | "
            f"Coverage: {coverage:.2f} | Consistency: {consistency:.2f}]\n"
            "Moderate historical evidence is available. "
            "Generate an insight but note any patterns that may be uncertain.\n\n"
        ),
        "Hard": (
            f"[QUERY DIFFICULTY: HARD | "
            f"Coverage: {coverage:.2f} | Consistency: {consistency:.2f}]\n"
            "Limited or inconsistent historical evidence for this query. "
            "Generate a cautious insight and clearly state that "
            "historical data coverage is weak. Avoid overconfident claims.\n\n"
        ),
    }
    return prefix_map.get(label, "")


# ---------------------------------------------------------------------------
# 6. Caution evaluation
# ---------------------------------------------------------------------------

def evaluate_caution(answer: str, difficulty_label: str) -> dict[str, Any]:
    """
    For Hard queries, check whether the answer contains hedging / caution language.
    For Easy/Medium queries, simply return passed=True (no caution required).

    Returns
    -------
    dict:
        requires_caution   bool
        caution_detected   bool
        cautious_pass      bool   (True if caution not required OR detected)
        matched_phrases    list[str]
    """
    requires_caution = (difficulty_label == "Hard")

    if not requires_caution:
        return {
            "requires_caution": False,
            "caution_detected": False,
            "cautious_pass":    True,
            "matched_phrases":  [],
        }

    answer_lower    = answer.lower()
    matched         = [p for p in _CAUTION_PHRASES if p in answer_lower]
    caution_detected = len(matched) > 0

    return {
        "requires_caution": True,
        "caution_detected": caution_detected,
        "cautious_pass":    caution_detected,
        "matched_phrases":  matched,
    }


def aggregate_difficulty_metrics(per_query: list[dict]) -> dict[str, Any]:
    """
    Average difficulty metrics over all queries.
    Each dict should contain keys from classify_query() + evaluate_caution().
    """
    if not per_query:
        return {}

    def _mean(key: str) -> float | None:
        vals = [r[key] for r in per_query if key in r and r[key] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    hard_queries = [r for r in per_query if r.get("difficulty_label") == "Hard"]
    cautious_correct = sum(
        1 for r in hard_queries if r.get("cautious_pass", False)
    )

    return {
        "avg_coverage_score":          _mean("coverage_score"),
        "avg_consistency_score":       _mean("consistency_score"),
        "avg_difficulty_score":        _mean("difficulty_score"),
        "n_easy":                      sum(1 for r in per_query if r.get("difficulty_label") == "Easy"),
        "n_medium":                    sum(1 for r in per_query if r.get("difficulty_label") == "Medium"),
        "n_hard":                      len(hard_queries),
        "cautious_response_accuracy":  round(cautious_correct / len(hard_queries), 4)
                                       if hard_queries else None,
        "n_queries":                   len(per_query),
    }
