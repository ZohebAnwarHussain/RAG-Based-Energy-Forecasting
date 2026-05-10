"""Unit tests for src/experiments/difficulty.py (Novelty 2).

Verifies coverage score, consistency score, difficulty score,
label assignment, prompt prefix building, and caution evaluation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.experiments.difficulty import (
    compute_coverage_score,
    compute_consistency_score,
    compute_difficulty_score,
    label_difficulty,
    classify_query,
    build_difficulty_prompt_prefix,
    evaluate_caution,
    EASY_THRESHOLD,
    HARD_THRESHOLD,
)


# -- compute_coverage_score ----------------------------------------------------

def test_coverage_all_above_threshold():
    """Coverage is 1.0 when all scores exceed the threshold."""
    assert compute_coverage_score([0.8, 0.9, 0.7], threshold=0.5) == 1.0


def test_coverage_none_above_threshold():
    """Coverage is 0.0 when no score exceeds the threshold."""
    assert compute_coverage_score([0.1, 0.2, 0.3], threshold=0.5) == 0.0


def test_coverage_partial():
    """Coverage reflects the fraction above threshold."""
    result = compute_coverage_score([0.6, 0.4, 0.8], threshold=0.5)
    assert abs(result - 2 / 3) < 1e-4


def test_coverage_empty_scores():
    """Coverage is 0.0 for empty score list."""
    assert compute_coverage_score([]) == 0.0


# -- compute_consistency_score -------------------------------------------------

def test_consistency_identical_scores():
    """Consistency is 1.0 when all scores are identical."""
    assert compute_consistency_score([0.7, 0.7, 0.7]) == 1.0


def test_consistency_varied_scores():
    """Consistency < 1.0 when scores vary."""
    result = compute_consistency_score([0.1, 0.9, 0.5])
    assert result < 1.0
    assert result > 0.0


def test_consistency_single_score():
    """Consistency is 1.0 with only one score."""
    assert compute_consistency_score([0.5]) == 1.0


# -- compute_difficulty_score --------------------------------------------------

def test_difficulty_score_range():
    """Difficulty score is in [0, 1]."""
    score = compute_difficulty_score(0.8, 0.9)
    assert 0.0 <= score <= 1.0


def test_difficulty_score_weighted_sum():
    """Difficulty score follows the weighted formula."""
    # 0.6 * 0.8 + 0.4 * 0.6 = 0.48 + 0.24 = 0.72
    result = compute_difficulty_score(0.8, 0.6, coverage_weight=0.6, consistency_weight=0.4)
    assert abs(result - 0.72) < 1e-4


# -- label_difficulty ----------------------------------------------------------

def test_label_easy():
    """Score above EASY_THRESHOLD is labelled Easy."""
    assert label_difficulty(EASY_THRESHOLD + 0.01) == "Easy"


def test_label_medium():
    """Score between HARD and EASY thresholds is labelled Medium."""
    mid = (EASY_THRESHOLD + HARD_THRESHOLD) / 2
    assert label_difficulty(mid) == "Medium"


def test_label_hard():
    """Score below HARD_THRESHOLD is labelled Hard."""
    assert label_difficulty(HARD_THRESHOLD - 0.01) == "Hard"


# -- classify_query ------------------------------------------------------------

def test_classify_query_returns_expected_keys():
    """classify_query must return all expected fields."""
    result = classify_query([0.8, 0.7, 0.9])
    expected_keys = {
        "coverage_score", "consistency_score",
        "difficulty_score", "difficulty_label",
    }
    assert set(result.keys()) == expected_keys


def test_classify_query_high_scores_easy():
    """High similarity scores should classify as Easy."""
    result = classify_query([0.9, 0.85, 0.88, 0.92, 0.87])
    assert result["difficulty_label"] == "Easy"


def test_classify_query_low_scores_hard():
    """Low similarity scores should classify as Hard."""
    result = classify_query([0.1, 0.05, 0.15, 0.08, 0.12])
    assert result["difficulty_label"] == "Hard"


# -- build_difficulty_prompt_prefix -------------------------------------------

def test_prompt_prefix_easy():
    """Easy prefix should mention confident generation."""
    prefix = build_difficulty_prompt_prefix({
        "difficulty_label": "Easy",
        "coverage_score": 0.9,
        "consistency_score": 0.85,
    })
    assert "EASY" in prefix
    assert "confident" in prefix.lower()


def test_prompt_prefix_hard():
    """Hard prefix should mention caution."""
    prefix = build_difficulty_prompt_prefix({
        "difficulty_label": "Hard",
        "coverage_score": 0.2,
        "consistency_score": 0.3,
    })
    assert "HARD" in prefix
    assert "caution" in prefix.lower() or "cautious" in prefix.lower()


def test_prompt_prefix_contains_scores():
    """Prefix must include the numeric coverage and consistency values."""
    prefix = build_difficulty_prompt_prefix({
        "difficulty_label": "Medium",
        "coverage_score": 0.55,
        "consistency_score": 0.60,
    })
    assert "0.55" in prefix
    assert "0.60" in prefix


# -- evaluate_caution ---------------------------------------------------------

def test_caution_not_required_for_easy():
    """Easy queries do not require caution language."""
    result = evaluate_caution("Demand was very high.", "Easy")
    assert result["requires_caution"] is False
    assert result["cautious_pass"] is True


def test_caution_detected_for_hard():
    """Hard query with caution language passes."""
    result = evaluate_caution(
        "There is limited evidence for this period. Exercise caution.",
        "Hard",
    )
    assert result["requires_caution"] is True
    assert result["caution_detected"] is True
    assert result["cautious_pass"] is True
    assert len(result["matched_phrases"]) > 0


def test_caution_missing_for_hard():
    """Hard query without caution language fails."""
    result = evaluate_caution(
        "Zone 1 demand was exactly 15000 MW in winter.",
        "Hard",
    )
    assert result["requires_caution"] is True
    assert result["caution_detected"] is False
    assert result["cautious_pass"] is False


def test_caution_return_keys():
    """Result must contain all expected keys."""
    result = evaluate_caution("test", "Easy")
    expected_keys = {
        "requires_caution", "caution_detected",
        "cautious_pass", "matched_phrases",
    }
    assert set(result.keys()) == expected_keys
