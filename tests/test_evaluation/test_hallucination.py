"""Unit tests for src/evaluation/hallucination.py.

Verifies the keyword-based hallucination check: must_include detection,
must_not_include detection, edge cases with empty inputs, and
case-insensitive matching.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.hallucination import check_hallucination


# -- Include checks -----------------------------------------------------------

def test_all_must_include_present():
    """include_pass is True when all must_include terms appear."""
    result = check_hallucination(
        answer="Zone 1 had peak demand of 800 MW in winter.",
        must_include=["Zone 1", "MW", "peak"],
        must_not_include=[],
    )
    assert result["include_pass"] is True
    assert result["missing_terms"] == []


def test_missing_must_include_term():
    """include_pass is False when a required term is absent."""
    result = check_hallucination(
        answer="The demand was high in winter.",
        must_include=["Zone 1", "MW"],
        must_not_include=[],
    )
    assert result["include_pass"] is False
    assert "Zone 1" in result["missing_terms"]
    assert "MW" in result["missing_terms"]


def test_must_include_is_case_insensitive():
    """Matching must be case-insensitive."""
    result = check_hallucination(
        answer="zone 1 had 500 mw demand.",
        must_include=["Zone 1", "MW"],
        must_not_include=[],
    )
    assert result["include_pass"] is True


# -- Exclude checks -----------------------------------------------------------

def test_no_forbidden_terms_present():
    """exclude_pass is True when no forbidden terms appear."""
    result = check_hallucination(
        answer="Zone 1 demand was 800 MW.",
        must_include=[],
        must_not_include=["household", "I think"],
    )
    assert result["exclude_pass"] is True
    assert result["forbidden_terms"] == []


def test_forbidden_term_detected():
    """exclude_pass is False when a forbidden term is found."""
    result = check_hallucination(
        answer="I think the household demand was low.",
        must_include=[],
        must_not_include=["household", "I think"],
    )
    assert result["exclude_pass"] is False
    assert "household" in result["forbidden_terms"]
    assert "I think" in result["forbidden_terms"]


# -- Overall pass --------------------------------------------------------------

def test_overall_pass_both_checks_pass():
    """overall_pass is True only when both include and exclude pass."""
    result = check_hallucination(
        answer="Zone 1 had 800 MW peak demand.",
        must_include=["Zone 1", "MW"],
        must_not_include=["household"],
    )
    assert result["overall_pass"] is True


def test_overall_fail_include_fails():
    """overall_pass is False when include check fails."""
    result = check_hallucination(
        answer="Demand was high.",
        must_include=["Zone 1"],
        must_not_include=[],
    )
    assert result["overall_pass"] is False


def test_overall_fail_exclude_fails():
    """overall_pass is False when exclude check fails."""
    result = check_hallucination(
        answer="I think Zone 1 had 800 MW.",
        must_include=["Zone 1"],
        must_not_include=["I think"],
    )
    assert result["overall_pass"] is False


# -- Edge cases ---------------------------------------------------------------

def test_empty_must_include():
    """include_pass is True when must_include list is empty."""
    result = check_hallucination(
        answer="Any answer at all.",
        must_include=[],
        must_not_include=[],
    )
    assert result["include_pass"] is True


def test_empty_must_not_include():
    """exclude_pass is True when must_not_include list is empty."""
    result = check_hallucination(
        answer="Any answer at all.",
        must_include=[],
        must_not_include=[],
    )
    assert result["exclude_pass"] is True


def test_empty_answer():
    """Empty answer should fail include check if terms required."""
    result = check_hallucination(
        answer="",
        must_include=["Zone 1"],
        must_not_include=[],
    )
    assert result["include_pass"] is False


def test_return_keys():
    """Result dict must contain all expected keys."""
    result = check_hallucination("test", ["a"], ["b"])
    expected_keys = {
        "missing_terms", "forbidden_terms",
        "include_pass", "exclude_pass", "overall_pass",
    }
    assert set(result.keys()) == expected_keys
