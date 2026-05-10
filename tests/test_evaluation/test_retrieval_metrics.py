"""Unit tests for src/evaluation/retrieval_metrics.py.

Verifies Recall@K, Precision@K, MRR, nDCG computation for individual
queries. Tests cover perfect retrieval, partial hits, complete misses,
empty inputs, and ranking order sensitivity.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.retrieval_metrics import (
    _recall_at_k,
    _precision_at_k,
    _mrr,
    _ndcg,
)


# -- Recall@K -----------------------------------------------------------------

def test_recall_perfect():
    """Recall is 1.0 when all expected docs are retrieved."""
    assert _recall_at_k(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_recall_partial():
    """Recall reflects fraction of expected docs found."""
    result = _recall_at_k(["a", "b", "x"], ["a", "b", "c"])
    assert abs(result - 2 / 3) < 1e-6


def test_recall_zero():
    """Recall is 0.0 when no expected docs appear in retrieved."""
    assert _recall_at_k(["x", "y", "z"], ["a", "b"]) == 0.0


def test_recall_empty_expected():
    """Recall is 0.0 when expected list is empty."""
    assert _recall_at_k(["a", "b"], []) == 0.0


def test_recall_empty_retrieved():
    """Recall is 0.0 when retrieved list is empty."""
    assert _recall_at_k([], ["a", "b"]) == 0.0


# -- Precision@K ---------------------------------------------------------------

def test_precision_perfect():
    """Precision is 1.0 when all retrieved docs are relevant."""
    assert _precision_at_k(["a", "b"], ["a", "b", "c"]) == 1.0


def test_precision_partial():
    """Precision reflects fraction of retrieved that are relevant."""
    result = _precision_at_k(["a", "x", "y"], ["a", "b"])
    assert abs(result - 1 / 3) < 1e-6


def test_precision_zero():
    """Precision is 0.0 when no retrieved doc is relevant."""
    assert _precision_at_k(["x", "y"], ["a", "b"]) == 0.0


def test_precision_empty_retrieved():
    """Precision is 0.0 when retrieved list is empty."""
    assert _precision_at_k([], ["a"]) == 0.0


# -- MRR -----------------------------------------------------------------------

def test_mrr_first_position():
    """MRR is 1.0 when the first relevant doc is at rank 1."""
    assert _mrr(["a", "b", "c"], ["a"]) == 1.0


def test_mrr_second_position():
    """MRR is 0.5 when the first relevant doc is at rank 2."""
    assert _mrr(["x", "a", "c"], ["a"]) == 0.5


def test_mrr_third_position():
    """MRR is 1/3 when the first relevant doc is at rank 3."""
    result = _mrr(["x", "y", "a"], ["a"])
    assert abs(result - 1 / 3) < 1e-6


def test_mrr_no_hit():
    """MRR is 0.0 when no relevant doc appears in retrieved."""
    assert _mrr(["x", "y", "z"], ["a"]) == 0.0


def test_mrr_multiple_relevant():
    """MRR uses the rank of the FIRST relevant doc."""
    assert _mrr(["x", "a", "b"], ["a", "b"]) == 0.5


# -- nDCG ----------------------------------------------------------------------

def test_ndcg_perfect_ranking():
    """nDCG is 1.0 when all relevant docs are at the top."""
    result = _ndcg(["a", "b", "x"], ["a", "b"])
    assert abs(result - 1.0) < 1e-6


def test_ndcg_imperfect_ranking():
    """nDCG < 1.0 when relevant docs are not at the top."""
    result = _ndcg(["x", "a", "b"], ["a", "b"])
    assert result < 1.0
    assert result > 0.0


def test_ndcg_no_hit():
    """nDCG is 0.0 when no relevant doc appears."""
    assert _ndcg(["x", "y", "z"], ["a", "b"]) == 0.0


def test_ndcg_empty_expected():
    """nDCG is 0.0 when expected list is empty."""
    assert _ndcg(["a", "b"], []) == 0.0


def test_ndcg_empty_retrieved():
    """nDCG is 0.0 when retrieved list is empty."""
    assert _ndcg([], ["a"]) == 0.0


def test_ndcg_single_hit_at_rank1():
    """nDCG is 1.0 when the only relevant doc is at rank 1."""
    result = _ndcg(["a", "x", "y"], ["a"])
    assert abs(result - 1.0) < 1e-6


def test_ndcg_single_hit_at_rank2():
    """nDCG < 1.0 when the only relevant doc is at rank 2."""
    result = _ndcg(["x", "a", "y"], ["a"])
    # DCG = 1/log2(3) = 0.6309, IDCG = 1/log2(2) = 1.0
    expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
    assert abs(result - expected) < 1e-4
