"""Unit tests for src/experiments/attribution.py (Novelty 1).

Verifies evidence ID assignment, citation parsing, attributed context
building, and attribution metric computation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.experiments.attribution import (
    assign_evidence_ids,
    build_attributed_context,
    parse_citations,
    parse_claims,
    compute_attribution_metrics,
)


# -- assign_evidence_ids -------------------------------------------------------

def test_assign_evidence_ids_adds_ids():
    """Each doc gets a sequential E1, E2, ... label."""
    docs = [{"text": "doc1"}, {"text": "doc2"}, {"text": "doc3"}]
    result = assign_evidence_ids(docs)
    assert result[0]["evidence_id"] == "E1"
    assert result[1]["evidence_id"] == "E2"
    assert result[2]["evidence_id"] == "E3"


def test_assign_evidence_ids_modifies_in_place():
    """assign_evidence_ids should mutate the original list."""
    docs = [{"text": "doc1"}]
    assign_evidence_ids(docs)
    assert "evidence_id" in docs[0]


def test_assign_evidence_ids_empty_list():
    """Empty doc list returns empty list without error."""
    assert assign_evidence_ids([]) == []


# -- parse_citations -----------------------------------------------------------

def test_parse_citations_single():
    """Single [E1] reference extracted correctly."""
    result = parse_citations("Zone 1 demand was high [E1].")
    assert result == ["E1"]


def test_parse_citations_multiple():
    """Multiple [En] references extracted and sorted."""
    result = parse_citations("Demand [E3] was high [E1] in winter [E2].")
    assert result == ["E1", "E2", "E3"]


def test_parse_citations_duplicates():
    """Duplicate [En] references are deduplicated."""
    result = parse_citations("[E1] supports claim. Also [E1] here.")
    assert result == ["E1"]


def test_parse_citations_none_found():
    """No citations returns empty list."""
    assert parse_citations("No citations here.") == []


def test_parse_citations_partial_match():
    """[P1] parent citations should NOT match the [En] pattern."""
    result = parse_citations("[P1] parent info. [E2] child info.")
    assert result == ["E2"]


# -- parse_claims --------------------------------------------------------------

def test_parse_claims_splits_sentences():
    """Claims are split on sentence boundaries."""
    claims = parse_claims(
        "Zone 1 had peak demand. It was 800 MW. Winter was the peak season."
    )
    assert len(claims) == 3


def test_parse_claims_filters_short():
    """Very short fragments (< 10 chars) are filtered out."""
    claims = parse_claims("OK. This is a much longer claim about energy demand.")
    # "OK" is < 10 chars, should be filtered
    assert len(claims) == 1


# -- build_attributed_context -------------------------------------------------

def test_build_attributed_context_includes_evidence_ids():
    """Context block includes [E1], [E2] headers."""
    docs = [
        {"evidence_id": "E1", "text": "Zone 1 data.", "metadata": {"zone": "1"}},
        {"evidence_id": "E2", "text": "Zone 2 data.", "metadata": {"zone": "2"}},
    ]
    context = build_attributed_context(docs)
    assert "[E1]" in context
    assert "[E2]" in context
    assert "Zone 1 data." in context


def test_build_attributed_context_empty_docs():
    """Empty doc list returns empty string."""
    assert build_attributed_context([]) == ""


# -- compute_attribution_metrics -----------------------------------------------

def test_attribution_metrics_full_coverage():
    """All claims cited yields coverage = 1.0."""
    docs = [
        {"evidence_id": "E1", "text": "Zone 1 load."},
        {"evidence_id": "E2", "text": "Zone 2 load."},
    ]
    answer = (
        "Zone 1 had high demand in winter [E1]. "
        "Zone 2 was lower in summer [E2]."
    )
    result = compute_attribution_metrics(answer, docs)
    assert result["attribution_coverage"] == 1.0
    assert result["citation_accuracy"] == 1.0
    assert result["spurious_citations"] == 0


def test_attribution_metrics_no_citations():
    """No citations yields coverage = 0.0."""
    docs = [{"evidence_id": "E1", "text": "Zone 1 load."}]
    answer = "Zone 1 had high demand in winter."
    result = compute_attribution_metrics(answer, docs)
    assert result["attribution_coverage"] == 0.0
    assert result["unsupported_claim_rate"] == 1.0


def test_attribution_metrics_spurious_citation():
    """Citation to non-existent evidence is counted as spurious."""
    docs = [{"evidence_id": "E1", "text": "Zone 1 load."}]
    answer = "Zone 1 had demand [E1]. Also [E5] confirms this."
    result = compute_attribution_metrics(answer, docs)
    assert result["spurious_citations"] == 1
    assert result["citation_accuracy"] < 1.0


def test_attribution_metrics_return_keys():
    """Result dict must contain all expected keys."""
    docs = [{"evidence_id": "E1", "text": "test"}]
    result = compute_attribution_metrics("test [E1].", docs)
    expected_keys = {
        "total_claims", "claims_with_citation", "attribution_coverage",
        "available_evidence_ids", "cited_evidence_ids",
        "correct_citations", "spurious_citations",
        "citation_accuracy", "unsupported_claim_rate",
    }
    assert expected_keys.issubset(set(result.keys()))
