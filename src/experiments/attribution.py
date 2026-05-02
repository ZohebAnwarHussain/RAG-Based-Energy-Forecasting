"""
src/experiments/attribution.py
================================
Novelty 1 — Evidence-Linked Retrieval Attribution

Responsibilities
----------------
1. assign_evidence_ids()   — tag each retrieved doc with [E1], [E2], ...
2. build_attributed_context() — format context block with IDs for the prompt
3. parse_citations()        — extract [En] references from LLM output
4. compute_attribution_metrics() — coverage, accuracy, unsupported claim rate
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# 1. Assign evidence IDs
# ---------------------------------------------------------------------------

def assign_evidence_ids(docs: list[dict]) -> list[dict]:
    """
    Add an 'evidence_id' field (e.g. 'E1', 'E2', ...) to each retrieved doc.

    Parameters
    ----------
    docs : list of dicts, each with at least a 'text' or 'page_content' key.

    Returns
    -------
    Same list with 'evidence_id' added in-place.
    """
    for i, doc in enumerate(docs, start=1):
        doc["evidence_id"] = f"E{i}"
    return docs


# ---------------------------------------------------------------------------
# 2. Build attributed context string for the prompt
# ---------------------------------------------------------------------------

def build_attributed_context(docs: list[dict]) -> str:
    """
    Format retrieved docs into a numbered context block.

    Output format:
        [E1] Zone: 1 | Date: 2012-01-01 | Granularity: daily
        <summary text>

        [E2] ...
    """
    lines = []
    for doc in docs:
        eid      = doc.get("evidence_id", "?")
        text     = doc.get("text") or doc.get("page_content", "")
        metadata = doc.get("metadata", {})

        meta_str = " | ".join(
            f"{k}: {v}" for k, v in metadata.items()
            if k in ("zone", "date", "granularity", "time_window", "source")
        )

        header = f"[{eid}]"
        if meta_str:
            header += f" {meta_str}"

        lines.append(f"{header}\n{text.strip()}\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Parse citations from LLM output
# ---------------------------------------------------------------------------

_CITATION_RE = re.compile(r"\[E(\d+)\]")


def parse_citations(text: str) -> list[str]:
    """
    Extract all [En] citation references from *text*.

    Returns
    -------
    Sorted, deduplicated list of evidence IDs, e.g. ['E1', 'E2', 'E4'].
    """
    matches = _CITATION_RE.findall(text)
    return sorted(set(f"E{m}" for m in matches), key=lambda x: int(x[1:]))


def parse_claims(text: str) -> list[str]:
    """
    Split generated insight into individual claims (sentences).
    A 'claim' is any sentence that ends in '.', '!', or '?'.
    """
    # Basic sentence splitter — good enough for structured energy insights
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


# ---------------------------------------------------------------------------
# 4. Attribution metrics
# ---------------------------------------------------------------------------

def compute_attribution_metrics(
    answer: str,
    docs: list[dict],
) -> dict[str, Any]:
    """
    Compute Novelty 1 attribution metrics for a single answer.

    Parameters
    ----------
    answer : The LLM-generated insight string.
    docs   : Retrieved docs WITH evidence_id assigned.

    Returns
    -------
    dict with keys:
        total_claims            int
        claims_with_citation    int
        attribution_coverage    float  (0–1)
        available_evidence_ids  list[str]
        cited_evidence_ids      list[str]
        correct_citations       int    (cited IDs that exist in docs)
        spurious_citations      int    (cited IDs NOT in docs)
        citation_accuracy       float  (0–1)
        unsupported_claim_rate  float  (0–1)
    """
    available_ids = {doc["evidence_id"] for doc in docs if "evidence_id" in doc}
    cited_ids     = set(parse_citations(answer))
    claims        = parse_claims(answer)

    total_claims = len(claims)

    # A claim "has a citation" if [En] appears within 120 chars before/after it
    # Simpler proxy: count sentences containing at least one [En]
    claims_with_citation = sum(
        1 for c in claims if _CITATION_RE.search(c)
    )

    correct_citations  = len(cited_ids & available_ids)
    spurious_citations = len(cited_ids - available_ids)
    total_cited        = len(cited_ids)

    attribution_coverage = (
        claims_with_citation / total_claims if total_claims > 0 else 0.0
    )
    citation_accuracy = (
        correct_citations / total_cited if total_cited > 0 else 0.0
    )
    unsupported_claim_rate = (
        1.0 - attribution_coverage
    )

    return {
        "total_claims":           total_claims,
        "claims_with_citation":   claims_with_citation,
        "attribution_coverage":   round(attribution_coverage, 4),
        "available_evidence_ids": sorted(available_ids, key=lambda x: int(x[1:])),
        "cited_evidence_ids":     sorted(cited_ids,     key=lambda x: int(x[1:])),
        "correct_citations":      correct_citations,
        "spurious_citations":     spurious_citations,
        "citation_accuracy":      round(citation_accuracy, 4),
        "unsupported_claim_rate": round(unsupported_claim_rate, 4),
    }


def aggregate_attribution_metrics(per_query: list[dict]) -> dict[str, Any]:
    """
    Average attribution metrics over a list of per-query result dicts.
    Each dict should be the output of compute_attribution_metrics().
    """
    if not per_query:
        return {}

    scalar_keys = [
        "total_claims", "claims_with_citation",
        "attribution_coverage", "correct_citations",
        "citation_accuracy", "unsupported_claim_rate",
    ]

    agg: dict[str, Any] = {}
    for k in scalar_keys:
        vals = [r[k] for r in per_query if k in r]
        agg[f"avg_{k}"] = round(sum(vals) / len(vals), 4) if vals else None

    agg["n_queries"] = len(per_query)
    return agg
