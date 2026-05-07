"""
experiments/exp_07_hierarchical_attribution.py  (v2 — improved retrieval)
===========================================================================
EXP_07_HIERARCHICAL_RAG_ATTRIBUTION — Hierarchical RAG + Evidence-Linked Attribution

CHANGES vs v1
--------------
IMPROVEMENT 1 — Cross-Encoder Reranking on child candidates
  Same as EXP_04 v2: reranker applied to child pool before parent expansion.
  Attribution prompt unchanged (already grounded via rules 1–5).
  Query expansion NOT applied (same rationale as EXP_04).

RUN_MODE:
  'baseline'  → original v1 behaviour
  'rerank'    → child reranking only (recommended, same as 'full' here)
  'full'      → same as 'rerank' for this experiment
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from config.models import MODELS, EXP_DEFAULTS
from config.paths import PATHS
from src.embedding.embedder import get_embeddings_model
from src.embedding.faiss_store import load_faiss_index
from src.retrieval.hierarchical import HierarchicalRetriever
from src.retrieval.reranker import CrossEncoderReranker, RECALL_MULTIPLIER
from src.evaluation.retrieval_metrics import (
    _recall_at_k, _precision_at_k, _mrr, _ndcg,
)
from src.evaluation.hallucination import check_hallucination
from src.experiments.groq_client import RotatingGroqClient
from src.experiments.metrics import (
        compute_answer_relevance,
        compute_semantic_similarity,
        compute_hallucination_rate,
        compute_insight_clarity,
        is_useful_answer,
        compute_retrieval_metrics_with_content_fallback,   # content-fallback fix
    )
from src.experiments.attribution import (
    compute_attribution_metrics,
    parse_citations,
    parse_claims,
)
from experiments.runner import (
    run_experiment,
    ExperimentResult,
    _save_results,
)

import pandas as pd

logger = logging.getLogger(__name__)

EXP_ID   = "EXP_07_HIERARCHICAL_RAG_ATTRIBUTION"
PIPELINE = "hierarchical_attribution"

RUN_MODE = "full"   # 'baseline' | 'rerank' | 'full' (rerank = full here)

# ---------------------------------------------------------------------------
# Attribution prompt (unchanged from v1 — child/parent [En]/[Pn] citation)
# ---------------------------------------------------------------------------

ATTRIBUTION_SYSTEM_MESSAGE = (
    "You are an expert energy systems analyst providing data-driven "
    "demand insights to utility managers and energy planners.\n\n"
    "Rules:\n"
    "1. Base your answer STRICTLY on the provided context summaries.\n"
    "2. Use specific numbers, dates, and zone identifiers from the context.\n"
    "3. If the context does not contain enough information to fully answer "
    "the question, explicitly state what information is missing.\n"
    "4. Do NOT hallucinate or introduce facts not present in the context.\n"
    "5. Do NOT speculate about causes unless the context supports it.\n"
    "6. Write 3-5 sentences in clear, stakeholder-friendly language.\n"
    "7. When comparing values, state both numbers and the direction of "
    "the difference (higher/lower, increase/decrease).\n"
    "8. CITATION REQUIREMENT: After each factual claim or sentence, cite "
    "the evidence document(s) it came from using square-bracket tags. "
    "Child (daily) evidence uses [E1], [E2], etc. "
    "Parent (weekly/monthly) evidence uses [P1], [P2], etc. "
    "Use child evidence [En] for specific peak/hourly claims. "
    "Use parent evidence [Pn] for broader trend or weekly/monthly claims. "
    "Every sentence must end with at least one citation."
)

ATTRIBUTION_HUMAN_TEMPLATE = (
    "Context (retrieved from the Energy Knowledge Base):\n"
    "Child summaries (daily detail) — cite as [E1], [E2], ...:\n"
    "---\n"
    "{child_context}\n"
    "---\n\n"
    "Parent summaries (weekly/monthly context) — cite as [P1], [P2], ...:\n"
    "---\n"
    "{parent_context}\n"
    "---\n\n"
    "Question: {question}\n\n"
    "Provide a factual, evidence-grounded answer. "
    "Cite child evidence [En] for specific claims and parent evidence [Pn] for broader trends."
)


def _build_attribution_messages(
    child_context: str, parent_context: str, question: str
) -> list[dict]:
    return [
        {"role": "system", "content": ATTRIBUTION_SYSTEM_MESSAGE},
        {"role": "user",   "content": ATTRIBUTION_HUMAN_TEMPLATE.format(
            child_context=child_context,
            parent_context=parent_context,
            question=question,
        )},
    ]


# ---------------------------------------------------------------------------
# Hierarchical attributed context (unchanged from v1)
# ---------------------------------------------------------------------------

_CHILD_CITATION_RE  = re.compile(r"\[E(\d+)\]")
_PARENT_CITATION_RE = re.compile(r"\[P(\d+)\]")


def _build_hierarchical_attributed_context(child_docs, parent_docs):
    children_with_ids = [dict(doc, evidence_id=f"E{i}") for i, doc in enumerate(child_docs, 1)]
    parents_with_ids  = [dict(doc, evidence_id=f"P{i}") for i, doc in enumerate(parent_docs, 1)]

    def _fmt(docs):
        lines = []
        for doc in docs:
            eid  = doc.get("evidence_id", "?")
            text = doc.get("page_content", doc.get("text", ""))
            meta = doc.get("metadata", doc)
            src  = meta.get("source", meta.get("row_id", ""))
            gran = meta.get("granularity", "")
            lines.append(f"[{eid}] source: {src} | granularity: {gran}\n{text.strip()}\n")
        return "\n".join(lines) if lines else "None available."

    return _fmt(children_with_ids), _fmt(parents_with_ids), children_with_ids, parents_with_ids


def _compute_hierarchical_attribution(answer, children_with_ids, parents_with_ids):
    available_child_ids  = {d["evidence_id"] for d in children_with_ids}
    available_parent_ids = {d["evidence_id"] for d in parents_with_ids}

    child_cited  = {f"E{m}" for m in _CHILD_CITATION_RE.findall(answer)}
    parent_cited = {f"P{m}" for m in _PARENT_CITATION_RE.findall(answer)}

    claims = parse_claims(answer)
    total_claims = len(claims)
    claims_with_any = sum(
        1 for c in claims
        if _CHILD_CITATION_RE.search(c) or _PARENT_CITATION_RE.search(c)
    )

    correct_child   = len(child_cited  & available_child_ids)
    spurious_child  = len(child_cited  - available_child_ids)
    correct_parent  = len(parent_cited & available_parent_ids)
    spurious_parent = len(parent_cited - available_parent_ids)

    total_cited   = len(child_cited) + len(parent_cited)
    total_correct = correct_child + correct_parent

    return {
        "total_claims":             total_claims,
        "claims_with_citation":     claims_with_any,
        "attribution_coverage":     round(claims_with_any / total_claims, 4) if total_claims else 0.0,
        "citation_accuracy":        round(total_correct / total_cited, 4) if total_cited else 0.0,
        "unsupported_claim_rate":   round(1.0 - (claims_with_any / total_claims), 4) if total_claims else 0.0,
        "correct_citations":        total_correct,
        "spurious_citations":       total_correct + spurious_child + spurious_parent - total_correct,
        "child_cited_ids":          sorted(child_cited,  key=lambda x: int(x[1:])),
        "correct_child_citations":  correct_child,
        "spurious_child_citations": spurious_child,
        "parent_cited_ids":         sorted(parent_cited, key=lambda x: int(x[1:])),
        "correct_parent_citations": correct_parent,
        "spurious_parent_citations":spurious_parent,
    }


# ---------------------------------------------------------------------------
# FAISS cache
# ---------------------------------------------------------------------------

_faiss_store = None
_embeddings  = None
_documents   = None


def _get_faiss_and_docs():
    global _faiss_store, _embeddings, _documents
    if _faiss_store is None:
        logger.info("Loading embedding model and FAISS index...")
        _embeddings  = get_embeddings_model()
        _faiss_store = load_faiss_index(PATHS["faiss_index"], _embeddings)
        _documents   = list(_faiss_store.docstore._dict.values())
        logger.info("FAISS loaded. %d documents.", len(_documents))
    return _faiss_store, _documents


def _parse_ids(raw) -> list[str]:
    try:
        return json.loads(raw) if isinstance(raw, str) else list(raw)
    except Exception:
        return []


def _parse_list(raw) -> list[str]:
    try:
        return json.loads(raw) if isinstance(raw, str) else list(raw)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Module-level KB summary lookup (loaded once, used for content fallback)
# ---------------------------------------------------------------------------

_id_to_summary: dict[str, str] = {}


def _load_id_to_summary() -> dict[str, str]:
    """
    Load row_id -> summary text for retrieval content-fallback scoring.
    Handles ~10 queries whose golden-dataset zone labels don't match
    any KB chunk -- pure ID matching always scores these as zero.
    """
    global _id_to_summary
    if _id_to_summary:
        return _id_to_summary
    csv_path = PATHS["summaries_csv"] / "combined_master_summaries.csv"
    df = pd.read_csv(csv_path, usecols=["row_id", "summary"])
    _id_to_summary = dict(zip(df["row_id"].astype(str), df["summary"].astype(str)))
    logger.info("Loaded %d KB summaries for retrieval content fallback.", len(_id_to_summary))
    return _id_to_summary


# ---------------------------------------------------------------------------
# generate_fn factory
# ---------------------------------------------------------------------------

def make_generate_fn(
    client:   RotatingGroqClient,
    k:        int,
    run_mode: str = RUN_MODE,
) -> Any:
    faiss_store, documents = _get_faiss_and_docs()
    id_to_summary  = _load_id_to_summary()   # loaded once, reused

    use_rerank = run_mode in ("rerank", "full")
    fetch_k    = k * RECALL_MULTIPLIER if use_rerank else k

    retriever = HierarchicalRetriever(faiss_store, documents, k=fetch_k)
    reranker  = CrossEncoderReranker() if use_rerank else None

    logger.info("EXP_07 generate_fn: k=%d fetch_k=%d mode=%s rerank=%s",
                k, fetch_k, run_mode, use_rerank)

    def generate_fn(query: dict, _retrieved_docs: list, top_k: int) -> dict:
        question         = query.get("question", query.get("user_query", ""))
        ground_truth     = query.get("ground_truth", query.get("reference_answer", ""))
        expected_ids     = _parse_ids(query.get("expected_summary_ids", "[]"))
        must_include     = _parse_list(query.get("answer_must_include", "[]"))
        must_not_include = _parse_list(query.get("answer_must_not_include", "[]"))

        # ── Hierarchical retrieval ───────────────────────────────────────────
        all_docs_scored = retriever.retrieve_with_scores(question, k=fetch_k)
        child_docs_raw  = [d for d in all_docs_scored if d["retrieval_method"] == "dense_child"]
        parent_docs_raw = [d for d in all_docs_scored if d["retrieval_method"] == "parent_expansion"]

        # ── Rerank children, keep top K ──────────────────────────────────────
        if use_rerank and reranker and child_docs_raw:
            child_candidates = [
                {"row_id": d["row_id"], "text": d["page_content"],
                 "score": d["score"], "page_content": d["page_content"],
                 "metadata": d, "retrieval_method": d["retrieval_method"]}
                for d in child_docs_raw
            ]
            reranked = reranker.rerank(
                query=question, candidates=child_candidates,
                top_k=top_k, text_key="text",
            )
            kept_ids   = {d["row_id"] for d in reranked}
            child_docs = [d for d in child_docs_raw if d["row_id"] in kept_ids][:top_k]
        else:
            child_docs = child_docs_raw[:top_k]

        # Re-filter parents to those belonging to kept children
        kept_parent_ids = {
            d.get("parent_id", "") or d.get("metadata", {}).get("parent_id", "")
            for d in child_docs
        }
        parent_docs = (
            [d for d in parent_docs_raw if d["row_id"] in kept_parent_ids]
            if any(kept_parent_ids) else parent_docs_raw
        )

        child_ids    = [d["row_id"] for d in child_docs]
        child_scores = [d.get("rerank_score", d.get("score", 0.0)) for d in child_docs]

        # Relevant chunk texts for content fallback
        retrieved_texts = [d.get("page_content", "") for d in child_docs]
        relevant_texts  = [id_to_summary.get(eid, "") for eid in expected_ids]

        _ret_metrics = compute_retrieval_metrics_with_content_fallback(
            retrieved_ids   = child_ids,
            relevant_ids    = expected_ids,
            k               = top_k,
            retrieved_texts = retrieved_texts,
            relevant_texts  = relevant_texts,
        )
        retrieval_metrics = {
            **_ret_metrics,
            "n_children":      len(child_docs),
            "n_parents_added": len(parent_docs),
        }

        # ── Build hierarchical attributed context ────────────────────────────
        (child_context, parent_context,
         children_with_ids, parents_with_ids) = _build_hierarchical_attributed_context(
            child_docs, parent_docs
        )

        # ── Generation ───────────────────────────────────────────────────────
        messages = _build_attribution_messages(child_context, parent_context, question)
        response = client.chat(
            messages=messages, model=MODELS["groq_rag"],
            temperature=EXP_DEFAULTS["temperature"],
            max_tokens=EXP_DEFAULTS["max_tokens"],
        )
        answer = response.choices[0].message.content.strip()

        attr_metrics = _compute_hierarchical_attribution(
            answer, children_with_ids, parents_with_ids
        )
        halluc_check = check_hallucination(answer, must_include, must_not_include)
        all_context_docs = [{"text": d["page_content"]} for d in child_docs + parent_docs]

        metrics = {
            **retrieval_metrics,
            "answer_relevance":    compute_answer_relevance(question, answer),
            "semantic_similarity": compute_semantic_similarity(answer, ground_truth)
                                   if ground_truth else None,
            "hallucination_rate":  compute_hallucination_rate(answer, all_context_docs),
            "insight_clarity":     compute_insight_clarity(answer),
            "is_useful":           int(is_useful_answer(answer, question)),
            "answer_word_count":   len(answer.split()),
            "include_pass":        int(halluc_check["include_pass"]),
            "exclude_pass":        int(halluc_check["exclude_pass"]),
            "overall_pass":        int(halluc_check["overall_pass"]),
            "missing_terms":       json.dumps(halluc_check["missing_terms"]),
            "forbidden_terms":     json.dumps(halluc_check["forbidden_terms"]),
            "attribution_coverage":    attr_metrics["attribution_coverage"],
            "citation_accuracy":       attr_metrics["citation_accuracy"],
            "unsupported_claim_rate":  attr_metrics["unsupported_claim_rate"],
            "total_claims":            attr_metrics["total_claims"],
            "claims_with_citation":    attr_metrics["claims_with_citation"],
            "correct_citations":       attr_metrics["correct_citations"],
            "spurious_citations":      attr_metrics["spurious_citations"],
            "correct_child_citations":   attr_metrics["correct_child_citations"],
            "spurious_child_citations":  attr_metrics["spurious_child_citations"],
            "correct_parent_citations":  attr_metrics["correct_parent_citations"],
            "spurious_parent_citations": attr_metrics["spurious_parent_citations"],
            "child_cited_ids":           json.dumps(attr_metrics["child_cited_ids"]),
            "parent_cited_ids":          json.dumps(attr_metrics["parent_cited_ids"]),
            "retrieved_ids":             json.dumps(child_ids),
            "similarity_scores":         json.dumps([round(s, 4) for s in child_scores]),
            "run_mode":                  run_mode,
            "used_rerank":               int(use_rerank),
        }

        return {
            "answer": answer,
            "retrieved_docs": [
                {"text": d["page_content"], "evidence_id": f"E{i+1}",
                 "retrieval_method": "dense_child"}
                for i, d in enumerate(child_docs)
            ] + [
                {"text": d["page_content"], "evidence_id": f"P{i+1}",
                 "retrieval_method": "parent_expansion"}
                for i, d in enumerate(parent_docs)
            ],
            "metrics": metrics,
        }

    return generate_fn


def _compute_agg(results: list[ExperimentResult]) -> None:
    for result in results:
        qrs   = result.query_results
        valid = [qr for qr in qrs if not qr.error]

        def _mean(key: str):
            vals = [qr.metrics.get(key) for qr in valid if qr.metrics.get(key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        def _rate(key: str):
            vals = [qr.metrics.get(key, 0) for qr in valid]
            return round(sum(vals) / len(vals), 4) if vals else None

        result.agg_metrics = {
            "exp_id": EXP_ID, "pipeline": PIPELINE, "run_mode": RUN_MODE,
            "top_k": result.top_k, "n_queries": len(qrs),
            "n_valid": len(valid), "n_errors": result.total_errors,
            "pct_useful":              _rate("is_useful"),
            "avg_answer_relevance":    _mean("answer_relevance"),
            "avg_semantic_similarity": _mean("semantic_similarity"),
            "avg_faithfulness": None, "avg_context_precision": None, "avg_context_recall": None,
            "avg_hallucination_rate":  _mean("hallucination_rate"),
            "avg_insight_clarity":     _mean("insight_clarity"),
            "avg_latency_sec": round(
                sum(qr.latency_sec for qr in valid) / max(len(valid), 1), 3
            ),
            "avg_recall_at_k":        _mean("recall_at_k"),
            "avg_precision_at_k":     _mean("precision_at_k"),
            "avg_mrr":                _mean("mrr"),
            "avg_ndcg_at_k":          _mean("ndcg_at_k"),
            "avg_relevant_available": _mean("relevant_available"),
            "avg_relevant_retrieved": _mean("relevant_retrieved"),
            "avg_n_children":         _mean("n_children"),
            "avg_n_parents_added":    _mean("n_parents_added"),
            "pct_include_pass": _rate("include_pass"),
            "pct_exclude_pass": _rate("exclude_pass"),
            "pct_overall_pass": _rate("overall_pass"),
            "avg_attribution_coverage":    _mean("attribution_coverage"),
            "avg_citation_accuracy":       _mean("citation_accuracy"),
            "avg_unsupported_claim_rate":  _mean("unsupported_claim_rate"),
            "avg_total_claims":            _mean("total_claims"),
            "avg_claims_with_citation":    _mean("claims_with_citation"),
            "avg_correct_citations":       _mean("correct_citations"),
            "avg_spurious_citations":      _mean("spurious_citations"),
            "avg_correct_child_citations": _mean("correct_child_citations"),
            "avg_correct_parent_citations":_mean("correct_parent_citations"),
            "total_time_sec": result.total_time_sec,
        }


def run(
    queries:     list[dict],
    k_values:    list[int] | None = None,
    outputs_dir: str | Path = "outputs/experiments",
    run_mode:    str = RUN_MODE,
) -> list[ExperimentResult]:
    if k_values is None:
        k_values = EXP_DEFAULTS["top_k_values"]

    client  = RotatingGroqClient()
    results = []

    for k in k_values:
        logger.info("─── EXP_07 at K=%d | mode=%s ───", k, run_mode)
        generate_fn = make_generate_fn(client, k, run_mode=run_mode)
        result = run_experiment(
            exp_id=EXP_ID, pipeline=PIPELINE, top_k=k,
            queries=queries, generate_fn=generate_fn,
            outputs_dir=outputs_dir, log_every=10,
        )
        results.append(result)

    _compute_agg(results)
    for result in results:
        out_dir = Path(outputs_dir) / EXP_ID / f"k{result.top_k}"
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_results(result, out_dir)

    client.log_stats()
    return results
