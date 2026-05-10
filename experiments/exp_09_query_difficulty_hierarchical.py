"""
experiments/exp_09_query_difficulty_hierarchical.py  (v2 — improved retrieval)
================================================================================
EXP_09_QUERY_DIFFICULTY_HIERARCHICAL — Query Difficulty Prediction + Hierarchical RAG

CHANGES vs v1
--------------
IMPROVEMENT 1 — Cross-Encoder Reranking on child candidates
  Same as EXP_04/07 v2: reranker applied to child pool (fetch_k = K*4)
  before parent expansion. Difficulty classification still uses the
  reranked child scores (more accurate signal after reranking).

WHY query expansion is NOT applied here (same rationale as EXP_04/07):
  Hierarchical retrieval's parent expansion is the differentiating factor.
  Expanding queries dilutes the parent-child relationship.

WHY the dense scorer is NOT boosted (same rationale as EXP_08):
  Difficulty classification signal = FAISS cosine scores. Must stay pure.

IMPROVEMENT 3 — Prompt: UNCHANGED.
  DIFFICULTY_SYSTEM_MESSAGE already grounded + has difficulty rule 8.

RUN_MODE:
  'baseline'  → original v1
  'rerank'    → child reranking (= 'full' for this experiment)
  'full'      → reranking only
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from config.models import MODELS, EXP_DEFAULTS
from config.paths import PATHS
from src.embedding.embedder import get_embeddings_model
from src.embedding.faiss_store import load_faiss_index
from src.retrieval.hierarchical import HierarchicalRetriever
from src.retrieval.reranker import CrossEncoderReranker, RECALL_MULTIPLIER
from src.rag.prompts import format_docs
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
from src.experiments.difficulty import (
    classify_query,
    build_difficulty_prompt_prefix,
    evaluate_caution,
)
from experiments.runner import (
    run_experiment,
    ExperimentResult,
    _save_results,
)

from langchain_core.documents import Document
import pandas as pd

logger = logging.getLogger(__name__)

EXP_ID   = "EXP_09_QUERY_DIFFICULTY_HIERARCHICAL"
PIPELINE = "difficulty_hierarchical"

RUN_MODE = "full"   # 'baseline' | 'rerank' | 'full'

# ---------------------------------------------------------------------------
# Difficulty-aware prompt (unchanged)
# ---------------------------------------------------------------------------

DIFFICULTY_SYSTEM_MESSAGE = (
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
    "8. IMPORTANT: The difficulty label at the top of the query indicates "
    "how much historical evidence is available. Adjust your confidence "
    "accordingly — for HARD queries, include explicit uncertainty language."
)

DIFFICULTY_HUMAN_TEMPLATE = (
    "{difficulty_prefix}"
    "Context (retrieved from the Energy Knowledge Base):\n"
    "Child summaries (daily detail):\n"
    "---\n"
    "{child_context}\n"
    "---\n\n"
    "Parent summaries (weekly/monthly broader context):\n"
    "---\n"
    "{parent_context}\n"
    "---\n\n"
    "Question: {question}\n\n"
    "Provide a factual, evidence-grounded answer appropriate to the "
    "difficulty level indicated above."
)


def _build_difficulty_messages(
    child_context: str, parent_context: str,
    question: str, difficulty_prefix: str,
) -> list[dict]:
    return [
        {"role": "system", "content": DIFFICULTY_SYSTEM_MESSAGE},
        {"role": "user",   "content": DIFFICULTY_HUMAN_TEMPLATE.format(
            difficulty_prefix=difficulty_prefix,
            child_context=child_context,
            parent_context=parent_context,
            question=question,
        )},
    ]


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


def _format_docs_plain(docs: list[dict]) -> str:
    parts = []
    for doc in docs:
        src  = doc.get("row_id", "unknown")
        gran = doc.get("granularity", "")
        ds   = doc.get("dataset", "")
        text = doc.get("page_content", "")
        parts.append(f"[{src}] ({ds}/{gran}):\n{text}")
    return "\n\n".join(parts) if parts else "None available."


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

    logger.info("EXP_09 generate_fn: k=%d fetch_k=%d mode=%s rerank=%s",
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

        # ── Rerank children ──────────────────────────────────────────────────
        if use_rerank and reranker and child_docs_raw:
            child_candidates = [
                {"row_id": d["row_id"], "text": d["page_content"],
                 "score": d["score"], "page_content": d["page_content"],
                 "metadata": d, "retrieval_method": d["retrieval_method"]}
                for d in child_docs_raw
            ]
            reranked     = reranker.rerank(
                query=question, candidates=child_candidates,
                top_k=top_k, text_key="text",
            )
            kept_ids    = {d["row_id"] for d in reranked}
            child_docs  = [d for d in child_docs_raw if d["row_id"] in kept_ids][:top_k]
        else:
            child_docs = child_docs_raw[:top_k]

        # Keep all parents from retrieve_with_scores — both exact and semantic fallback.
        # Do NOT re-filter by child parent_id field: semantic fallback parents have
        # different row_ids from the children's parent_id field by design.
        parent_docs = parent_docs_raw

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

        # ── Difficulty classification using RERANKED child scores ────────────
        # Post-reranking, use rerank_score as the signal if available,
        # otherwise fall back to original FAISS scores
        scores_for_classification = [
            d.get("rerank_score", d.get("score", 0.0)) for d in child_docs
        ]
        classification    = classify_query(scores_for_classification)
        difficulty_prefix = build_difficulty_prompt_prefix(classification)

        # Additional hierarchical signal: parent diversity
        parent_ids_of_children = [
            d.get("metadata", {}).get("parent_id", "") if isinstance(d, dict) else ""
            for d in child_docs
        ]
        parent_id_counts = Counter(p for p in parent_ids_of_children if p)
        n_unique_parents = len(parent_id_counts)

        # ── Build context ────────────────────────────────────────────────────
        child_context  = _format_docs_plain(child_docs)
        parent_context = _format_docs_plain(parent_docs)

        # ── Generation ───────────────────────────────────────────────────────
        messages = _build_difficulty_messages(
            child_context, parent_context, question, difficulty_prefix
        )
        response = client.chat(
            messages=messages, model=MODELS["groq_rag"],
            temperature=EXP_DEFAULTS["temperature"],
            max_tokens=EXP_DEFAULTS["max_tokens"],
        )
        answer = response.choices[0].message.content.strip()

        caution_result = evaluate_caution(answer, classification["difficulty_label"])
        halluc_check   = check_hallucination(answer, must_include, must_not_include)
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
            "coverage_score":      classification["coverage_score"],
            "consistency_score":   classification["consistency_score"],
            "difficulty_score":    classification["difficulty_score"],
            "difficulty_label":    classification["difficulty_label"],
            "requires_caution":    int(caution_result["requires_caution"]),
            "caution_detected":    int(caution_result["caution_detected"]),
            "cautious_pass":       int(caution_result["cautious_pass"]),
            "caution_phrases":     json.dumps(caution_result["matched_phrases"]),
            "n_unique_parent_ids": n_unique_parents,
            "child_scores_used":   json.dumps([round(s, 4) for s in scores_for_classification]),
            "retrieved_ids":       json.dumps(child_ids),
            "similarity_scores":   json.dumps([round(s, 4) for s in child_scores]),
            "run_mode":            run_mode,
            "used_rerank":         int(use_rerank),
        }

        return {
            "answer":         answer,
            "retrieved_docs": [
                {"text": d["page_content"], "evidence_id": f"E{i+1}",
                 "retrieval_method": d["retrieval_method"]}
                for i, d in enumerate(child_docs + parent_docs)
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

        hard_queries     = [qr for qr in valid if qr.metrics.get("difficulty_label") == "Hard"]
        cautious_correct = sum(1 for qr in hard_queries if qr.metrics.get("cautious_pass", 0))

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
            "avg_coverage_score":    _mean("coverage_score"),
            "avg_consistency_score": _mean("consistency_score"),
            "avg_difficulty_score":  _mean("difficulty_score"),
            "n_easy":   sum(1 for qr in valid if qr.metrics.get("difficulty_label") == "Easy"),
            "n_medium": sum(1 for qr in valid if qr.metrics.get("difficulty_label") == "Medium"),
            "n_hard":   len(hard_queries),
            "cautious_response_accuracy": round(
                cautious_correct / len(hard_queries), 4
            ) if hard_queries else None,
            "avg_n_unique_parent_ids": _mean("n_unique_parent_ids"),
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
        logger.info("─── EXP_09 at K=%d | mode=%s ───", k, run_mode)
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
