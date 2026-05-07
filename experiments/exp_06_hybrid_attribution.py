"""
experiments/exp_06_hybrid_attribution.py  (v2 — improved retrieval)
=====================================================================
EXP_06_HYBRID_RAG_ATTRIBUTION — Hybrid RAG + Evidence-Linked Attribution

CHANGES vs v1
--------------
IMPROVEMENT 1 + 2 — Cross-Encoder Reranking + Query Expansion
  Same improvements as EXP_03 v2 applied to the hybrid retrieval stage.
  HybridRetriever.retrieve() gains fetch_k parameter.
  Attribution prompt is unchanged (same rationale as EXP_05).

RUN_MODE controls retrieval improvements only — prompt is fixed.
  'baseline'      → original v1 behaviour
  'rerank'        → cross-encoder reranking after RRF fusion
  'expand'        → query expansion before hybrid retrieval
  'expand+rerank' → both (recommended)
  'full'          → same as expand+rerank for this experiment
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from config.models import MODELS, EXP_DEFAULTS
from config.paths import PATHS
from src.embedding.embedder import get_embeddings_model
from src.embedding.faiss_store import load_faiss_index
from src.retrieval.reranker import CrossEncoderReranker, RECALL_MULTIPLIER
from src.retrieval.query_expander import QueryExpander
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
    assign_evidence_ids,
    build_attributed_context,
    compute_attribution_metrics,
)
from experiments.runner import (
    run_experiment,
    ExperimentResult,
    _save_results,
)

import pandas as pd

logger = logging.getLogger(__name__)

EXP_ID   = "EXP_06_HYBRID_RAG_ATTRIBUTION"
PIPELINE = "hybrid_attribution"

# ---------------------------------------------------------------------------
# RUN MODE — retrieval improvements only
# ---------------------------------------------------------------------------
RUN_MODE = "full"

# ---------------------------------------------------------------------------
# Attribution prompt (unchanged from v1)
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
    "the evidence document(s) it came from using square-bracket tags "
    "such as [E1], [E2], or [E1][E3]. Every sentence must end with at "
    "least one citation. Only cite evidence IDs that appear in the context."
)

ATTRIBUTION_HUMAN_TEMPLATE = (
    "Context (retrieved from the Energy Knowledge Base — cite using the [En] labels):\n"
    "---\n"
    "{context}\n"
    "---\n\n"
    "Question: {question}\n\n"
    "Provide a factual, evidence-grounded answer. "
    "Cite the evidence ID (e.g. [E1]) at the end of every sentence."
)


def _build_attribution_messages(context: str, question: str) -> list[dict]:
    return [
        {"role": "system", "content": ATTRIBUTION_SYSTEM_MESSAGE},
        {"role": "user",   "content": ATTRIBUTION_HUMAN_TEMPLATE.format(
            context=context, question=question)},
    ]


# ---------------------------------------------------------------------------
# HybridRetriever v2 — adds fetch_k parameter
# ---------------------------------------------------------------------------

class HybridRetriever:
    """FAISS dense + BM25 sparse fused via RRF. v2 adds fetch_k."""
    RRF_K = 60

    def __init__(self, faiss_obj, embeddings, top_k: int = 5):
        self.faiss_obj  = faiss_obj
        self.embeddings = embeddings
        self.top_k      = top_k

        self._row_id_to_doc = {
            doc.metadata["row_id"]: doc
            for doc in faiss_obj.docstore._dict.values()
        }
        self._row_ids   = list(self._row_id_to_doc.keys())
        self._doc_texts = [
            self._row_id_to_doc[rid].page_content for rid in self._row_ids
        ]
        tokenised = [t.lower().split() for t in self._doc_texts]
        self.bm25 = BM25Okapi(tokenised)
        logger.info("HybridRetriever ready: %d docs, top_k=%d",
                    len(self._row_ids), top_k)

    def retrieve(self, query: str, fetch_k: int | None = None) -> list[dict]:
        k    = fetch_k or self.top_k
        pool = min(len(self._row_ids), k * 4)

        query_vec  = self.embeddings.embed_query(query)
        dense_hits = self.faiss_obj.similarity_search_by_vector(query_vec, k=pool)
        dense_ranked = {doc.metadata["row_id"]: rank
                        for rank, doc in enumerate(dense_hits)}

        bm25_scores     = self.bm25.get_scores(query.lower().split())
        bm25_ranked_idx = np.argsort(bm25_scores)[::-1][:pool]
        bm25_ranked     = {self._row_ids[idx]: rank
                           for rank, idx in enumerate(bm25_ranked_idx)}

        all_ids    = set(dense_ranked) | set(bm25_ranked)
        rrf_scores = {
            rid: (
                1.0 / (self.RRF_K + dense_ranked.get(rid, pool)) +
                1.0 / (self.RRF_K + bm25_ranked.get(rid, pool))
            )
            for rid in all_ids
        }
        top_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:k]

        return [
            {
                "id": rid, "row_id": rid,
                "text":     self._row_id_to_doc[rid].page_content,
                "score":    rrf_scores[rid],
                "metadata": self._row_id_to_doc[rid].metadata,
            }
            for rid in top_ids
        ]


# ---------------------------------------------------------------------------
# FAISS cache
# ---------------------------------------------------------------------------

_faiss_store = None
_embeddings  = None


def _get_faiss_and_embeddings():
    global _faiss_store, _embeddings
    if _faiss_store is None:
        logger.info("Loading embedding model and FAISS index...")
        _embeddings  = get_embeddings_model()
        _faiss_store = load_faiss_index(PATHS["faiss_index"], _embeddings)
        logger.info("FAISS loaded.")
    return _faiss_store, _embeddings


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
# Helpers
# ---------------------------------------------------------------------------

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
# generate_fn factory
# ---------------------------------------------------------------------------

def make_generate_fn(
    client:   RotatingGroqClient,
    k:        int,
    run_mode: str = RUN_MODE,
) -> Any:
    faiss_obj, embeddings = _get_faiss_and_embeddings()
    id_to_summary  = _load_id_to_summary()   # loaded once, reused
    retriever = HybridRetriever(faiss_obj=faiss_obj, embeddings=embeddings, top_k=k)

    use_rerank    = run_mode in ("rerank", "expand+rerank", "full")
    use_expansion = run_mode in ("expand", "expand+rerank", "full")
    fetch_k       = k * RECALL_MULTIPLIER if use_rerank else k

    reranker = CrossEncoderReranker() if use_rerank else None
    expander = QueryExpander(client)  if use_expansion else None

    logger.info(
        "EXP_06 generate_fn: k=%d fetch_k=%d mode=%s (rerank=%s expand=%s)",
        k, fetch_k, run_mode, use_rerank, use_expansion,
    )

    def generate_fn(query: dict, _docs: list, top_k: int) -> dict:
        question         = query.get("question", query.get("user_query", ""))
        ground_truth     = query.get("ground_truth", query.get("reference_answer", ""))
        expected_ids     = _parse_ids(query.get("expected_summary_ids", "[]"))
        must_include     = _parse_list(query.get("answer_must_include", "[]"))
        must_not_include = _parse_list(query.get("answer_must_not_include", "[]"))

        # ── Retrieval ────────────────────────────────────────────────────────
        if use_expansion:
            all_candidates = expander.expand_and_retrieve(
                query=question,
                retrieve_fn=lambda q: retriever.retrieve(q, fetch_k=fetch_k),
                top_k=top_k,
            )
        else:
            all_candidates = retriever.retrieve(question, fetch_k=fetch_k)

        if use_rerank and reranker and all_candidates:
            scored_docs = reranker.rerank(
                query=question, candidates=all_candidates,
                top_k=top_k, text_key="text",
            )
        else:
            scored_docs = all_candidates[:top_k]

        retrieved_ids = [d["row_id"] for d in scored_docs]
        scores        = [d.get("rerank_score", d.get("score", 0.0)) for d in scored_docs]

        # Relevant chunk texts for content fallback
        retrieved_texts = [d.get("text", "") for d in scored_docs]
        relevant_texts  = [id_to_summary.get(eid, "") for eid in expected_ids]

        retrieval_metrics = compute_retrieval_metrics_with_content_fallback(
            retrieved_ids   = retrieved_ids,
            relevant_ids    = expected_ids,
            k               = top_k,
            retrieved_texts = retrieved_texts,
            relevant_texts  = relevant_texts,
        )

        # ── Build attributed context ─────────────────────────────────────────
        docs_for_attribution = [
            {
                "text": d["text"],
                "metadata": {
                    "source":      d["row_id"],
                    "granularity": d["metadata"].get("granularity", ""),
                    "dataset":     d["metadata"].get("dataset", ""),
                },
            }
            for d in scored_docs
        ]
        docs_with_ids      = assign_evidence_ids(docs_for_attribution)
        attributed_context = build_attributed_context(docs_with_ids)

        # ── Generation ───────────────────────────────────────────────────────
        messages = _build_attribution_messages(attributed_context, question)
        response = client.chat(
            messages=messages,
            model=MODELS["groq_rag"],
            temperature=EXP_DEFAULTS["temperature"],
            max_tokens=EXP_DEFAULTS["max_tokens"],
        )
        answer = response.choices[0].message.content.strip()

        attr_metrics = compute_attribution_metrics(answer, docs_with_ids)
        halluc_check = check_hallucination(answer, must_include, must_not_include)

        metrics = {
            **retrieval_metrics,
            "answer_relevance":    compute_answer_relevance(question, answer),
            "semantic_similarity": compute_semantic_similarity(answer, ground_truth)
                                   if ground_truth else None,
            "hallucination_rate":  compute_hallucination_rate(
                                       answer, [{"text": d["text"]} for d in docs_with_ids],
                                   ),
            "insight_clarity":     compute_insight_clarity(answer),
            "is_useful":           int(is_useful_answer(answer, question)),
            "answer_word_count":   len(answer.split()),
            "include_pass":        int(halluc_check["include_pass"]),
            "exclude_pass":        int(halluc_check["exclude_pass"]),
            "overall_pass":        int(halluc_check["overall_pass"]),
            "missing_terms":       json.dumps(halluc_check["missing_terms"]),
            "forbidden_terms":     json.dumps(halluc_check["forbidden_terms"]),
            "attribution_coverage":   attr_metrics["attribution_coverage"],
            "citation_accuracy":      attr_metrics["citation_accuracy"],
            "unsupported_claim_rate": attr_metrics["unsupported_claim_rate"],
            "total_claims":           attr_metrics["total_claims"],
            "claims_with_citation":   attr_metrics["claims_with_citation"],
            "correct_citations":      attr_metrics["correct_citations"],
            "spurious_citations":     attr_metrics["spurious_citations"],
            "cited_evidence_ids":     json.dumps(attr_metrics["cited_evidence_ids"]),
            "retrieved_ids":          json.dumps(retrieved_ids),
            "similarity_scores":      json.dumps([round(s, 6) for s in scores]),
            "run_mode":               run_mode,
            "used_rerank":            int(use_rerank),
            "used_expansion":         int(use_expansion),
        }

        return {
            "answer":         answer,
            "retrieved_docs": [
                {"text": d["text"], "metadata": d["metadata"],
                 "evidence_id": d["evidence_id"]}
                for d in docs_with_ids
            ],
            "metrics": metrics,
        }

    return generate_fn


# ---------------------------------------------------------------------------
# Aggregate metrics + public entry point (same schema as v1 + run_mode)
# ---------------------------------------------------------------------------

def _compute_agg(results: list[ExperimentResult]) -> None:
    for result in results:
        qrs   = result.query_results
        valid = [qr for qr in qrs if not qr.error]

        def _mean(key: str):
            vals = [qr.metrics.get(key) for qr in valid
                    if qr.metrics.get(key) is not None]
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
            "avg_faithfulness": None, "avg_context_precision": None,
            "avg_context_recall": None,
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
            "pct_include_pass": _rate("include_pass"),
            "pct_exclude_pass": _rate("exclude_pass"),
            "pct_overall_pass": _rate("overall_pass"),
            "avg_attribution_coverage":   _mean("attribution_coverage"),
            "avg_citation_accuracy":      _mean("citation_accuracy"),
            "avg_unsupported_claim_rate": _mean("unsupported_claim_rate"),
            "avg_total_claims":           _mean("total_claims"),
            "avg_claims_with_citation":   _mean("claims_with_citation"),
            "avg_correct_citations":      _mean("correct_citations"),
            "avg_spurious_citations":     _mean("spurious_citations"),
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
        logger.info("─── EXP_06 at K=%d | mode=%s ───", k, run_mode)
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
