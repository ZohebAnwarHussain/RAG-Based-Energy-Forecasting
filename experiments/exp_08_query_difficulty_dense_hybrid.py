"""
experiments/exp_08_query_difficulty_dense_hybrid.py  (v2 — improved retrieval)
================================================================================
EXP_08_QUERY_DIFFICULTY_DENSE_HYBRID — Query Difficulty Prediction + Hybrid RAG

CHANGES vs v1
--------------
IMPROVEMENT 1 + 2 — Reranking + Expansion on the HYBRID generation side only.

The experiment has two distinct retrieval stages:
  Stage A — Dense FAISS → similarity scores → difficulty classification
  Stage B — Hybrid RRF  → top-K docs → LLM generation

WHAT CHANGES:
  Stage A (dense difficulty scorer): UNCHANGED.
    The dense scores are the classification signal itself. Adding expansion
    here would give a different signal than what EXP_09 uses (child scores),
    breaking the experiment's design intent. The scorer must stay pure FAISS.

  Stage B (hybrid generator): GETS reranking + expansion.
    Same improvements as EXP_06: HybridRetriever gains fetch_k,
    reranker applied after RRF fusion, expander optionally unions variants.

IMPROVEMENT 3 — Prompt: UNCHANGED.
  DIFFICULTY_SYSTEM_MESSAGE already includes rules 1–5 equivalent to
  GROUNDED_RAG_PROMPT. Replacing it would lose the difficulty-level
  instruction in rule 8, which is the novelty of this experiment.

RUN_MODE controls Stage B only:
  'baseline'      → original v1 behaviour
  'rerank'        → reranking on hybrid generation side
  'expand'        → query expansion on hybrid generation side
  'expand+rerank' → both (recommended)
  'full'          → same as expand+rerank
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from config.models import MODELS, EXP_DEFAULTS
from config.paths import PATHS
from src.embedding.embedder import get_embeddings_model
from src.embedding.faiss_store import load_faiss_index
from src.retrieval.dense import DenseRetriever
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

import pandas as pd

logger = logging.getLogger(__name__)

EXP_ID   = "EXP_08_QUERY_DIFFICULTY_DENSE_HYBRID"
PIPELINE = "difficulty_dense_hybrid"

RUN_MODE = "full"   # applies to Stage B (hybrid generation) only

# ---------------------------------------------------------------------------
# Difficulty-aware prompt (unchanged — do NOT replace with GROUNDED_RAG_PROMPT)
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
    "---\n"
    "{context}\n"
    "---\n\n"
    "Question: {question}\n\n"
    "Provide a factual, evidence-grounded answer appropriate to the "
    "difficulty level indicated above."
)


def _build_difficulty_messages(
    context: str, question: str, difficulty_prefix: str
) -> list[dict]:
    return [
        {"role": "system", "content": DIFFICULTY_SYSTEM_MESSAGE},
        {"role": "user",   "content": DIFFICULTY_HUMAN_TEMPLATE.format(
            difficulty_prefix=difficulty_prefix,
            context=context, question=question,
        )},
    ]


# ---------------------------------------------------------------------------
# HybridRetriever v2 — adds fetch_k parameter (Stage B only)
# ---------------------------------------------------------------------------

class HybridRetriever:
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
            {"row_id": rid, "text": self._row_id_to_doc[rid].page_content,
             "score": rrf_scores[rid], "metadata": self._row_id_to_doc[rid].metadata}
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


def _format_hybrid_context(retrieved: list[dict]) -> str:
    parts = []
    for doc in retrieved:
        src  = doc["row_id"]
        meta = doc.get("metadata", {})
        gran = meta.get("granularity", "")
        ds   = meta.get("dataset", "")
        parts.append(f"[{src}] ({ds}/{gran}):\n{doc['text']}")
    return "\n\n".join(parts)


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
    faiss_store, embeddings = _get_faiss_and_embeddings()
    id_to_summary  = _load_id_to_summary()   # loaded once, reused

    # Stage A: dense scorer — ALWAYS baseline, no improvements
    dense_retriever = DenseRetriever(faiss_store, k=k)

    # Stage B: hybrid generator — gets improvements
    use_rerank    = run_mode in ("rerank", "expand+rerank", "full")
    use_expansion = run_mode in ("expand", "expand+rerank", "full")
    fetch_k       = k * RECALL_MULTIPLIER if use_rerank else k

    hybrid_retriever = HybridRetriever(faiss_obj=faiss_store, embeddings=embeddings, top_k=k)
    reranker         = CrossEncoderReranker() if use_rerank else None
    expander         = QueryExpander(client)  if use_expansion else None

    logger.info(
        "EXP_08 generate_fn: k=%d fetch_k=%d mode=%s "
        "(stageA=baseline, stageB: rerank=%s expand=%s)",
        k, fetch_k, run_mode, use_rerank, use_expansion,
    )

    def generate_fn(query: dict, _docs: list, top_k: int) -> dict:
        question         = query.get("question", query.get("user_query", ""))
        ground_truth     = query.get("ground_truth", query.get("reference_answer", ""))
        expected_ids     = _parse_ids(query.get("expected_summary_ids", "[]"))
        must_include     = _parse_list(query.get("answer_must_include", "[]"))
        must_not_include = _parse_list(query.get("answer_must_not_include", "[]"))

        # ── Stage A: dense scoring for difficulty (UNCHANGED) ─────────────────
        dense_scored = dense_retriever.retrieve_with_scores(question, k=top_k)
        dense_scores = [d["score"] for d in dense_scored]

        classification    = classify_query(dense_scores)
        difficulty_prefix = build_difficulty_prompt_prefix(classification)

        # ── Stage B: hybrid generation with improvements ──────────────────────
        if use_expansion:
            all_candidates = expander.expand_and_retrieve(
                query=question,
                retrieve_fn=lambda q: hybrid_retriever.retrieve(q, fetch_k=fetch_k),
                top_k=top_k,
            )
        else:
            all_candidates = hybrid_retriever.retrieve(question, fetch_k=fetch_k)

        if use_rerank and reranker and all_candidates:
            hybrid_docs = reranker.rerank(
                query=question, candidates=all_candidates,
                top_k=top_k, text_key="text",
            )
        else:
            hybrid_docs = all_candidates[:top_k]

        retrieved_ids = [d["row_id"] for d in hybrid_docs]
        hybrid_scores = [d.get("rerank_score", d.get("score", 0.0)) for d in hybrid_docs]

        # Relevant chunk texts for content fallback
        retrieved_texts = [d.get("text", "") for d in hybrid_docs]
        relevant_texts  = [id_to_summary.get(eid, "") for eid in expected_ids]

        retrieval_metrics = compute_retrieval_metrics_with_content_fallback(
            retrieved_ids   = retrieved_ids,
            relevant_ids    = expected_ids,
            k               = top_k,
            retrieved_texts = retrieved_texts,
            relevant_texts  = relevant_texts,
        )

        # ── Generate with difficulty-aware prompt ─────────────────────────────
        context  = _format_hybrid_context(hybrid_docs)
        messages = _build_difficulty_messages(context, question, difficulty_prefix)
        response = client.chat(
            messages=messages, model=MODELS["groq_rag"],
            temperature=EXP_DEFAULTS["temperature"],
            max_tokens=EXP_DEFAULTS["max_tokens"],
        )
        answer = response.choices[0].message.content.strip()

        caution_result = evaluate_caution(answer, classification["difficulty_label"])
        halluc_check   = check_hallucination(answer, must_include, must_not_include)

        metrics = {
            **retrieval_metrics,
            "answer_relevance":    compute_answer_relevance(question, answer),
            "semantic_similarity": compute_semantic_similarity(answer, ground_truth)
                                   if ground_truth else None,
            "hallucination_rate":  compute_hallucination_rate(
                                       answer, [{"text": d["text"]} for d in hybrid_docs],
                                   ),
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
            "dense_scores_for_difficulty": json.dumps([round(s, 4) for s in dense_scores]),
            "retrieved_ids":       json.dumps(retrieved_ids),
            "similarity_scores":   json.dumps([round(s, 6) for s in hybrid_scores]),
            "run_mode":            run_mode,
            "used_rerank":         int(use_rerank),
            "used_expansion":      int(use_expansion),
        }

        return {
            "answer":         answer,
            "retrieved_docs": [
                {"text": d["text"], "metadata": d.get("metadata", {}),
                 "evidence_id": f"E{i+1}"}
                for i, d in enumerate(hybrid_docs)
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
        logger.info("─── EXP_08 at K=%d | mode=%s ───", k, run_mode)
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
