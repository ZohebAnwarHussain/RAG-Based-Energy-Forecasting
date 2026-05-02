"""
experiments/exp_02_dense_rag.py
================================
EXP_02_DENSE_RAG — Dense Semantic Retrieval + RAG Generation

Objective
---------
Measure baseline RAG performance using FAISS dense retrieval +
Llama 3.3 70B generation. Runs at K = 3, 5, 10 (three separate passes).

Pipeline
--------
  1. Load FAISS index (built in Notebook 03)
  2. For each query: embed → FAISS similarity search → top-K docs
  3. Format retrieved docs as context → Llama 3.3 70B → answer
  4. Compute per-query metrics:
       Retrieval : Recall@K, Precision@K, MRR, nDCG
       Generation: Answer Relevance, Semantic Similarity, Faithfulness proxy
       Grounding : Hallucination check (must_include / must_not_include)
  5. Save results to outputs/experiments/EXP_02_DENSE_RAG/k{K}/

Config
------
  Embedding : sentence-transformers/all-MiniLM-L6-v2 (local)
  Vector DB : FAISS (IndexFlatIP, cosine similarity)
  LLM       : llama-3.3-70b-versatile (Groq, rotating client)
  K values  : 3, 5, 10
  Temperature: 0
  Max tokens : 500

Rate-limit handling
-------------------
  RotatingGroqClient cycles across up to 6 Groq API keys.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from config.models import MODELS, EXP_DEFAULTS
from config.paths import PATHS
from src.embedding.embedder import get_embeddings_model
from src.embedding.faiss_store import load_faiss_index
from src.retrieval.dense import DenseRetriever
from src.rag.prompts import format_docs, RAG_PROMPT
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
)
from experiments.runner import (
    run_experiment_multi_k,
    ExperimentResult,
    _save_results,
)

logger = logging.getLogger(__name__)

EXP_ID   = "EXP_02_DENSE_RAG"
PIPELINE = "dense"


# ---------------------------------------------------------------------------
# FAISS index loader (cached — loaded once, reused across all K values)
# ---------------------------------------------------------------------------

_faiss_store = None
_embeddings  = None


def _get_faiss_store():
    global _faiss_store, _embeddings
    if _faiss_store is None:
        logger.info("Loading embedding model and FAISS index...")
        _embeddings  = get_embeddings_model()
        _faiss_store = load_faiss_index(PATHS["faiss_index"], _embeddings)
        logger.info("FAISS index loaded.")
    return _faiss_store


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_messages(context: str, question: str) -> list[dict]:
    """Format the RAG prompt messages using the existing RAG_PROMPT template."""
    prompt_value = RAG_PROMPT.format_messages(
        context=context,
        question=question,
    )
    # Convert LangChain messages → plain dicts for RotatingGroqClient
    return [
        {"role": msg.type if msg.type != "human" else "user",
         "content": msg.content}
        for msg in prompt_value
    ]


# ---------------------------------------------------------------------------
# Generation function factory
# ---------------------------------------------------------------------------

def make_generate_fn(client: RotatingGroqClient, k: int) -> Any:
    """
    Return generate_fn for one K value.
    Closure captures the shared RotatingGroqClient and retriever.
    A fresh DenseRetriever is created per K so k is correctly set.
    """
    faiss_store = _get_faiss_store()
    retriever   = DenseRetriever(faiss_store, k=k)

    def generate_fn(query: dict, _retrieved_docs: list, top_k: int) -> dict:
        question     = query.get("question", query.get("user_query", ""))
        ground_truth = query.get("ground_truth", query.get("reference_answer", ""))
        expected_ids = _parse_ids(query.get("expected_summary_ids", "[]"))
        must_include    = _parse_list(query.get("answer_must_include", "[]"))
        must_not_include = _parse_list(query.get("answer_must_not_include", "[]"))

        # ── Retrieval ────────────────────────────────────────────────────
        scored_docs = retriever.retrieve_with_scores(question, k=top_k)
        docs        = retriever.retrieve(question, k=top_k)

        retrieved_ids    = [d["row_id"] for d in scored_docs]
        similarity_scores = [d["score"]  for d in scored_docs]

        # ── Retrieval metrics ────────────────────────────────────────────
        retrieval_metrics = {
            "recall_at_k":    round(_recall_at_k(retrieved_ids, expected_ids), 4),
            "precision_at_k": round(_precision_at_k(retrieved_ids, expected_ids), 4),
            "mrr":            round(_mrr(retrieved_ids, expected_ids), 4),
            "ndcg_at_k":      round(_ndcg(retrieved_ids, expected_ids), 4),
            "relevant_available": len(expected_ids),
            "relevant_retrieved": len(set(retrieved_ids) & set(expected_ids)),
        }

        # ── RAG generation ───────────────────────────────────────────────
        context  = format_docs(docs)
        messages = _build_messages(context, question)

        response = client.chat(
            messages=messages,
            model=MODELS["groq_rag"],
            temperature=EXP_DEFAULTS["temperature"],
            max_tokens=EXP_DEFAULTS["max_tokens"],
        )
        answer = response.choices[0].message.content.strip()

        # ── Generation metrics ───────────────────────────────────────────
        halluc_check = check_hallucination(answer, must_include, must_not_include)

        metrics = {
            # Retrieval
            **retrieval_metrics,
            # Generation
            "answer_relevance":    compute_answer_relevance(question, answer),
            "semantic_similarity": compute_semantic_similarity(answer, ground_truth)
                                   if ground_truth else None,
            "hallucination_rate":  compute_hallucination_rate(
                                       answer,
                                       [{"text": d["page_content"]} for d in scored_docs]
                                   ),
            "insight_clarity":     compute_insight_clarity(answer),
            "is_useful":           int(is_useful_answer(answer, question)),
            "answer_word_count":   len(answer.split()),
            # Hallucination keyword checks
            "include_pass":   int(halluc_check["include_pass"]),
            "exclude_pass":   int(halluc_check["exclude_pass"]),
            "overall_pass":   int(halluc_check["overall_pass"]),
            "missing_terms":  json.dumps(halluc_check["missing_terms"]),
            "forbidden_terms":json.dumps(halluc_check["forbidden_terms"]),
            # Retrieval detail
            "retrieved_ids":       json.dumps(retrieved_ids),
            "similarity_scores":   json.dumps([round(s, 4) for s in similarity_scores]),
        }

        return {
            "answer":         answer,
            "retrieved_docs": [{"text": d["page_content"],
                                 "metadata": d,
                                 "evidence_id": f"E{i+1}"}
                                for i, d in enumerate(scored_docs)],
            "metrics":        metrics,
        }

    return generate_fn


# ---------------------------------------------------------------------------
# Aggregate metrics helper
# ---------------------------------------------------------------------------

def _compute_agg(results: list[ExperimentResult]) -> None:
    """Populate agg_metrics for each K-level result."""
    for result in results:
        qrs   = result.query_results
        valid = [qr for qr in qrs if not qr.error]

        def _mean(key: str) -> float | None:
            vals = [qr.metrics.get(key) for qr in valid
                    if qr.metrics.get(key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        def _rate(key: str) -> float | None:
            vals = [qr.metrics.get(key, 0) for qr in valid if not qr.error]
            return round(sum(vals) / len(vals), 4) if vals else None

        result.agg_metrics = {
            "exp_id":                  EXP_ID,
            "pipeline":                PIPELINE,
            "top_k":                   result.top_k,
            "n_queries":               len(qrs),
            "n_valid":                 len(valid),
            "n_errors":                result.total_errors,
            # Table 1 — Overall RAG Performance
            "pct_useful":              _rate("is_useful"),
            "avg_answer_relevance":    _mean("answer_relevance"),
            "avg_semantic_similarity": _mean("semantic_similarity"),
            "avg_faithfulness":        None,   # filled by RAGAS
            "avg_context_precision":   None,   # filled by RAGAS
            "avg_context_recall":      None,   # filled by RAGAS
            "avg_hallucination_rate":  _mean("hallucination_rate"),
            "avg_insight_clarity":     _mean("insight_clarity"),
            "avg_latency_sec":         round(
                sum(qr.latency_sec for qr in valid) / max(len(valid), 1), 3
            ),
            # Table 2 — Retrieval Quality
            "avg_recall_at_k":         _mean("recall_at_k"),
            "avg_precision_at_k":      _mean("precision_at_k"),
            "avg_mrr":                 _mean("mrr"),
            "avg_ndcg_at_k":           _mean("ndcg_at_k"),
            "avg_relevant_available":  _mean("relevant_available"),
            "avg_relevant_retrieved":  _mean("relevant_retrieved"),
            # Hallucination keyword checks
            "pct_include_pass":        _rate("include_pass"),
            "pct_exclude_pass":        _rate("exclude_pass"),
            "pct_overall_pass":        _rate("overall_pass"),
            "total_time_sec":          result.total_time_sec,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_ids(raw: str) -> list[str]:
    """Parse expected_summary_ids JSON string → list of str."""
    try:
        return json.loads(raw) if isinstance(raw, str) else list(raw)
    except Exception:
        return []


def _parse_list(raw: str) -> list[str]:
    """Parse must_include / must_not_include JSON string → list of str."""
    try:
        return json.loads(raw) if isinstance(raw, str) else list(raw)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    queries:     list[dict],
    k_values:    list[int] | None = None,
    outputs_dir: str | Path = "outputs/experiments",
) -> list[ExperimentResult]:
    """
    Run EXP_02_DENSE_RAG at K = 3, 5, 10.

    Parameters
    ----------
    queries      : list of dicts — must have 'question'/'user_query',
                   'ground_truth'/'reference_answer',
                   'expected_summary_ids', 'answer_must_include',
                   'answer_must_not_include'
    k_values     : K values to run. Defaults to EXP_DEFAULTS["top_k_values"]
    outputs_dir  : root experiments output directory

    Returns
    -------
    List of ExperimentResult — one per K value.
    """
    if k_values is None:
        k_values = EXP_DEFAULTS["top_k_values"]

    client = RotatingGroqClient()

    results = []
    for k in k_values:
        logger.info("─── EXP_02 at K=%d ───", k)
        generate_fn = make_generate_fn(client, k)

        result = run_experiment(
            exp_id=EXP_ID,
            pipeline=PIPELINE,
            top_k=k,
            queries=queries,
            generate_fn=generate_fn,
            outputs_dir=outputs_dir,
            log_every=10,
        )
        results.append(result)

    # Compute and save aggregate metrics
    _compute_agg(results)
    for result in results:
        out_dir = Path(outputs_dir) / EXP_ID / f"k{result.top_k}"
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_results(result, out_dir)
        logger.info(
            "[EXP_02 | k=%d] agg_metrics: %s",
            result.top_k, result.agg_metrics,
        )

    client.log_stats()
    return results


# ---------------------------------------------------------------------------
# Required import (kept here to avoid circular at module level)
# ---------------------------------------------------------------------------
from experiments.runner import run_experiment  # noqa: E402
