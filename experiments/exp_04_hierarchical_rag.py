"""
experiments/exp_04_hierarchical_rag.py  (v2 — improved)
=========================================================
EXP_04_HIERARCHICAL_RAG — Parent-child context expansion via FAISS

CHANGES vs v1
--------------
TWO improvements applied. Query expansion is deliberately NOT applied
here — see rationale below.

IMPROVEMENT 1 — Cross-Encoder Reranking on child candidates
  Before parent expansion, FAISS fetches K * RECALL_MULTIPLIER child
  candidates. The cross-encoder re-scores and picks the best K children.
  Parent docs are then resolved from those K reranked children.

  WHY: Context Precision is 0.000 in EXP_02/03. The wrong child docs
  are ranked first, causing bad parent expansion too (wrong parents
  are added). Fixing child ranking fixes the cascade.

  WHY NOT query expansion here: Hierarchical retrieval's strength is
  the parent context it adds. Expanding queries would flood the candidate
  pool with docs from multiple parent trees, diluting the focused
  child→parent relationship that makes EXP_04 distinct from EXP_03.
  Keep expansion for flat retrieval (EXP_02/03/05/06).

IMPROVEMENT 2 — Grounding-Focused Prompt
  RAG_PROMPT replaced with GROUNDED_RAG_PROMPT for the generation step.
  The hierarchical context (children + parents) gives the model more
  material to ground in — the grounded prompt ensures it uses it.

Pipeline
--------
  1. FAISS retrieves K * RECALL_MULTIPLIER child candidates
  2. Cross-encoder reranks → keeps top K children
  3. Parent docs resolved from top K children (unchanged logic)
  4. Combined child + parent context → GROUNDED_RAG_PROMPT → LLM
  5. Metrics: child IDs only for retrieval metrics (unchanged)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from config.models import MODELS, EXP_DEFAULTS
from config.paths import PATHS
from src.embedding.embedder import get_embeddings_model
from src.embedding.faiss_store import load_faiss_index
from src.retrieval.hierarchical import HierarchicalRetriever
from src.retrieval.reranker import CrossEncoderReranker, RECALL_MULTIPLIER
from src.rag.prompts import format_docs, RAG_PROMPT
from src.rag.prompts import GROUNDED_RAG_PROMPT
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
from experiments.runner import (
    run_experiment,
    ExperimentResult,
    _save_results,
)

import pandas as pd

logger = logging.getLogger(__name__)

EXP_ID   = "EXP_04_HIERARCHICAL_RAG"
PIPELINE = "hierarchical"

# ---------------------------------------------------------------------------
# RUN MODE
# ---------------------------------------------------------------------------
# 'baseline'        → original v1 behaviour
# 'rerank'          → child reranking only
# 'improved_prompt' → grounded prompt only
# 'full'            → reranking + grounded prompt (recommended)

RUN_MODE = "full"

# ---------------------------------------------------------------------------
# FAISS + document list (cached)
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
        logger.info(
            "FAISS loaded. %d documents available for parent lookup.",
            len(_documents),
        )
    return _faiss_store, _documents


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
# Prompt builder
# ---------------------------------------------------------------------------

def _build_messages(
    context: str, question: str, use_grounded: bool
) -> list[dict]:
    prompt_template = GROUNDED_RAG_PROMPT if use_grounded else RAG_PROMPT
    prompt_value = prompt_template.format_messages(
        context=context, question=question,
    )
    return [
        {"role": "user" if msg.type == "human" else msg.type,
         "content": msg.content}
        for msg in prompt_value
    ]


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
    """
    Return a generate_fn for one K value.

    Reranking operates on CHILD candidates only — parents are resolved
    after reranking and are never subject to reranking themselves.
    Retrieval metrics use CHILD IDs only (parents are contextual, not ranked).
    """
    faiss_store, documents = _get_faiss_and_docs()
    id_to_summary  = _load_id_to_summary()   # loaded once, reused

    use_rerank          = run_mode in ("rerank", "full")
    use_grounded_prompt = run_mode in ("improved_prompt", "full")

    # Over-fetch children so reranker has a pool to choose from
    fetch_k = k * RECALL_MULTIPLIER if use_rerank else k

    # HierarchicalRetriever with over-fetch k for child retrieval
    retriever = HierarchicalRetriever(faiss_store, documents, k=fetch_k)
    reranker  = CrossEncoderReranker() if use_rerank else None

    logger.info(
        "EXP_04 generate_fn: k=%d fetch_k=%d mode=%s "
        "(rerank=%s grounded=%s)",
        k, fetch_k, run_mode, use_rerank, use_grounded_prompt,
    )

    def generate_fn(query: dict, _retrieved_docs: list, top_k: int) -> dict:
        question         = query.get("question", query.get("user_query", ""))
        ground_truth     = query.get("ground_truth", query.get("reference_answer", ""))
        expected_ids     = _parse_ids(query.get("expected_summary_ids", "[]"))
        must_include     = _parse_list(query.get("answer_must_include", "[]"))
        must_not_include = _parse_list(query.get("answer_must_not_include", "[]"))

        # ── Retrieval: fetch children + their parents ─────────────────────────
        # retrieve_with_scores returns children (dense_child) + parents (parent_expansion)
        all_docs = retriever.retrieve_with_scores(question, k=fetch_k)

        child_docs_raw  = [d for d in all_docs if d["retrieval_method"] == "dense_child"]
        parent_docs_raw = [d for d in all_docs if d["retrieval_method"] == "parent_expansion"]

        # ── Reranking: re-score child candidates, keep top K ─────────────────
        if use_rerank and reranker and child_docs_raw:
            # Convert to dict format expected by reranker (needs 'text' key)
            child_candidates = [
                {
                    "row_id":      d["row_id"],
                    "text":        d["page_content"],
                    "score":       d["score"],
                    "page_content": d["page_content"],
                    "metadata":    d,
                    "retrieval_method": d["retrieval_method"],
                }
                for d in child_docs_raw
            ]
            reranked_children = reranker.rerank(
                query=question,
                candidates=child_candidates,
                top_k=top_k,
                text_key="text",
            )
            # Convert back to original format, keeping top K children only
            top_child_ids = {d["row_id"] for d in reranked_children}
            child_docs    = [
                d for d in child_docs_raw if d["row_id"] in top_child_ids
            ][:top_k]
        else:
            child_docs = child_docs_raw[:top_k]

        # Parent docs stay as-is (resolved from top children by HierarchicalRetriever)
        # Re-filter parents to only those belonging to the kept children's parent_ids
        kept_parent_ids = {
            d.get("parent_id", "") or d.get("metadata", {}).get("parent_id", "")
            for d in child_docs
        }
        # If parent filtering is available, apply it; otherwise keep all
        if kept_parent_ids and any(kept_parent_ids):
            parent_docs = [
                d for d in parent_docs_raw
                if d["row_id"] in kept_parent_ids
            ]
        else:
            parent_docs = parent_docs_raw

        final_all_docs  = child_docs + parent_docs
        child_ids       = [d["row_id"] for d in child_docs]
        child_scores    = [
            d.get("rerank_score", d.get("score", 0.0)) for d in child_docs
        ]
        n_parents_added = len(parent_docs)

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
            "n_parents_added": n_parents_added,
        }

        # ── RAG generation ───────────────────────────────────────────────────
        # Build LangChain Documents from final combined doc list
        lc_docs = [
            Document(
                page_content=d["page_content"],
                metadata=d if not isinstance(d.get("metadata"), dict) else d["metadata"],
            )
            for d in final_all_docs
        ]
        context  = format_docs(lc_docs)
        messages = _build_messages(context, question, use_grounded_prompt)

        response = client.chat(
            messages=messages,
            model=MODELS["groq_rag"],
            temperature=EXP_DEFAULTS["temperature"],
            max_tokens=EXP_DEFAULTS["max_tokens"],
        )
        answer = response.choices[0].message.content.strip()

        # ── Generation metrics ───────────────────────────────────────────────
        halluc_check = check_hallucination(answer, must_include, must_not_include)

        metrics = {
            **retrieval_metrics,
            "answer_relevance":    compute_answer_relevance(question, answer),
            "semantic_similarity": compute_semantic_similarity(answer, ground_truth)
                                   if ground_truth else None,
            "hallucination_rate":  compute_hallucination_rate(
                                       answer,
                                       [{"text": d["page_content"]} for d in final_all_docs],
                                   ),
            "insight_clarity":     compute_insight_clarity(answer),
            "is_useful":           int(is_useful_answer(answer, question)),
            "answer_word_count":   len(answer.split()),
            "include_pass":        int(halluc_check["include_pass"]),
            "exclude_pass":        int(halluc_check["exclude_pass"]),
            "overall_pass":        int(halluc_check["overall_pass"]),
            "missing_terms":       json.dumps(halluc_check["missing_terms"]),
            "forbidden_terms":     json.dumps(halluc_check["forbidden_terms"]),
            "retrieved_ids":       json.dumps(child_ids),
            "similarity_scores":   json.dumps([round(s, 4) for s in child_scores]),
            "run_mode":            run_mode,
            "used_rerank":         int(use_rerank),
            "used_grounded_prompt": int(use_grounded_prompt),
        }

        return {
            "answer":         answer,
            "retrieved_docs": [
                {
                    "text":             d["page_content"],
                    "metadata":         d,
                    "evidence_id":      f"E{i+1}",
                    "retrieval_method": d["retrieval_method"],
                }
                for i, d in enumerate(final_all_docs)
            ],
            "metrics": metrics,
        }

    return generate_fn


# ---------------------------------------------------------------------------
# Aggregate metrics (unchanged from v1, + run_mode)
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
            "exp_id":   EXP_ID,
            "pipeline": PIPELINE,
            "run_mode": RUN_MODE,
            "top_k":    result.top_k,
            "n_queries": len(qrs),
            "n_valid":   len(valid),
            "n_errors":  result.total_errors,
            "pct_useful":              _rate("is_useful"),
            "avg_answer_relevance":    _mean("answer_relevance"),
            "avg_semantic_similarity": _mean("semantic_similarity"),
            "avg_faithfulness":        None,
            "avg_context_precision":   None,
            "avg_context_recall":      None,
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
            "total_time_sec":   result.total_time_sec,
        }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

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
        logger.info("─── EXP_04 at K=%d | mode=%s ───", k, run_mode)
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
        logger.info("[EXP_04 | k=%d | mode=%s] agg_metrics saved.",
                    result.top_k, run_mode)

    client.log_stats()
    return results
