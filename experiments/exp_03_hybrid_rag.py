"""
EXP_03 — Hybrid RAG (Dense FAISS + BM25 sparse, RRF fusion)

Fixes applied vs original:
  1. Embedder() dropped — use get_embeddings_model() directly (HuggingFaceEmbeddings)
  2. embeddings.embed_query(str) used instead of non-existent embedder.embed([str])
  3. groq.chat() returns ChatCompletion — extract via .choices[0].message.content
  4. Full per-query metrics computed (matching EXP_02 pattern)
  5. _compute_agg() added and called — populates result.agg_metrics
  6. result.top_k used throughout (ExperimentResult has no .k attribute)
  7. row_id key added to retrieved dict for metric consistency
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from config.models import MODELS, EXP_DEFAULTS
from config.paths import PATHS
from experiments.runner import run_experiment, ExperimentResult, _save_results
from src.embedding.embedder import get_embeddings_model
from src.embedding.faiss_store import load_faiss_index
from src.experiments.groq_client import RotatingGroqClient
from src.evaluation.retrieval_metrics import (
    _recall_at_k, _precision_at_k, _mrr, _ndcg,
)
from src.evaluation.hallucination import check_hallucination
from src.experiments.metrics import (
        compute_answer_relevance,
        compute_semantic_similarity,
        compute_hallucination_rate,
        compute_insight_clarity,
        is_useful_answer,
        compute_retrieval_metrics_with_content_fallback,   # content-fallback fix
    )
from src.rag.prompts import RAG_PROMPT, format_docs

logger = logging.getLogger(__name__)

EXP_ID    = "EXP_03_HYBRID_RAG"
PIPELINE  = "hybrid_rag"
FAISS_DIR = PATHS["faiss_index"]


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
# Hybrid Retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """FAISS dense + BM25 sparse fused via Reciprocal Rank Fusion."""
    RRF_K = 60

    def __init__(self, faiss_obj, embeddings, top_k: int = 5):
        self.faiss_obj  = faiss_obj
        self.embeddings = embeddings   # HuggingFaceEmbeddings instance
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

    def retrieve(self, query: str) -> list[dict]:
        pool = min(len(self._row_ids), self.top_k * 4)

        # Dense — embed_query returns a single vector (list[float])
        query_vec  = self.embeddings.embed_query(query)
        dense_hits = self.faiss_obj.similarity_search_by_vector(query_vec, k=pool)
        dense_ranked = {
            doc.metadata["row_id"]: rank
            for rank, doc in enumerate(dense_hits)
        }

        # Sparse
        bm25_scores     = self.bm25.get_scores(query.lower().split())
        bm25_ranked_idx = np.argsort(bm25_scores)[::-1][:pool]
        bm25_ranked     = {
            self._row_ids[idx]: rank
            for rank, idx in enumerate(bm25_ranked_idx)
        }

        # RRF
        all_ids    = set(dense_ranked) | set(bm25_ranked)
        rrf_scores = {
            rid: (
                1.0 / (self.RRF_K + dense_ranked.get(rid, pool)) +
                1.0 / (self.RRF_K + bm25_ranked.get(rid, pool))
            )
            for rid in all_ids
        }
        top_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:self.top_k]

        return [
            {
                "id":       rid,
                "row_id":   rid,
                "text":     self._row_id_to_doc[rid].page_content,
                "score":    rrf_scores[rid],
                "metadata": self._row_id_to_doc[rid].metadata,
            }
            for rid in top_ids
        ]

    def retrieve_as_documents(self, query: str) -> list[Document]:
        return [
            Document(page_content=d["text"], metadata=d["metadata"])
            for d in self.retrieve(query)
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_messages(context: str, question: str) -> list[dict]:
    prompt_value = RAG_PROMPT.format_messages(context=context, question=question)
    return [
        {"role": "user" if msg.type == "human" else msg.type, "content": msg.content}
        for msg in prompt_value
    ]

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

def _make_generate_fn(groq: RotatingGroqClient, retriever: HybridRetriever):
    id_to_summary = _load_id_to_summary()   # loaded once, reused

    def generate_fn(query_row: dict, _docs: list[dict], top_k: int) -> dict:
        question         = query_row.get("question", query_row.get("user_query", ""))
        ground_truth     = query_row.get("ground_truth", query_row.get("reference_answer", ""))
        expected_ids     = _parse_ids(query_row.get("expected_summary_ids", "[]"))
        must_include     = _parse_list(query_row.get("answer_must_include", "[]"))
        must_not_include = _parse_list(query_row.get("answer_must_not_include", "[]"))

        # Retrieval
        retrieved     = retriever.retrieve(question)
        retrieved_ids = [d["row_id"] for d in retrieved]
        scores        = [d["score"]  for d in retrieved]

        # Relevant chunk texts for content fallback
        retrieved_texts = [d.get("text", "") for d in retrieved]
        relevant_texts  = [id_to_summary.get(eid, "") for eid in expected_ids]

        retrieval_metrics = compute_retrieval_metrics_with_content_fallback(
            retrieved_ids   = retrieved_ids,
            relevant_ids    = expected_ids,
            k               = top_k,
            retrieved_texts = retrieved_texts,
            relevant_texts  = relevant_texts,
        )

        # Generation
        context  = format_docs(retriever.retrieve_as_documents(question))
        messages = _build_messages(context, question)

        response = groq.chat(
            messages=messages,
            model=MODELS["groq_rag"],
            temperature=EXP_DEFAULTS["temperature"],
            max_tokens=EXP_DEFAULTS["max_tokens"],
        )
        # groq.chat() returns groq.types.chat.ChatCompletion
        answer = response.choices[0].message.content.strip()

        halluc_check = check_hallucination(answer, must_include, must_not_include)

        metrics = {
            **retrieval_metrics,
            "answer_relevance":    compute_answer_relevance(question, answer),
            "semantic_similarity": compute_semantic_similarity(answer, ground_truth)
                                   if ground_truth else None,
            "hallucination_rate":  compute_hallucination_rate(
                                       answer, [{"text": d["text"]} for d in retrieved]
                                   ),
            "insight_clarity":     compute_insight_clarity(answer),
            "is_useful":           int(is_useful_answer(answer, question)),
            "answer_word_count":   len(answer.split()),
            "include_pass":        int(halluc_check["include_pass"]),
            "exclude_pass":        int(halluc_check["exclude_pass"]),
            "overall_pass":        int(halluc_check["overall_pass"]),
            "missing_terms":       json.dumps(halluc_check["missing_terms"]),
            "forbidden_terms":     json.dumps(halluc_check["forbidden_terms"]),
            "retrieved_ids":       json.dumps(retrieved_ids),
            "similarity_scores":   json.dumps([round(s, 6) for s in scores]),
        }

        return {
            "answer":         answer,
            "retrieved_docs": [
                {"text": d["text"], "metadata": d["metadata"], "evidence_id": f"E{i+1}"}
                for i, d in enumerate(retrieved)
            ],
            "metrics": metrics,
        }

    return generate_fn


# ---------------------------------------------------------------------------
# Aggregate metrics
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
            "pct_include_pass": _rate("include_pass"),
            "pct_exclude_pass": _rate("exclude_pass"),
            "pct_overall_pass": _rate("overall_pass"),
            "total_time_sec":   result.total_time_sec,
        }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_exp_03(
    golden_df:   pd.DataFrame,
    k_values:    list[int] | None = None,
    outputs_dir: Path | None = None,
) -> list[ExperimentResult]:
    if k_values is None:
        k_values = [3, 5, 10]
    if outputs_dir is None:
        outputs_dir = Path("outputs") / "experiments"

    groq       = RotatingGroqClient()
    embeddings = get_embeddings_model()   # single load, shared across all K

    logger.info("Loading FAISS index...")
    faiss_obj = load_faiss_index(FAISS_DIR, embeddings)
    logger.info("FAISS loaded. %d docs", len(faiss_obj.docstore._dict))

    queries = golden_df.to_dict(orient="records")

    results: list[ExperimentResult] = []
    for k in k_values:
        logger.info("─── EXP_03 at K=%d ───", k)
        retriever   = HybridRetriever(faiss_obj=faiss_obj, embeddings=embeddings, top_k=k)
        generate_fn = _make_generate_fn(groq, retriever)

        result = run_experiment(
            exp_id      = EXP_ID,
            pipeline    = PIPELINE,
            top_k       = k,
            queries     = queries,
            generate_fn = generate_fn,
            outputs_dir = outputs_dir,
            log_every   = 10,
        )
        results.append(result)

    _compute_agg(results)
    for result in results:
        out_dir = Path(outputs_dir) / EXP_ID / f"k{result.top_k}"
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_results(result, out_dir)
        logger.info("[EXP_03 | k=%d] agg_metrics: %s", result.top_k, result.agg_metrics)

    groq.log_stats()
    return results