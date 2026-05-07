"""
src/experiments/metrics.py
============================
Per-query metric computation functions shared across all experiments.

Functions
---------
compute_answer_relevance()                        — semantic sim between question and answer
compute_semantic_similarity()                     — cosine sim between answer and ground truth
compute_hallucination_rate()                      — fraction of answer sentences unsupported by context
compute_insight_clarity()                         — readability proxy (avg sentence length)
is_useful_answer()                                — boolean proxy for Table 1 "Correct/Useful Insights"
compute_retrieval_metrics()                       — Recall@K, Precision@K, MRR, nDCG (ID match only)
compute_retrieval_metrics_with_content_fallback() — same metrics with text-overlap fallback
                                                    for zone-mismatch / re-indexed chunk cases
"""

from __future__ import annotations

import math
import re
from difflib import SequenceMatcher
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Lazy-load sentence transformer (avoid import cost at module level)
# ---------------------------------------------------------------------------

_model = None

def _get_embedding_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        from config.models import MODELS
        _model = SentenceTransformer(MODELS["embedding"])
    return _model


def _embed(texts: list[str]) -> np.ndarray:
    model = _get_embedding_model()
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two unit-normalised vectors."""
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# 1. Answer Relevance
# ---------------------------------------------------------------------------

def compute_answer_relevance(question: str, answer: str) -> float:
    """
    Semantic similarity between the question and the answer.
    Proxy for how well the answer addresses the query.

    Returns float in [0, 1].
    """
    if not answer.strip():
        return 0.0
    embs = _embed([question, answer])
    return round(_cosine_sim(embs[0], embs[1]), 4)


# ---------------------------------------------------------------------------
# 2. Semantic Similarity (vs ground truth)
# ---------------------------------------------------------------------------

def compute_semantic_similarity(answer: str, ground_truth: str) -> Optional[float]:
    """
    Cosine similarity between the answer and the reference ground truth.
    Returns None if ground_truth is empty.
    """
    if not answer.strip() or not ground_truth.strip():
        return None
    embs = _embed([answer, ground_truth])
    return round(_cosine_sim(embs[0], embs[1]), 4)


# ---------------------------------------------------------------------------
# 3. Hallucination Rate
# ---------------------------------------------------------------------------

def compute_hallucination_rate(
    answer: str,
    context_docs: list[dict],
    sim_threshold: float = 0.45,
) -> float:
    """
    Fraction of answer sentences that have no supporting evidence in context.

    For No-RAG (context_docs=[]), every sentence is unsupported → rate = 1.0.
    For RAG experiments, each sentence is embedded and compared to each context doc.

    Returns float in [0, 1].
    """
    sentences = _split_sentences(answer)
    if not sentences:
        return 0.0

    if not context_docs:
        return 1.0

    context_texts = [
        d.get("text") or d.get("page_content", "") for d in context_docs
    ]
    context_texts = [t for t in context_texts if t.strip()]

    if not context_texts:
        return 1.0

    all_texts = sentences + context_texts
    all_embs  = _embed(all_texts)
    sent_embs = all_embs[:len(sentences)]
    ctx_embs  = all_embs[len(sentences):]

    unsupported = 0
    for s_emb in sent_embs:
        max_sim = max(_cosine_sim(s_emb, c_emb) for c_emb in ctx_embs)
        if max_sim < sim_threshold:
            unsupported += 1

    return round(unsupported / len(sentences), 4)


# ---------------------------------------------------------------------------
# 4. Insight Clarity
# ---------------------------------------------------------------------------

def compute_insight_clarity(answer: str) -> float:
    """
    Readability proxy: inverse of average sentence length (normalised).

    Shorter, punchy sentences → higher clarity score.
    Score in [0, 1]:  1 = very clear (avg ≤ 10 words), 0 = very dense (avg ≥ 40 words).
    """
    sentences = _split_sentences(answer)
    if not sentences:
        return 0.0

    avg_len = sum(len(s.split()) for s in sentences) / len(sentences)

    MIN_LEN, MAX_LEN = 10.0, 40.0
    clarity = 1.0 - max(0.0, min(1.0, (avg_len - MIN_LEN) / (MAX_LEN - MIN_LEN)))
    return round(clarity, 4)


# ---------------------------------------------------------------------------
# 5. Useful answer proxy
# ---------------------------------------------------------------------------

def is_useful_answer(
    answer: str,
    question: str,
    relevance_threshold: float = 0.40,
    min_words: int = 20,
) -> bool:
    """
    Automatic proxy for 'Correct / Useful Insights' in Table 1.

    An answer is considered useful if:
      1. It contains at least *min_words* words.
      2. Its answer_relevance score >= *relevance_threshold*.
    """
    if len(answer.split()) < min_words:
        return False
    relevance = compute_answer_relevance(question, answer)
    return relevance >= relevance_threshold


# ---------------------------------------------------------------------------
# 6. Retrieval metrics — ID matching only (original, preserved)
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(
    retrieved_ids: list[str],
    relevant_ids:  list[str],
    k: int,
) -> dict[str, float]:
    """
    Compute Recall@K, Precision@K, MRR, nDCG@K for a single query.
    Uses exact chunk ID string matching only.

    Parameters
    ----------
    retrieved_ids : ordered list of retrieved doc IDs (top-k)
    relevant_ids  : ground-truth relevant doc IDs
    k             : cutoff

    Returns
    -------
    dict: recall_at_k, precision_at_k, mrr, ndcg_at_k,
          relevant_available, relevant_retrieved
    """
    retrieved_k = retrieved_ids[:k]
    rel_set     = set(relevant_ids)

    hits = [1 if doc_id in rel_set else 0 for doc_id in retrieved_k]

    recall    = sum(hits) / max(len(rel_set), 1)
    precision = sum(hits) / k if k > 0 else 0.0

    mrr = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            mrr = 1.0 / rank
            break

    dcg  = sum(h / math.log2(r + 1) for r, h in enumerate(hits, start=1))
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(rel_set), k) + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {
        "recall_at_k":          round(recall, 4),
        "precision_at_k":       round(precision, 4),
        "mrr":                  round(mrr, 4),
        "ndcg_at_k":            round(ndcg, 4),
        "relevant_available":   len(rel_set),
        "relevant_retrieved":   sum(hits),
    }


# ---------------------------------------------------------------------------
# 7. Retrieval metrics — with content-overlap fallback (new)
# ---------------------------------------------------------------------------

def compute_retrieval_metrics_with_content_fallback(
    retrieved_ids:     list[str],
    relevant_ids:      list[str],
    k:                 int,
    retrieved_texts:   list[str] | None = None,
    relevant_texts:    list[str] | None = None,
    content_threshold: float = 0.55,
) -> dict[str, float]:
    """
    Compute Recall@K, Precision@K, MRR, nDCG@K with a content-overlap fallback.

    WHY THIS EXISTS
    ---------------
    Ten queries in the golden dataset ask about zones that either do not
    exist in the KB (Zone 20, Zone 21) or reference a different zone than
    the one in the assigned relevance label (e.g. query asks Zone 5 but
    label points to Zone 18). Pure ID matching always scores these as zero
    even when the retriever finds a genuinely useful chunk.

    This function uses exact ID matching first. If that fails AND text
    content is provided, it falls back to SequenceMatcher text-overlap
    (ratio >= content_threshold). A content hit is still counted as a
    hit for Recall/Precision/MRR/nDCG.

    Parameters
    ----------
    retrieved_ids      : ordered list of retrieved doc IDs (top-k)
    relevant_ids       : ground-truth relevant doc IDs from golden dataset
    k                  : cutoff
    retrieved_texts    : page_content of each retrieved doc (same order as retrieved_ids)
    relevant_texts     : summary text of each relevant chunk (same order as relevant_ids)
    content_threshold  : SequenceMatcher ratio for a content hit (default 0.55)

    Returns
    -------
    dict: recall_at_k, precision_at_k, mrr, ndcg_at_k,
          relevant_available, relevant_retrieved
    """
    retrieved_k = retrieved_ids[:k]
    rel_set     = set(relevant_ids)

    def _is_hit(doc_id: str, idx: int) -> bool:
        # Primary: exact ID match
        if doc_id in rel_set:
            return True
        # Fallback: text content overlap
        if (
            retrieved_texts is not None
            and relevant_texts is not None
            and idx < len(retrieved_texts)
            and retrieved_texts[idx].strip()
        ):
            ret_text = retrieved_texts[idx].lower()
            for rel_text in relevant_texts:
                if not rel_text.strip():
                    continue
                ratio = SequenceMatcher(
                    None, ret_text, rel_text.lower()
                ).ratio()
                if ratio >= content_threshold:
                    return True
        return False

    hits = [
        1 if _is_hit(doc_id, i) else 0
        for i, doc_id in enumerate(retrieved_k)
    ]

    recall    = sum(hits) / max(len(rel_set), 1)
    precision = sum(hits) / k if k > 0 else 0.0

    mrr = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            mrr = 1.0 / rank
            break

    dcg  = sum(h / math.log2(r + 1) for r, h in enumerate(hits, start=1))
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(rel_set), k) + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {
        "recall_at_k":          round(recall, 4),
        "precision_at_k":       round(precision, 4),
        "mrr":                  round(mrr, 4),
        "ndcg_at_k":            round(ndcg, 4),
        "relevant_available":   len(rel_set),
        "relevant_retrieved":   sum(hits),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences; filter very short fragments."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip().split()) >= 3]
