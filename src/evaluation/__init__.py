"""Evaluation pipeline (Phase 6).

Three evaluation components:

    Retrieval Metrics
        Recall@k, Precision@k, MRR, nDCG — measure whether the correct
        KB chunks were retrieved for each golden query.

    RAGAS Metrics
        Faithfulness, answer relevancy, context precision, context recall —
        measure the quality of RAG-generated answers against Gemini 2.5
        Flash reference answers. Uses Llama 3.3 70B as the LLM judge.

    Hallucination Checks
        Keyword-based guards using answer_must_include and
        answer_must_not_include from the golden dataset. Lightweight
        complement to RAGAS faithfulness.

Usage:
    from src.evaluation import (
        compute_retrieval_metrics,
        run_ragas_evaluation,
        check_hallucination,
    )
"""

from src.evaluation.hallucination import (
    check_hallucination,
    compute_hallucination_rate,
)
from src.evaluation.ragas_metrics import run_ragas_evaluation
from src.evaluation.retrieval_metrics import compute_retrieval_metrics

__all__ = [
    "compute_retrieval_metrics",
    "run_ragas_evaluation",
    "check_hallucination",
    "compute_hallucination_rate",
]
