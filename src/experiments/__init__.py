"""
src/experiments/
=================
Supporting modules for the 10 thesis experiments.

Modules
-------
groq_client.py   — RotatingGroqClient: cycles across 6 Groq API keys,
                   handles 429 rate limits automatically.
                   NOTE: uses deferred import of config.groq_keys inside
                   __init__() to avoid circular import via config/__init__.py

metrics.py       — Per-query metric functions shared across all experiments:
                   compute_answer_relevance, compute_semantic_similarity,
                   compute_hallucination_rate, compute_insight_clarity,
                   is_useful_answer, compute_retrieval_metrics

attribution.py   — Novelty 1: evidence ID assignment, citation parsing,
                   attribution coverage and accuracy metrics

difficulty.py    — Novelty 2: coverage/consistency scoring, difficulty
                   labelling (Easy/Medium/Hard), caution evaluation

result_tables.py — Builds all 5 result tables as DataFrames + exports to CSV/MD

Typical import pattern in experiment modules
--------------------------------------------
    from src.experiments.groq_client import RotatingGroqClient
    from src.experiments.metrics import (
        compute_answer_relevance,
        compute_semantic_similarity,
        compute_hallucination_rate,
        compute_insight_clarity,
        is_useful_answer,
    )
"""
