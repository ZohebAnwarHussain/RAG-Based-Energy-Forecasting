"""
experiments/
=============
One module per experiment (EXP_01 to EXP_10).
All share the runner.py execution engine.

Each experiment module exposes:
    run(queries, ..., outputs_dir) -> ExperimentResult

Experiments
-----------
exp_01_no_rag.py             — No-RAG LLM baseline (Group A)
exp_02_dense_rag.py          — Dense RAG at K=3,5,10 (Group A)
exp_03_hybrid_rag.py         — Hybrid RAG at K=3,5,10 (Group A)
exp_04_hierarchical_rag.py   — Hierarchical RAG at K=3,5,10 (Group A)
exp_05_dense_attribution.py  — Dense RAG + Evidence Attribution (Group B / Novelty 1)
exp_06_hybrid_attribution.py — Hybrid RAG + Evidence Attribution (Group B / Novelty 1)
exp_07_hier_attribution.py   — Hierarchical RAG + Evidence Attribution (Group B / Novelty 1)
exp_08_difficulty_dense.py   — Query Difficulty + Dense/Hybrid RAG (Group C / Novelty 2)
exp_09_difficulty_hier.py    — Query Difficulty + Hierarchical RAG (Group C / Novelty 2)
exp_10_final_comparison.py   — Final ranking across all methods (Group D)
"""
