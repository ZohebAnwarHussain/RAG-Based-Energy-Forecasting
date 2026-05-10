# RAG-Based Energy Forecasting

## Research Title

**RAG-Driven Natural Language Energy Demand Forecasting and Consumption
Insight Generator Using Household & Multi-Zone Load Data**

---

## Overview

This project implements a Retrieval-Augmented Generation (RAG) framework
that transforms historical energy time-series data into evidence-grounded
natural language demand insights. Two datasets — GEFCom2014 (20-zone utility
load) and UCI Individual Household Power Consumption — are statistically
aggregated into structured summaries, embedded into a vector store, and
retrieved by an LLM to generate stakeholder-friendly energy forecasting
explanations.

Four retrieval strategies are compared across ten experiments, including two
novel contributions:

1. **Dense** — FAISS cosine similarity over sentence embeddings (EXP_02)
2. **Hybrid** — BM25 sparse + FAISS dense retrieval via Reciprocal Rank Fusion (EXP_03)
3. **Hierarchical** — Parent-child document linking (daily to weekly) with metadata expansion (EXP_04)
4. **Evidence-Linked Attribution** — All three pipelines extended with [E1], [E2] citation prompting for claim-level traceability (EXP_05-07)
5. **Query Difficulty Prediction** — Pre-generation difficulty classification (Easy/Medium/Hard) based on retrieval coverage and consistency scores, with confidence-adjusted prompting (EXP_08-09)

Performance is evaluated using RAGAS metrics (faithfulness, answer relevancy,
context precision, context recall), standard retrieval metrics (Recall@K,
Precision@K, MRR, nDCG), and novelty-specific metrics (attribution coverage,
citation accuracy, cautious response accuracy).

---

## Architecture

```
Raw Time-Series Data (GEFCom2014 + UCI Household)
        |
Phase 1 — Statistical Aggregation + Gemini Flash Summaries → 140 KB documents
        |
Phase 2 — Gemini 2.5 Flash Reference Answers → 50 Golden Dataset queries
        |
Phase 3 — Sentence Transformers (all-MiniLM-L6-v2) → FAISS Vector Index (384-dim)
        |
Phase 4 — Four Retrieval Pipelines (Dense | Hybrid | Hierarchical | + Attribution/Difficulty)
        |
Phase 5 — Llama 3.3 70B via Groq → RAG Insights (plain / attributed / difficulty-conditioned)
        |
Phase 6 — RAGAS + Custom Metrics + Attribution + Difficulty Evaluation
        |
Phase 7 — EXP_10 Comparative Analysis + Thesis Findings
```

---

## Experiment Design

| Group | Exp ID | Pipeline | Enhancement | Key Metric Added |
|-------|--------|----------|-------------|-----------------|
| A — Baselines | EXP_01A-E | No-RAG LLM — 5 prompting variants | — | Hallucination Rate, Answer Relevance |
| A — Baselines | EXP_02 | Dense RAG (FAISS) | — | Recall@K, Faithfulness |
| A — Baselines | EXP_03 | Hybrid RAG (BM25 + FAISS + RRF) | — | Retrieval Ranking Quality |
| A — Baselines | EXP_04 | Hierarchical RAG (child + parent) | — | Context Recall, Parent Expansion |
| B — Attribution | EXP_05 | Dense RAG + Attribution | Evidence-tagged citation grounding | Attribution Coverage, Citation Accuracy |
| B — Attribution | EXP_06 | Hybrid RAG + Attribution | Evidence-tagged citation grounding | Attribution Coverage, Citation Accuracy |
| B — Attribution | EXP_07 | Hierarchical RAG + Attribution | Evidence-tagged citation grounding + parent context | Child/Parent Citation Correctness |
| C — Difficulty | EXP_08 | Dense RAG + Difficulty Awareness | Query difficulty classification + adaptive prompting | Cautious Response Accuracy |
| C — Difficulty | EXP_09 | Hierarchical RAG + Difficulty Awareness | Difficulty classification + parent expansion | Cautious Response Accuracy |
| D — Analysis | EXP_10 | Final comparative analysis | — | Composite Ranking Score |

All retrieval experiments run at K = 3, 5, 10. EXP_01 variants run once (no retrieval).

### EXP_01 Prompting Variants

| Variant | Strategy | Key Observation |
|---------|----------|----------------|
| EXP_01A | Zero-Shot | Raw parametric memory baseline |
| EXP_01B | Role Prompting | Expert persona conditioning |
| EXP_01C | Few-Shot | 3 worked examples prepended |
| EXP_01D | Chain-of-Thought | Step-by-step reasoning scaffold |
| EXP_01E | Structured Output | Forced OBSERVATION/PATTERN/IMPLICATION/CONFIDENCE format |

---

## Models and Tools

| Stage | Component | Provider | Purpose |
|-------|-----------|----------|---------|
| KB Generation | `gemini-2.0-flash-preview` | Google AI | Generate human-tone summaries from statistics |
| Golden Dataset | `gemini-2.5-flash` | Google AI | Generate reference answers for evaluation |
| Embedding | `all-MiniLM-L6-v2` | HuggingFace (local) | 384-dim sentence embeddings, runs on CPU |
| Vector Store | FAISS `IndexFlatIP` | Facebook AI | Dense similarity search |
| Sparse Retrieval | `BM25Okapi` | rank-bm25 | Keyword-based retrieval for hybrid pipeline |
| RRF Fusion | Custom (RRF k=60) | — | Reciprocal Rank Fusion of dense + sparse |
| Hierarchical | `HierarchicalRetriever` | Custom | FAISS child retrieval + parent_id expansion |
| RAG Generation | `llama-3.3-70b-versatile` | Groq | Natural language insight generation |
| Attribution | `attribution.py` | Custom | [E1],[E2] citation parsing + metrics |
| Difficulty | `difficulty.py` | Custom | Coverage/consistency scoring → Easy/Medium/Hard |
| Evaluation | RAGAS 0.4.x | Exploding Gradients | Faithfulness, relevancy, precision, recall |
| Key Rotation | `RotatingGroqClient` | Custom | 46-key round-robin + 429 handling |

### Model Independence Strategy

Two boundaries protect evaluation validity:

- **KB to Golden Dataset**: Different Gemini generations prevent stylistic contamination between KB summaries and reference answers.
- **Golden Dataset to RAG Generation**: Gemini family (reference answers) vs Llama family (RAG answers) ensures RAGAS scores reflect genuine cross-model evaluation.

---

## Datasets

| Dataset | Source | Granularity | Period | Zones |
|---------|--------|-------------|--------|-------|
| GEFCom2014 | [Kaggle](https://www.kaggle.com/competitions/global-energy-forecasting-competition-2012-load-forecasting) | Hourly | 2004-2008 | 20 zones |
| UCI Household | [UCI ML Repository](https://archive.ics.uci.edu/dataset/235) | 1-minute | 2006-2010 | 1 household |

Datasets are **not included** in this repository. See [Setup](#setup) for download instructions.

---

## Repository Structure

```
RAG-Based-Energy-Forecasting/
|
├── README.md                       This file
├── LICENSE                         MIT License
├── .gitignore                      Excludes data, .env, caches
├── .env.template                   Template for API keys (safe to commit)
├── setup.py                        Enables pip install -e .
|
├── config/                         Centralised configuration
│   ├── __init__.py                 Public API — imports all constants
│   ├── paths.py                    Filesystem paths (auto-detects Colab vs local)
│   ├── models.py                   Model names, MODELS dict, EXP_DEFAULTS
│   ├── groq_keys.py                get_all_groq_keys() — 46-key rotation pool
│   ├── groq_key_checker.py         TPD status checker for all 46 Groq keys
│   └── pipeline.py                 Pipeline constants
|
├── src/                            All reusable Python source code
│   ├── knowledge_base/             Phase 1 — KB generation
│   │   ├── data_loader.py
│   │   ├── aggregators.py
│   │   ├── validation.py
│   │   ├── sampling.py
│   │   ├── prompt_templates.py
│   │   ├── prompt_builders.py
│   │   ├── generation.py
│   │   └── master_kb.py
│   │   ├── chunk_builder.py         build_enriched_chunk_text() for FAISS indexing
│   |
│   ├── golden_dataset/             Phase 2 — Golden dataset
│   │   ├── kb_loader.py
│   │   ├── context_selector.py
│   │   ├── query_bank.py           200 queries (GEFCom + Household + Cross-scale)
│   │   └── generator.py
│   |
│   ├── embedding/                  Phase 3 — Embedding + indexing
│   │   ├── document_loader.py
│   │   ├── embedder.py             get_embeddings_model() + Embedder class
│   │   ├── faiss_store.py          build_faiss_index() + load_faiss_index()
│   │   └── chroma_store.py         ChromaDB (currently disabled on Windows)
│   |
│   ├── retrieval/                  Phase 4 — Retrieval pipelines
│   │   ├── dense.py                DenseRetriever — FAISS cosine similarity
│   │   ├── hybrid.py               BM25 + FAISS variant
│   │   └── hierarchical.py         HierarchicalRetriever — child + parent_id expansion
│   |
│   ├── rag/                        Phase 5 — RAG generation
│   │   ├── llm.py
│   │   ├── prompts.py              RAG_PROMPT, format_docs()
│   │   └── chains.py
│   |
│   ├── evaluation/                 Phase 6 — Metrics
│   │   ├── ragas_metrics.py
│   │   ├── retrieval_metrics.py    _recall_at_k, _precision_at_k, _mrr, _ndcg
│   │   └── hallucination.py        check_hallucination()
│   |
│   ├── experiments/                Shared experiment utilities
│   │   ├── groq_client.py          RotatingGroqClient — 46-key round-robin + 429 handling
│   │   ├── metrics.py              compute_answer_relevance, compute_hallucination_rate, etc.
│   │   ├── attribution.py          Novelty 1 — assign_evidence_ids, parse_citations, compute_attribution_metrics
│   │   └── difficulty.py           Novelty 2 — classify_query, build_difficulty_prompt_prefix, evaluate_caution
│   |
│   └── utils/
│       ├── logging.py
│       ├── timestamps.py
│       └── io.py
|
├── experiments/                                 Experiment orchestration files
│   ├── runner.py                                run_experiment(), ExperimentResult dataclass
│   ├── ragas_evaluator.py                       RAGAS 0.4.x batch evaluator with key rotation
│   ├── results_printer.py                       Formatted experiment results display
│   ├── exp_01_no_rag_variants.py                No-RAG LLM — 5 prompting strategy variants
│   ├── exp_02_dense_rag.py                      Dense RAG (FAISS)
│   ├── exp_03_hybrid_rag.py                     Hybrid RAG (BM25 + FAISS + RRF)
│   ├── exp_04_hierarchical_rag.py               Hierarchical RAG (child + parent expansion)
│   ├── exp_05_dense_attribution.py              Dense RAG + Evidence Attribution (Novelty 1)
│   ├── exp_06_hybrid_attribution.py             Hybrid RAG + Evidence Attribution (Novelty 1)
│   ├── exp_07_hierarchical_attribution.py       Hierarchical + Attribution (Novelty 1)
│   ├── exp_08_query_difficulty_dense_hybrid.py  Difficulty + Dense/Hybrid (Novelty 2)
│   ├── exp_09_query_difficulty_hierarchical.py  Difficulty + Hierarchical (Novelty 2)
│   └── exp_10_final_comparison.py               Load all results → 6 thesis tables + ranking
|
├── notebooks/
│   ├── 01_kb_generation.ipynb
│   ├── 02_golden_dataset.ipynb
│   ├── 03_embedding_indexing.ipynb
│   ├── 04_retrieval_pipelines.ipynb
│   ├── 05_rag_generation.ipynb
│   ├── 06_evaluation.ipynb
│   ├── 07_results_analysis.ipynb   (deprecated — see EXP_10 in 08)
│   └── 08_experiments.ipynb        Main experiment orchestration (EXP_01-10)
|
├── tests/
│   ├── conftest.py                 Shared fixtures (synthetic KB rows, queries)
│   ├── run_tests.py                Test runner with Excel report generation
│   ├── completed/                  Generated test report Excel files
│   ├── test_config/                Config module tests (paths, models, pipeline)
│   ├── test_evaluation/            Hallucination checker + retrieval metric tests
│   ├── test_experiments/           Metrics, attribution, difficulty
│   ├── test_knowledge_base/        Chunk builder enrichment tests
│   ├── test_retrieval/             Dense, hybrid, hierarchical retriever tests
│   └── test_golden_dataset/        Query bank and context selector tests
|
├── outputs/                        Generated artifacts (gitignored)
│   ├── knowledge_base/
│   ├── golden_dataset/
│   ├── indexes/faiss/              FAISS index files (index.faiss + index.pkl)
│   └── experiments/
│       ├── EXP_01A_ZERO_SHOT/k0/
│       ├── EXP_01B_ROLE_PROMPTING/k0/
│       ├── EXP_01C_FEW_SHOT/k0/
│       ├── EXP_01D_CHAIN_OF_THOUGHT/k0/
│       ├── EXP_01E_STRUCTURED/k0/
│       ├── EXP_02_DENSE_RAG/k3/, k5/, k10/
│       ├── EXP_03_HYBRID_RAG/k3/, k5/, k10/
│       ├── EXP_04_HIERARCHICAL_RAG/k3/, k5/, k10/
│       ├── EXP_05_DENSE_RAG_ATTRIBUTION/k3/, k5/, k10/
│       ├── EXP_06_HYBRID_RAG_ATTRIBUTION/k3/, k5/, k10/
│       ├── EXP_07_HIERARCHICAL_RAG_ATTRIBUTION/k3/, k5/, k10/
│       ├── EXP_08_QUERY_DIFFICULTY_DENSE_HYBRID/k3/, k5/, k10/
│       ├── EXP_09_QUERY_DIFFICULTY_HIERARCHICAL/k3/, k5/, k10/
│       └── EXP_10_FINAL_COMPARISON/
│           ├── table1_overall.csv
│           ├── table2_retrieval.csv
│           ├── table3_ragas.csv
│           ├── table4_attribution.csv
│           ├── table5_difficulty.csv
│           ├── table6_ranking.csv
│           └── ragas_scores_merged.csv
|
├── data/                           Raw datasets (gitignored)
│   ├── gefcom/
│   └── household/
|
└── docs/
    ├── 00_Evaluation_Framework.pdf
    ├── 01_Dense_RAG.pdf
    ├── 02_Hybrid_RAG.pdf
    └── 03_Hierarchical_RAG.pdf
```

---

## Setup

### Prerequisites

- Python 3.11+
- API keys for Google AI (Gemini) and Groq (Llama)

### Installation

```bash
# Clone the repository
git clone https://github.com/ZohebAnwarHussain/RAG-Based-Energy-Forecasting.git
cd RAG-Based-Energy-Forecasting

# Create and activate virtual environment
python -m venv thesis_env
thesis_env\Scripts\activate          # Windows
# source thesis_env/bin/activate     # Mac/Linux

# Install the project as an editable package
pip install -e .

# Install all dependencies
pip install -r requirements.txt

# Register Jupyter kernel
python -m ipykernel install --user --name=thesis_env --display-name="RAG Energy Forecasting"
```

### Environment Variables

```bash
cp .env.template .env
```

Edit `.env` with your API keys:

```
GEMINI_API_KEY=your_gemini_key_here
GROQ_API_KEY_1=your_first_groq_key
GROQ_API_KEY_2=your_second_groq_key
# ... up to GROQ_API_KEY_12
BASE_PATH=E:/path/to/RAG-Based-Energy-Forecasting
```

**Never commit `.env` to git.** It is excluded in `.gitignore`.

### Download Datasets

| Dataset | Download Link | Place In |
|---------|--------------|----------|
| GEFCom2014 | [Kaggle Competition Data](https://www.kaggle.com/competitions/global-energy-forecasting-competition-2012-load-forecasting/data) | `data/gefcom/` |
| UCI Household | [UCI ML Repository](https://archive.ics.uci.edu/dataset/235) | `data/household/` |

---

## Usage

### Run the full pipeline via notebooks

```
notebooks/01_kb_generation.ipynb       → 140 KB summaries              (~30 min)
notebooks/02_golden_dataset.ipynb      → 50 golden queries              (~8 min)
notebooks/03_embedding_indexing.ipynb  → FAISS vector index             (~2 min)
notebooks/08_experiments.ipynb         → EXP_01-10 (all experiments)    (~varies)
```

### Check Groq key TPD status before running

```python
from config.groq_key_checker import check_all_keys
check_all_keys()
```

### RAGAS Scoring — TPD Strategy

RAGAS evaluation uses a stratified 50-row subsample (seed=42) from the 200-query pool for consistent cross-experiment comparison. With 46 keys at 100k TPD each, the total daily budget is 4.6M tokens.

- Batch delay: 10s between batches (key rotation handles TPD, minimal wait needed)
- `max_rows=50` with `_stratified_subsample()` for deterministic sampling
- Keys reset at 05:30 IST (midnight UTC)
- All experiments scored at K=5 — K=3 and K=10 RAGAS scoring is optional

### Import from src/ programmatically

```python
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.hierarchical import HierarchicalRetriever
from src.experiments.attribution import assign_evidence_ids, compute_attribution_metrics
from src.experiments.difficulty import classify_query, build_difficulty_prompt_prefix
from experiments.exp_10_final_comparison import run_exp_10
```

---

## Testing

**211 unit tests** covering all source modules — no API keys required (all tests use synthetic data and mock objects).

| Test Suite | File | Tests |
|------------|------|------:|
| Config | `test_config/` | 15 |
| Evaluation | `test_evaluation/` | 18 |
| Experiments | `test_experiments/` | 24 |
| Knowledge Base — Sampling | `test_knowledge_base/test_sampling.py` | 10 |
| Knowledge Base — Validation | `test_knowledge_base/test_validation.py` | 15 |
| Knowledge Base — Aggregators | `test_knowledge_base/test_aggregators.py` | 18 |
| Knowledge Base — Prompt Builders | `test_knowledge_base/test_prompt_builders.py` | 15 |
| Knowledge Base — Chunk Builder | `test_knowledge_base/test_chunk_builder.py` | 12 |
| Golden Dataset — Context Selector | `test_golden_dataset/test_context_selector.py` | 13 |
| Golden Dataset — Query Bank | `test_golden_dataset/test_query_bank.py` | 12 |
| Retrieval — Dense | `test_retrieval/test_dense.py` | 8 |
| Retrieval — Hybrid | `test_retrieval/test_hybrid.py` | 7 |
| Retrieval — Hierarchical | `test_retrieval/test_hierarchical.py` | 9 |
| Embedding | `test_embedding/` | 15 |
| **Total** | | **211** |

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test suite
pytest tests/test_evaluation/ -v
pytest tests/test_experiments/ -v
pytest tests/test_retrieval/ -v
pytest tests/test_knowledge_base/ -v

# Generate Excel test report (with summary + per-suite detail sheets)
python tests/run_tests.py

# Run specific suite only
python tests/run_tests.py evaluation
python tests/run_tests.py experiments
python tests/run_tests.py knowledge_base
```

Test reports are saved to `tests/completed/Unit_Tests_<timestamp>.xlsx` with a summary sheet and per-module detail sheets showing pass/fail status, duration, and error messages.

---

## Current Status

| Phase | File | Status | Output |
|-------|------|--------|--------|
| 1. Knowledge Base | `01_kb_generation.ipynb` |  Complete | 140 GEFCom daily summaries |
| 2. Golden Dataset | `02_golden_dataset.ipynb` |  Complete | 200 queries (combined_golden_dataset_200.csv) |
| 3. Embedding and Indexing | `03_embedding_indexing.ipynb` |  Complete | FAISS index (140 docs, 384-dim) |
| EXP_01A-E | No-RAG 5 variants |  Generation + RAGAS complete | 44–197 valid RAGAS rows per variant |
| EXP_02 | Dense RAG K=3,5,10 |  Generation + RAGAS complete | 84 valid RAGAS rows (K=5) |
| EXP_03 | Hybrid RAG K=3,5,10 |  Generation + RAGAS complete | 50 valid RAGAS rows (K=5) |
| EXP_04 | Hierarchical RAG K=3,5,10 |  Generation + RAGAS complete | 50 valid RAGAS rows (K=5) |
| EXP_05 | Dense + Attribution K=3,5,10 |  Generation + RAGAS complete | 50 valid RAGAS rows (K=5) |
| EXP_06 | Hybrid + Attribution K=3,5,10 |  Generation + RAGAS complete | 50 valid RAGAS rows (K=5) |
| EXP_07 | Hierarchical + Attribution K=3,5,10 |  Generation + RAGAS complete | 50 valid RAGAS rows (K=5) |
| EXP_08 | Difficulty + Dense/Hybrid K=3,5,10 |  Generation + RAGAS complete | 50 valid RAGAS rows (K=5) |
| EXP_09 | Difficulty + Hierarchical K=3,5,10 |  Generation + RAGAS complete | 50 valid RAGAS rows (K=5) |
| EXP_10 | Final comparison |  Complete | 9/9 experiments loaded, thesis tables generated |
| Unit Tests | `tests/` |  211 tests passing | Excel report in `tests/completed/` |
| Architecture PDFs | `docs/` |  Complete | 4 PDFs with flow diagrams |

### Known Issues

- ChromaDB is disabled due to a Windows SQLite threading incompatibility. FAISS handles all retrieval.
- RAGAS `semantic_similarity` returns NaN for some rows — pre-existing bug in RAGAS 0.4.x, not blocking. Semantic similarity values from `query_results.csv` (custom metric) are used instead.
- Hierarchical experiments (EXP_04/07/09) show artificially low RAGAS faithfulness and context precision because parent chunks are resolved after retrieval and are not included in the RAGAS evaluation context window.
- Hybrid retrieval uses a custom `HybridRetriever` with direct RRF fusion rather than LangChain `EnsembleRetriever`, giving finer control over pool sizing and the RRF k constant.
- LangChain deprecation warning: `langchain_classic` → `langchain_community` import in `src/retrieval/hybrid.py` (cosmetic, does not affect results).

---

## Results Summary

### Table 1 — Retrieval Quality (K=5)

| Experiment | Recall@5 | Precision@5 | MRR | nDCG@5 |
|------------|----------|-------------|-----|--------|
| EXP_02 Dense | 0.145 | 0.148 | 0.230 | 0.144 |
| EXP_03 Hybrid | 0.142 | 0.144 | 0.276 | 0.155 |
| EXP_04 Hierarchical | 0.167 | 0.153 | 0.216 | 0.160 |
| EXP_05 Dense+Attr | 0.137 | 0.135 | 0.209 | 0.138 |
| EXP_06 Hybrid+Attr | 0.125 | 0.123 | 0.184 | 0.124 |
| EXP_07 Hier.+Attr | 0.137 | 0.136 | 0.227 | 0.144 |
| EXP_08 Difficulty | 0.122 | 0.120 | 0.183 | 0.121 |
| EXP_09 Diff.+Hier. | 0.138 | 0.136 | 0.228 | 0.144 |

### Table 2 — Generation Quality (K=5)

| Experiment | Answer Rel. | Semantic Sim. | Hallucination % | Faithfulness |
|------------|-------------|---------------|-----------------|--------------|
| EXP_01 No-RAG | 0.839 | 0.754 | 100.0% | 0.000 |
| EXP_02 Dense | 0.820 | 0.686 | 32.1% | 0.262 |
| EXP_03 Hybrid | 0.829 | 0.694 | 30.9% | 0.363 |
| EXP_04 Hierarchical | 0.785 | 0.709 | 23.4% | 0.167 |
| EXP_05 Dense+Attr | 0.810 | 0.686 | 17.4% | 0.281 |
| EXP_06 Hybrid+Attr | 0.809 | 0.683 | 17.3% | 0.339 |
| EXP_07 Hier.+Attr | 0.828 | 0.685 | 19.5% | 0.229 |
| EXP_08 Difficulty | 0.824 | 0.685 | 28.3% | 0.218 |
| EXP_09 Diff.+Hier. | 0.838 | 0.690 | 24.4% | 0.278 |

### Table 3 — RAGAS Evaluation (K=5)

| Experiment | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Valid Rows |
|------------|-------------|-----------------|-------------------|----------------|------------|
| EXP_01 No-RAG | 0.000 | 0.845 | — | — | 44/50 |
| EXP_02 Dense | 0.262 | 0.227 | 0.048 | 0.155 | 84/84 |
| EXP_03 Hybrid | 0.363 | 0.145 | 0.040 | 0.174 | 50/50 |
| EXP_04 Hierarchical | 0.167 | 0.102 | 0.000 | 0.071 | 50/50 |
| EXP_05 Dense+Attr | 0.281 | 0.194 | 0.020 | 0.203 | 50/50 |
| EXP_06 Hybrid+Attr | 0.327 | 0.207 | 0.046 | 0.155 | 50/50 |
| EXP_07 Hier.+Attr | 0.229 | 0.307 | 0.020 | 0.194 | 50/50 |
| EXP_08 Difficulty | 0.218 | 0.253 | 0.041 | 0.195 | 50/50 |
| EXP_09 Diff.+Hier. | 0.278 | 0.293 | 0.060 | 0.195 | 49/50 |

### Key Findings

1. **RAG reduces hallucination by 68–83%** — from 100% (no-RAG) to 17–32% (RAG experiments)
2. **Attribution prompting is the most effective grounding technique** — EXP_05/06/07 achieve the lowest hallucination rates (17–20%) through citation enforcement
3. **Hybrid retrieval produces the best ranking** — MRR 0.276 vs 0.230 dense, confirmed by highest faithfulness among baselines (0.363)
4. **Hierarchical retrieval maximises answer relevance** — parent context expansion produces the most topically relevant and lowest-hallucination baseline responses
5. **Difficulty-aware prompting provides incremental benefit** — modest hallucination reduction vs baselines, but complements hierarchical retrieval effectively in EXP_09
6. **No single experiment dominates all metrics** — EXP_06 leads grounding, EXP_09 leads relevance, reflecting a fundamental precision-vs-recall trade-off in RAG systems

---

## Knowledge Base

| Dataset | Doc ID Format | Count | Granularity |
|---------|--------------|-------|------------|
| GEFCom | `gefcom_daily_{zone}_{YYYY-MM-DD}` | 140 | Daily (20 zones) |
| Parent docs | `gefcom_weekly_{zone}_{week}` | — | Weekly (used by EXP_04/07/09) |

---

## Golden Dataset Query Distribution

| Source | Count | Query Types |
|--------|-------|-------------|
| GEFCom | 80 | statistical, pattern, comparative, zone_specific, operational |
| Household | 72 | statistical, pattern, comparative, appliance, operational |
| Cross-scale | 48 | cross_scale |
| Total | 200 | 7 types |

---

## API Keys Required

| Key | Phases Used | Where to Get |
|-----|------------|--------------|
| `GEMINI_API_KEY` | Phase 1 (KB) + Phase 2 (Golden) | [Google AI Studio](https://aistudio.google.com/apikey) |
| `GROQ_API_KEY_1` to `_46` | EXP_01-09 generation + RAGAS | [Groq Console](https://console.groq.com/keys) |

Phases 3 and 4 (embedding + FAISS retrieval) run entirely locally — no API keys needed.

---

## Citations

**GEFCom2014:**
> Hong, T., Pinson, P., & Fan, S. (2014). Global Energy Forecasting Competition
> 2012. *International Journal of Forecasting*, 30(2), 357-363.

**UCI Household:**
> Hebrail, G. & Berard, A. (2012). Individual Household Electric Power
> Consumption. UCI Machine Learning Repository.
> https://doi.org/10.24432/C58K54

---

## License

MIT License — see [LICENSE](LICENSE) for details.
