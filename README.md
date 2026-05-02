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
4. **Novelty 1 — Evidence-Linked Attribution** — All three pipelines extended with [E1], [E2] citation prompting for claim-level traceability (EXP_05-07)
5. **Novelty 2 — Query Difficulty Prediction** — Pre-generation difficulty classification (Easy/Medium/Hard) based on retrieval coverage and consistency scores, with confidence-adjusted prompting (EXP_08-09)

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

| Group | Exp ID | Pipeline | Novelty | Key Metric Added |
|-------|--------|----------|---------|-----------------|
| A | EXP_01A-E | No-RAG LLM — 5 prompting variants | None | Hallucination Rate, Answer Relevance |
| A | EXP_02 | Dense RAG (FAISS) | None | Recall@K, Faithfulness |
| A | EXP_03 | Hybrid RAG (BM25+FAISS+RRF) | None | Retrieval Quality |
| A | EXP_04 | Hierarchical RAG (child+parent) | None | Context Recall |
| B | EXP_05 | Dense RAG + Attribution | Novelty 1 | Attribution Coverage, Citation Accuracy |
| B | EXP_06 | Hybrid RAG + Attribution | Novelty 1 | Attribution Coverage, Citation Accuracy |
| B | EXP_07 | Hierarchical RAG + Attribution | Novelty 1 | Child/Parent Citation Correctness |
| C | EXP_08 | Dense/Hybrid + Difficulty | Novelty 2 | Cautious Response Accuracy |
| C | EXP_09 | Hierarchical + Difficulty | Novelty 2 | Cautious Response Accuracy |
| D | EXP_10 | Final comparative analysis | — | Composite Ranking Score |

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
| Key Rotation | `RotatingGroqClient` | Custom | 12-key round-robin + 429 handling |

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
│   ├── groq_keys.py                get_all_groq_keys() — 12-key rotation pool
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
│   |
│   ├── golden_dataset/             Phase 2 — Golden dataset
│   │   ├── kb_loader.py
│   │   ├── context_selector.py
│   │   ├── query_bank.py           50 queries (20 GEFCom + 18 Household + 12 Cross-scale)
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
│   │   ├── groq_client.py          RotatingGroqClient — 12-key round-robin + 429 handling
│   │   ├── metrics.py              compute_answer_relevance, compute_hallucination_rate, etc.
│   │   ├── attribution.py          Novelty 1 — assign_evidence_ids, parse_citations, compute_attribution_metrics
│   │   └── difficulty.py           Novelty 2 — classify_query, build_difficulty_prompt_prefix, evaluate_caution
│   |
│   └── utils/
│       ├── logging.py
│       ├── timestamps.py
│       └── io.py
|
├── experiments/                    Experiment orchestration files
│   ├── runner.py                   run_experiment(), ExperimentResult dataclass
│   ├── ragas_evaluator.py          RAGAS 0.4.x batch evaluator with key rotation
│   ├── groq_key_checker.py         TPD status checker for all 12 Groq keys
│   ├── exp_01_no_rag_variants.py   No-RAG LLM — 5 prompting strategy variants
│   ├── exp_02_dense_rag.py         Dense RAG (FAISS)
│   ├── exp_03_hybrid_rag.py        Hybrid RAG (BM25 + FAISS + RRF)
│   ├── exp_04_hierarchical_rag.py  Hierarchical RAG (child + parent expansion)
│   ├── exp_05_dense_attribution.py Dense RAG + Evidence Attribution (Novelty 1)
│   ├── exp_06_hybrid_attribution.py Hybrid RAG + Evidence Attribution (Novelty 1)
│   ├── exp_07_hierarchical_attribution.py Hierarchical + Attribution (Novelty 1)
│   ├── exp_08_query_difficulty_dense_hybrid.py Difficulty + Dense/Hybrid (Novelty 2)
│   ├── exp_09_query_difficulty_hierarchical.py Difficulty + Hierarchical (Novelty 2)
│   └── exp_10_final_comparison.py  Load all results → 6 thesis tables + ranking
|
├── notebooks/
│   ├── 01_kb_generation.ipynb
│   ├── 02_golden_dataset.ipynb
│   ├── 03_embedding_indexing.ipynb
│   ├── 04_retrieval_pipelines.ipynb
│   ├── 05_rag_generation.ipynb
│   ├── 06_evaluation.ipynb
│   ├── 07_results_analysis.ipynb
│   └── 08_experiments.ipynb        Main experiment orchestration (EXP_01-10)
|
├── tests/
│   ├── test_knowledge_base/
│   ├── test_golden_dataset/
│   ├── test_retrieval/
│   └── test_evaluation/
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
    ├── PDF1_Dense_RAG.pdf
    ├── PDF2_Hybrid_RAG.pdf
    ├── PDF3_Hierarchical_RAG.pdf   (pending)
    └── PDF0_Evaluation_Framework.pdf (pending)
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

# Install dependencies
pip install google-genai groq pandas numpy tqdm python-dotenv
pip install langchain langchain-community langchain-huggingface langchain-groq
pip install sentence-transformers faiss-cpu
pip install rank-bm25 ragas datasets
pip install black isort flake8 pytest pytest-cov

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
from groq_key_checker import check_all_keys
check_all_keys()
```

### RAGAS Scoring — TPD Strategy

RAGAS evaluation consumes approximately 600k tokens per K value per experiment.
With 12 keys at 100k TPD each, the total daily budget is 1.2M tokens.

- Run one K value per day for RAGAS scoring
- Keys reset at 05:30 IST (midnight UTC)
- Use `batch_size=2` to reduce per-key burst consumption
- K=3 on Day 1, K=5 on Day 2, K=10 on Day 3 per experiment

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

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Current Status

| Phase | File | Status | Output |
|-------|------|--------|--------|
| 1. Knowledge Base | `01_kb_generation.ipynb` | Complete | 140 GEFCom daily summaries |
| 2. Golden Dataset | `02_golden_dataset.ipynb` | Complete | 50 queries (20 GEFCom + 18 Household + 12 Cross-scale) |
| 3. Embedding and Indexing | `03_embedding_indexing.ipynb` | Complete | FAISS index (140 docs, 384-dim) |
| EXP_01A-E | No-RAG 5 variants | Generation complete | RAGAS complete for 01A, pending for 01B-E |
| EXP_02 | Dense RAG K=3,5,10 | Generation complete | RAGAS K=3/5 done, K=10 pending re-run |
| EXP_03 | Hybrid RAG K=3,5,10 | Generation complete | RAGAS K=3 done (40/50), K=5/10 pending re-run |
| EXP_04 | Hierarchical RAG | Ready to run | `exp_04_hierarchical_rag.py` complete |
| EXP_05 | Dense + Attribution | Ready to run | `exp_05_dense_attribution.py` complete |
| EXP_06 | Hybrid + Attribution | Ready to run | `exp_06_hybrid_attribution.py` complete |
| EXP_07 | Hierarchical + Attribution | Ready to run | `exp_07_hierarchical_attribution.py` complete |
| EXP_08 | Difficulty + Dense/Hybrid | Ready to run | `exp_08_query_difficulty_dense_hybrid.py` complete |
| EXP_09 | Difficulty + Hierarchical | Ready to run | `exp_09_query_difficulty_hierarchical.py` complete |
| EXP_10 | Final comparison | Pending all above | `exp_10_final_comparison.py` complete |

### Known Issues

- ChromaDB is disabled due to a Windows SQLite threading incompatibility. FAISS handles all retrieval.
- EXP_03 RAGAS K=5 and K=10 are pending re-run due to TPD exhaustion. Run each K on a separate day after the 05:30 IST reset.
- EXP_02 RAGAS K=10 had only 3/50 valid rows and is scheduled for re-run.
- Hybrid retrieval uses a custom HybridRetriever with direct RRF fusion rather than LangChain EnsembleRetriever, giving finer control over pool sizing and the RRF k constant.
- RAGAS scoring for no-RAG variants (EXP_01B-E) is scheduled across 4 consecutive days due to TPD constraints.

---

## Results Summary

| Experiment | K | Answer Rel. | Hallucination % | RAGAS Faithfulness | Valid Rows |
|------------|---|------------|----------------|-------------------|-----------|
| EXP_01A Zero-Shot | — | 0.839 | 100.0% | 0.000 | 49/50 |
| EXP_01B Role | — | TBD | 100.0% | pending | — |
| EXP_01C Few-Shot | — | TBD | 100.0% | pending | — |
| EXP_01D Chain-of-Thought | — | TBD | 100.0% | pending | — |
| EXP_01E Structured | — | TBD | 100.0% | pending | — |
| EXP_02 Dense | 3 | 0.789 | 23.0% | 0.266 | 32/50 |
| EXP_02 Dense | 5 | 0.807 | 20.4% | 0.267 | 28/50 |
| EXP_02 Dense | 10 | 0.815 | 20.7% | ~0.382* | 3/50 |
| EXP_03 Hybrid | 3 | TBD | TBD | 0.245 | 40/50 |
| EXP_03 Hybrid | 5 | TBD | TBD | pending | — |
| EXP_03 Hybrid | 10 | TBD | TBD | pending | — |

*EXP_02 K=10 RAGAS scores are indicative only — scheduled for re-run.

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
| GEFCom | 20 | statistical, pattern, comparative, zone_specific, operational |
| Household | 18 | statistical, pattern, comparative, appliance, operational |
| Cross-scale | 12 | cross_scale |
| Total | 50 | 7 types |

---

## API Keys Required

| Key | Phases Used | Where to Get |
|-----|------------|--------------|
| `GEMINI_API_KEY` | Phase 1 (KB) + Phase 2 (Golden) | [Google AI Studio](https://aistudio.google.com/apikey) |
| `GROQ_API_KEY_1` to `_12` | EXP_01-09 generation + RAGAS | [Groq Console](https://console.groq.com/keys) |

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
