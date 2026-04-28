# RAG-Based Energy Forecasting

**MSc Thesis — Liverpool John Moores University, UK**

| | |
|---|---|
| **Author** | Zoheb Anwar Hussain (Student ID: 1196931) |
| **Supervisor** | Ankan Dutta |
| **Programme** | Master of Science — Liverpool John Moores University |
| **Submission** | 2026 |

---

## Research Title

**RAG-Driven Natural Language Energy Demand Forecasting and Consumption
Insight Generator Using Household & Multi-Zone Load Data**

---

## Overview

This project implements a Retrieval-Augmented Generation (RAG) framework
that transforms historical energy time-series data into evidence-grounded
natural language demand insights. Two datasets — GEFCom2012 (20-zone utility
load) and UCI Individual Household Power Consumption — are statistically
aggregated into structured summaries, embedded into a vector store, and
retrieved by an LLM to generate stakeholder-friendly energy forecasting
explanations.

Three retrieval strategies are compared:

1. **Dense** — FAISS cosine similarity over sentence embeddings
2. **Hybrid** — BM25 sparse retrieval combined with dense retrieval via
   LangChain `EnsembleRetriever`
3. **Hierarchical** — Parent-child document linking (daily → weekly → monthly)
   with metadata-filtered retrieval

Performance is evaluated using RAGAS metrics (faithfulness, answer relevancy,
context precision, context recall) and standard retrieval metrics (Recall@k,
Precision@k, MRR, nDCG).

---

## Architecture

```
Raw Time-Series Data (GEFCom2012 + UCI Household)
        ↓
Phase 1 — Statistical Aggregation + Gemini 3 Flash Summaries → 480 KB entries
        ↓
Phase 2 — Gemini 2.5 Flash Reference Answers → 50 Golden Dataset queries
        ↓
Phase 3 — LangChain CSVLoader + Sentence Transformers → FAISS Vector Index
        ↓
Phase 4 — Three Retrieval Pipelines (Dense | Hybrid | Hierarchical)
        ↓
Phase 5 — LangChain LCEL + Llama 3.3 70B via Groq → RAG Insights
        ↓
Phase 6 — RAGAS Evaluation + Retrieval Metrics
        ↓
Phase 7 — Comparative Analysis + Thesis Findings
```

---

## Models and Tools

| Stage | Component | Provider | Purpose |
|-------|-----------|----------|---------|
| KB Generation | `gemini-3-flash-preview` | Google AI | Generate human-tone summaries from statistics |
| Golden Dataset | `gemini-2.5-flash` | Google AI | Generate reference answers for evaluation |
| Document Loading | LangChain `CSVLoader` | LangChain | Load KB summaries as `Document` objects |
| Embedding | `all-MiniLM-L6-v2` | HuggingFace (local) | 384-dim sentence embeddings, runs on CPU |
| Vector Store | FAISS `IndexFlatIP` | Facebook AI | Dense similarity search |
| Sparse Retrieval | `BM25Retriever` | LangChain | Keyword-based retrieval for hybrid pipeline |
| Hybrid Retrieval | `EnsembleRetriever` | LangChain | Weighted fusion of BM25 + dense |
| RAG Generation | `llama-3.3-70b-versatile` | Groq | Natural language insight generation |
| RAG Chain | LCEL (pipe operator) | LangChain | `retriever \| prompt \| llm \| parser` |
| Evaluation | RAGAS | Exploding Gradients | Faithfulness, relevancy, precision, recall |

### Model Independence Strategy

Two boundaries protect evaluation validity:

- **KB → Golden Dataset**: Different Gemini generations (3 Flash vs 2.5 Flash)
  prevent stylistic contamination between KB summaries and reference answers.
- **Golden Dataset → RAG Generation**: Gemini family (reference answers) vs
  Llama family (RAG answers) ensures RAGAS scores reflect genuine cross-model
  evaluation rather than a model grading its own output.

---

## Datasets

| Dataset | Source | Granularity | Period | Zones |
|---------|--------|-------------|--------|-------|
| GEFCom2012 | [Kaggle](https://www.kaggle.com/competitions/global-energy-forecasting-competition-2012-load-forecasting) | Hourly | 2004–2008 | 20 zones |
| UCI Household | [UCI ML Repository](https://archive.ics.uci.edu/dataset/235) | 1-minute | 2006–2010 | 1 household |

Datasets are **not included** in this repository. See [Setup](#setup) for download instructions.

---

## Repository Structure

```
RAG-Based-Energy-Forecasting/
│
├── README.md                       This file
├── LICENSE                         MIT License
├── .gitignore                      Excludes data, .env, caches
├── .env.template                   Template for API keys (safe to commit)
├── setup.py                        Enables `pip install -e .`
├── pyproject.toml                  Black, isort, flake8, pytest config
│
├── config/                         Centralised configuration
│   ├── __init__.py                 Public API — imports all constants
│   ├── paths.py                    Filesystem paths (auto-detects Colab vs local)
│   ├── models.py                   Model names, temperatures, token limits
│   └── pipeline.py                 Pipeline constants (pilot size, rate limits)
│
├── src/                            All reusable Python source code
│   ├── __init__.py
│   │
│   ├── knowledge_base/             Phase 1 — KB generation (pure Python + Gemini)
│   │   ├── data_loader.py          Load raw GEFCom + Household CSVs
│   │   ├── aggregators.py          Statistical aggregation (9 functions)
│   │   ├── validation.py           Data + summary quality validation
│   │   ├── sampling.py             Stratified sampling across zones/years
│   │   ├── prompt_templates.py     10 Gemini prompt template strings
│   │   ├── prompt_builders.py      10 prompt-input row builders
│   │   ├── generation.py           Gemini API client, call, batch generation
│   │   └── master_kb.py            Master KB builder with metadata + parent_id
│   │
│   ├── golden_dataset/             Phase 2 — Golden dataset (pure Python + Gemini)
│   │   ├── kb_loader.py            Load master KB for context selection
│   │   ├── context_selector.py     Single-dataset + cross-scale context selection
│   │   ├── query_bank.py           50 queries (20 GEFCom + 18 Household + 12 Cross-scale)
│   │   └── generator.py            Gemini 2.5 Flash client + generation + assembly
│   │
│   ├── embedding/                  Phase 3 — LangChain document loading + indexing
│   │   ├── document_loader.py      CSVLoader → Document objects (summary as page_content)
│   │   ├── embedder.py             HuggingFaceEmbeddings wrapper (all-MiniLM-L6-v2)
│   │   ├── faiss_store.py          FAISS index build + save + load
│   │   └── chroma_store.py         ChromaDB build + persist + load (currently disabled)
│   │
│   ├── retrieval/                  Phase 4 — Three retrieval pipelines
│   │   ├── dense.py                Pipeline 1: FAISS cosine similarity
│   │   ├── hybrid.py               Pipeline 2: BM25 + FAISS EnsembleRetriever
│   │   └── hierarchical.py         Pipeline 3: Parent-child metadata retrieval
│   │
│   ├── rag/                        Phase 5 — RAG generation
│   │   ├── llm.py                  ChatGroq wrapper (Llama 3.3 70B)
│   │   ├── prompts.py              Generation prompt templates
│   │   └── chains.py               LCEL chain construction
│   │
│   ├── evaluation/                 Phase 6 — Metrics and evaluation
│   │   ├── ragas_metrics.py        RAGAS faithfulness, relevancy, precision, recall
│   │   ├── retrieval_metrics.py    Recall@k, Precision@k, MRR, nDCG
│   │   └── hallucination.py        answer_must_include / answer_must_not_include checks
│   │
│   └── utils/                      Shared utilities
│       ├── logging.py              Logger setup + progress section headers
│       ├── timestamps.py           DD-MM-YYYY HH:MM:SS UTC formatting
│       └── io.py                   CSV append-mode helpers
│
├── notebooks/                      Orchestration notebooks (thin, import from src/)
│   ├── 01_kb_generation.ipynb      Phase 1 — generates ~480 KB summaries
│   ├── 02_golden_dataset.ipynb     Phase 2 — generates 50 golden queries
│   ├── 03_embedding_indexing.ipynb  Phase 3 — builds FAISS vector index
│   ├── 04_retrieval_pipelines.ipynb Phase 4 — runs 3 retrieval strategies
│   ├── 05_rag_generation.ipynb     Phase 5 — generates RAG answers
│   ├── 06_evaluation.ipynb         Phase 6 — RAGAS + retrieval metrics
│   └── 07_results_analysis.ipynb   Phase 7 — charts + findings
│
├── tests/                          Pytest unit tests
│   ├── test_knowledge_base/
│   ├── test_golden_dataset/
│   ├── test_retrieval/
│   └── test_evaluation/
│
├── outputs/                        Generated artifacts (gitignored)
│   ├── knowledge_base/             KB summaries + intermediate files
│   ├── golden_dataset/             Golden dataset CSVs
│   ├── indexes/                    FAISS index files
│   ├── retrieval_results/
│   ├── rag_results/
│   ├── evaluation_results/
│   └── charts/
│
├── data/                           Raw datasets (gitignored)
│   ├── gefcom/
│   └── household/
│
├── logs/                           Runtime logs (gitignored)
│
├── scripts/                        Setup and utility scripts
│
└── docs/                           Stage-by-stage documentation
    ├── 01_knowledge_base.md
    ├── 02_golden_dataset.md
    ├── 03_embedding_indexing.md
    ├── 04_retrieval_pipelines.md
    ├── 05_rag_generation.md
    ├── 06_evaluation.md
    └── 07_results_analysis.md
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
pip install sentence-transformers faiss-cpu chromadb
pip install rank-bm25 ragas datasets
pip install black isort flake8 pytest pytest-cov

# Register Jupyter kernel
python -m ipykernel install --user --name=thesis_env --display-name="LJMU Thesis"
```

### Environment Variables

```bash
cp .env.template .env
```

Edit `.env` and fill in your API keys:

```
GEMINI_API_KEY=your_gemini_key_here
GROQ_API_KEY=your_groq_key_here
BASE_PATH=E:/path/to/RAG-Based-Energy-Forecasting
```

**Never commit `.env` to git.** It is excluded in `.gitignore`.

### Download Datasets

| Dataset | Download Link | Place In |
|---------|--------------|----------|
| GEFCom2012 | [Kaggle Competition Data](https://www.kaggle.com/competitions/global-energy-forecasting-competition-2012-load-forecasting/data) | `data/gefcom/` |
| UCI Household | [UCI ML Repository](https://archive.ics.uci.edu/dataset/235) | `data/household/` |

---

## Usage

### Run the full pipeline via notebooks

Open each notebook in VS Code or JupyterLab and run cells top-to-bottom:

```
notebooks/01_kb_generation.ipynb       → ~480 KB summaries        (~70 min)
notebooks/02_golden_dataset.ipynb      → 50 golden queries         (~8 min)
notebooks/03_embedding_indexing.ipynb   → FAISS vector index        (~2 min)
notebooks/04_retrieval_pipelines.ipynb  → retrieval results         (pending)
notebooks/05_rag_generation.ipynb       → RAG answers               (pending)
notebooks/06_evaluation.ipynb           → RAGAS scores              (pending)
notebooks/07_results_analysis.ipynb     → charts + findings         (pending)
```

### Import from src/ programmatically

```python
from src.knowledge_base import load_gefcom_data, generate_summaries
from src.golden_dataset import generate_golden_dataset, GEFCOM_QUERIES
from src.embedding import load_kb_documents, get_embeddings_model, build_faiss_index
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

| Phase | Notebook | Status | Output |
|-------|----------|--------|--------|
| 1. Knowledge Base | `01_kb_generation.ipynb` | ✅ Complete | 480 summaries across 10 types |
| 2. Golden Dataset | `02_golden_dataset.ipynb` | ✅ Complete | 50 queries (20 GEFCom + 18 Household + 12 Cross-scale) |
| 3. Embedding & Indexing | `03_embedding_indexing.ipynb` | ✅ Complete | FAISS index built and validated |
| 4. Retrieval Pipelines | `04_retrieval_pipelines.ipynb` | 🔄 In Progress | Dense, Hybrid, Hierarchical |
| 5. RAG Generation | `05_rag_generation.ipynb` | ⏳ Pending | Llama 3.3 70B via Groq |
| 6. Evaluation | `06_evaluation.ipynb` | ⏳ Pending | RAGAS metrics |
| 7. Results Analysis | `07_results_analysis.ipynb` | ⏳ Pending | Charts + comparative findings |

### Known Issues

- **ChromaDB** is currently disabled due to a Windows SQLite threading
  incompatibility. The `chroma_store.py` module is implemented and ready
  but crashes the Jupyter kernel on Windows. FAISS handles all retrieval
  for now. Metadata filtering is done post-retrieval in Python. ChromaDB
  will be re-enabled when tested on a Linux/Colab environment.

---

## Knowledge Base Summary Types

| Dataset | Types | Pilot Count |
|---------|-------|-------------|
| GEFCom | daily, weekly, monthly, seasonal, system_level | ~64 each |
| Household | daily, weekly, monthly, appliance, yearly | 5–50 each |
| **Total** | **10 types** | **~480 summaries** |

---

## Golden Dataset Query Distribution

| Source | Count | Query Types |
|--------|-------|-------------|
| GEFCom | 20 | statistical, pattern, comparative, zone_specific, operational |
| Household | 18 | statistical, pattern, comparative, appliance, operational |
| Cross-scale | 12 | cross_scale (spans both datasets at multiple granularities) |
| **Total** | **50** | **7 query types** |

| Difficulty | Count |
|------------|-------|
| Easy | ~12 |
| Medium | ~22 |
| Hard | ~16 |

---

## API Keys Required

| Key | Phases Used | Where to Get |
|-----|------------|--------------|
| `GEMINI_API_KEY` | Phase 1 (KB) + Phase 2 (Golden) | [Google AI Studio](https://aistudio.google.com/apikey) |
| `GROQ_API_KEY` | Phase 5 (RAG) + Phase 6 (RAGAS) | [Groq Console](https://console.groq.com/keys) |

Phases 3 and 4 (embedding + retrieval) run entirely locally — no API keys needed.

---

## Citations

**GEFCom2012:**
> Hong, T., Pinson, P., & Fan, S. (2014). Global Energy Forecasting Competition
> 2012. *International Journal of Forecasting*, 30(2), 357–363.

**UCI Household:**
> Hebrail, G. & Berard, A. (2012). Individual Household Electric Power
> Consumption. UCI Machine Learning Repository.
> https://doi.org/10.24432/C58K54

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

**Zoheb Anwar Hussain**
MSc Candidate, Liverpool John Moores University
GitHub: [@ZohebAnwarHussain](https://github.com/ZohebAnwarHussain)
