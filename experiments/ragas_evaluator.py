"""
experiments/ragas_evaluator.py
================================
Standalone RAGAS evaluator for completed experiment results.
Compatible with RAGAS 0.4.x + Groq via LangChain.

Critical compatibility notes
------------------------------
RAGAS 0.4.x has TWO metric namespaces:
  - ragas.metrics.collections  → requires llm_factory (OpenAI/Anthropic ONLY)
  - ragas.metrics              → works with LangchainLLMWrapper (Groq OK)

We MUST use ragas.metrics (old-style) with .llm assignment pattern.
DO NOT use ragas.metrics.collections — incompatible with Groq.

answer_relevancy.strictness = 1 is required.
Default strictness=3 sends n=3 to Groq → HTTP 400 "n must be at most 1".

Key rotation
------------
Each batch uses a different Groq key (round-robin).
LLM is reassigned to each metric before every batch call.

No-RAG handling
---------------
EXP_01 (K=0): only faithfulness + answer_relevancy scored.
context_precision and context_recall set to NaN explicitly.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from config.models import MODELS, EXP_DEFAULTS
from config.groq_keys import get_all_groq_keys

logger = logging.getLogger(__name__)

RAGAS_METRIC_COLS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "semantic_similarity",   # cosine similarity vs reference answer (post-RAGAS, no OpenAI needed)
]

BATCH_DELAY = 30.0   # seconds between batches


# ---------------------------------------------------------------------------
# Key rotator
# ---------------------------------------------------------------------------

class _KeyRotator:
    """
    Round-robin key rotator with TPD exhaustion tracking.

    When a key returns a TPD 429 error, call mark_exhausted(key) to
    permanently blacklist it for the rest of the session. Subsequent
    calls to next_key() will skip all blacklisted keys automatically.

    This prevents the second-pass problem where the rotator cycles back
    to keys that were already exhausted in earlier batches.
    """

    def __init__(self) -> None:
        self._keys      = get_all_groq_keys()
        self._idx       = 0
        self._exhausted: set[str] = set()
        logger.info("RAGAS key rotator: %d key(s) available.", len(self._keys))

    def mark_exhausted(self, key: str) -> None:
        """Blacklist a key that has hit its TPD limit."""
        if key not in self._exhausted:
            self._exhausted.add(key)
            remaining = len(self._keys) - len(self._exhausted)
            logger.warning(
                "Key ...%s marked TPD exhausted. %d/%d keys remaining.",
                key[-4:], remaining, len(self._keys),
            )
            print(
                f"  Key ...{key[-4:]} TPD exhausted — blacklisted. "
                f"{remaining}/{len(self._keys)} keys still available.",
                flush=True,
            )

    def next_key(self) -> str | None:
        """
        Return the next available (non-exhausted) key.
        Returns None if ALL keys are exhausted.
        Skips exhausted keys transparently.
        """
        n = len(self._keys)
        for _ in range(n):
            key = self._keys[self._idx % n]
            self._idx += 1
            if key not in self._exhausted:
                return key
        # All keys exhausted
        logger.error("All %d Groq keys are TPD exhausted.", n)
        return None

    @property
    def n_keys(self) -> int:
        return len(self._keys)

    @property
    def n_available(self) -> int:
        return len(self._keys) - len(self._exhausted)

    def current_label(self) -> str:
        avail = self.n_available
        return f"Key {(self._idx % self.n_keys) + 1}/{self.n_keys} ({avail} available)"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_experiment_results(exp_dir: Path) -> pd.DataFrame:
    csv_path = exp_dir / "query_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No query_results.csv at {csv_path}\n"
            "Run the experiment first before RAGAS scoring."
        )
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows from %s", len(df), csv_path)
    return df


def _build_ragas_rows(
    results_df: pd.DataFrame,
    golden_df:  pd.DataFrame,
) -> tuple[list[dict], list[str], list[str]]:
    golden_by_qid: dict = {}
    golden_by_gid: dict = {}
    for _, row in golden_df.iterrows():
        qid = str(row.get("query_id", ""))
        gid = str(row.get("golden_id", ""))
        gt  = str(row.get("ground_truth", row.get("reference_answer", "")))
        if qid:
            golden_by_qid[qid] = gt
        if gid:
            golden_by_gid[gid] = gt

    rows, gids, pipes = [], [], []
    for _, row in results_df.iterrows():
        question = str(row.get("question") or row.get("user_query") or "")
        answer   = str(row.get("answer")   or row.get("rag_answer")  or "")
        if not answer or answer == "nan":
            continue
        contexts = _extract_contexts(row)
        qid      = str(row.get("query_id",  ""))
        gid      = str(row.get("golden_id", ""))
        gt       = golden_by_qid.get(qid) or golden_by_gid.get(gid) or ""
        rows.append({
            "question":     question,
            "answer":       answer,
            "contexts":     contexts,
            "ground_truth": gt,
        })
        gids.append(gid or qid)
        pipes.append(str(row.get("pipeline", "unknown")))

    logger.info("Built %d RAGAS rows (%d skipped).",
                len(rows), len(results_df) - len(rows))
    return rows, gids, pipes


def _extract_contexts(row: pd.Series) -> list[str]:
    raw_docs = row.get("retrieved_docs")
    if raw_docs and str(raw_docs) not in ("nan", "[]", ""):
        try:
            docs  = json.loads(str(raw_docs))
            texts = []
            for d in (docs if isinstance(docs, list) else []):
                if isinstance(d, dict):
                    texts.append(d.get("text") or d.get("page_content") or str(d))
                else:
                    texts.append(str(d))
            clean = [t for t in texts if t.strip()]
            if clean:
                return clean
        except (json.JSONDecodeError, TypeError):
            pass
    raw_ctx = row.get("retrieved_context") or row.get("context") or ""
    if raw_ctx and str(raw_ctx) not in ("nan", ""):
        chunks = [c.strip() for c in str(raw_ctx).split("\n\n") if c.strip()]
        return chunks if chunks else [str(raw_ctx)]
    return ["No context retrieved."]


def _build_hf_dataset(rows: list[dict], is_no_rag: bool):
    """Build HuggingFace Dataset for old-style RAGAS evaluate().

    semantic_similarity is computed post-RAGAS — 'reference' column no longer needed.
    We include it for all runs. The metric computes cosine similarity between
    the generated answer embedding and the reference answer embedding.
    """
    from datasets import Dataset
    return Dataset.from_dict({
        "question":     [r["question"]                          for r in rows],
        "answer":       [r["answer"]                            for r in rows],
        "contexts":     [[] if is_no_rag else r["contexts"]     for r in rows],
        "ground_truth": [r["ground_truth"]                      for r in rows],
    })


# ---------------------------------------------------------------------------
# Core RAGAS runner
# ---------------------------------------------------------------------------

def _run_ragas_04x(
    ragas_rows:  list[dict],
    golden_ids:  list[str],
    pipelines:   list[str],
    output_path: Path,
    is_no_rag:   bool = False,
    batch_size:  int  = 5,
) -> Optional[pd.DataFrame]:
    """
    Run RAGAS evaluate() using OLD-STYLE ragas.metrics (not .collections).

    Pattern for each batch:
      1. Get next Groq key → build LangchainLLMWrapper
      2. Assign .llm to each metric instance
      3. Build HuggingFace Dataset for this batch
      4. Call evaluate() with metric instances (not classes)
      5. Save incrementally
    """
    # ── Imports ──────────────────────────────────────────────────────────────
    try:
        from ragas import evaluate
        from ragas.metrics import (         # OLD-STYLE — works with Groq
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_groq import ChatGroq
        from langchain_huggingface import HuggingFaceEmbeddings
        from ragas import RunConfig
    except ImportError as e:
        logger.error("RAGAS import failed: %s", e)
        return None

    # ── Fix strictness BEFORE any batch runs ─────────────────────────────────
    # answer_relevancy.strictness=3 sends n=3 → Groq HTTP 400
    answer_relevancy.strictness = 1

    # ── Shared embeddings ─────────────────────────────────────────────────────
    _hf_emb = HuggingFaceEmbeddings(
        model_name=MODELS["embedding"],
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(_hf_emb)

    # Set embeddings on answer_relevancy (only metric that needs them)
    answer_relevancy.embeddings = ragas_embeddings

    # ── SemanticSimilarity is computed post-RAGAS using custom cosine metric ──
    # ragas.metrics.collections.SemanticSimilarity requires OpenAI modern
    # embeddings interface which is incompatible with LangchainEmbeddingsWrapper.
    # semantic_similarity is appended to ragas_scores.csv after RAGAS runs,
    # using compute_semantic_similarity() from src/experiments/metrics.py
    # (cosine similarity via all-MiniLM-L6-v2 — same model, same result).

    # ── Metrics list ─────────────────────────────────────────────────────────
    if is_no_rag:
        active_metrics = [faithfulness, answer_relevancy]
        logger.info(
            "No-RAG mode: faithfulness + answer_relevancy + semantic_similarity only. "
            "context_precision and context_recall will be NaN."
        )
    else:
        active_metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

    # ── Run config ────────────────────────────────────────────────────────────
    run_cfg = RunConfig(timeout=120, max_retries=3, max_wait=60)

    rotator    = _KeyRotator()
    all_scores: list[pd.DataFrame] = []
    n_batches  = (len(ragas_rows) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start       = batch_idx * batch_size
        end         = min(start + batch_size, len(ragas_rows))
        batch_rows  = ragas_rows[start:end]
        batch_gids  = golden_ids[start:end]
        batch_pipes = pipelines[start:end]

        # ── Rotate key → build LLM wrapper ───────────────────────────────────
        key = rotator.next_key()

        # All keys exhausted — stop scoring
        if key is None:
            print(
                f"  All keys TPD exhausted at batch {batch_idx + 1}/{n_batches}. "
                f"Stopping. {len(all_scores) * batch_size} rows scored.",
                flush=True,
            )
            logger.error("All keys exhausted at batch %d.", batch_idx + 1)
            break

        label = rotator.current_label()
        llm   = LangchainLLMWrapper(
            ChatGroq(
                model=MODELS["groq_rag"],
                temperature=0,
                max_tokens=2048,
                api_key=key,
            )
        )

        # ── Assign LLM to ALL active metrics ─────────────────────────────────
        for metric in active_metrics:
            metric.llm = llm

        logger.info(
            "RAGAS batch %d/%d | rows %d-%d | %s",
            batch_idx + 1, n_batches, start, end - 1, label,
        )
        print(
            f"  Batch {batch_idx + 1}/{n_batches} | "
            f"rows {start}-{end-1} | {label}",
            flush=True,
        )

        dataset = _build_hf_dataset(batch_rows, is_no_rag)

        try:
            result   = evaluate(
                dataset=dataset,
                metrics=active_metrics,
                run_config=run_cfg,
                raise_exceptions=False,
                show_progress=True,
            )
            batch_df = result.to_pandas()

            # Normalise column name (RAGAS sometimes returns answer_relevance)
            batch_df = batch_df.rename(columns={
                "answer_relevance": "answer_relevancy",
                "semantic_similarity": "semantic_similarity",
            })

            # Check if the batch scored nothing — may indicate TPD exhaustion
            # even without a raised exception (RAGAS swallows some 429s)
            n_scored = int(batch_df["faithfulness"].notna().sum()) \
                if "faithfulness" in batch_df.columns else 0
            if n_scored == 0 and len(batch_rows) > 0:
                logger.warning(
                    "Batch %d scored 0/%d rows — possible silent TPD failure "
                    "on key ...%s. Marking exhausted.",
                    batch_idx + 1, len(batch_rows), key[-4:],
                )
                rotator.mark_exhausted(key)

        except Exception as exc:
            exc_str = str(exc)
            # Detect TPD exhaustion in the exception message
            if "tokens per day" in exc_str.lower() or "tpd" in exc_str.lower():
                rotator.mark_exhausted(key)
            logger.warning(
                "RAGAS batch %d failed: %s — NaN for this batch.",
                batch_idx + 1, exc,
            )
            print(f"  Batch {batch_idx + 1} failed: {exc_str[:120]}", flush=True)
            batch_df = pd.DataFrame([
                {col: float("nan") for col in RAGAS_METRIC_COLS}
                for _ in batch_rows
            ])

        # Ensure all 4 columns exist
        for col in RAGAS_METRIC_COLS:
            if col not in batch_df.columns:
                batch_df[col] = float("nan")

        batch_df["golden_id"] = batch_gids
        batch_df["pipeline"]  = batch_pipes
        all_scores.append(batch_df)

        # Incremental save
        combined = pd.concat(all_scores, ignore_index=True)
        combined.to_csv(output_path, index=False)
        n_valid = int(combined["faithfulness"].notna().sum())
        print(f"  {n_valid}/{len(combined)} rows scored so far", flush=True)

        if batch_idx < n_batches - 1:
            print(f"   Waiting {BATCH_DELAY:.0f}s...", flush=True)
            time.sleep(BATCH_DELAY)

    final_df = pd.concat(all_scores, ignore_index=True)

    # No-RAG: null context metrics explicitly
    if is_no_rag:
        for col in ["context_precision", "context_recall"]:
            final_df[col] = float("nan")
        logger.info("context_precision/recall set to NaN (No-RAG).")

    # ── Semantic Similarity — computed post-RAGAS via cosine similarity ───────
    # Uses compute_semantic_similarity() from src/experiments/metrics.py
    # which applies all-MiniLM-L6-v2 cosine similarity between answer and
    # ground_truth. This is semantically equivalent to RAGAS SemanticSimilarity
    # but works with LangchainEmbeddingsWrapper (no OpenAI required).
    try:
        from src.experiments.metrics import compute_semantic_similarity
        sem_scores = []
        for _, row in final_df.iterrows():
            answer       = str(row.get("answer", "") or "")
            ground_truth = str(row.get("ground_truth", "") or "")
            if answer and ground_truth:
                score = compute_semantic_similarity(answer, ground_truth)
            else:
                score = float("nan")
            sem_scores.append(score)
        final_df["semantic_similarity"] = sem_scores
        valid_sem = sum(1 for s in sem_scores if s is not None and s == s)
        logger.info(
            "Semantic similarity computed: %d/%d valid rows.",
            valid_sem, len(final_df),
        )
    except Exception as exc:
        logger.warning("Semantic similarity computation failed: %s", exc)
        final_df["semantic_similarity"] = float("nan")
    final_df.to_csv(output_path, index=False)
    return final_df


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def run_ragas_for_experiment(
    exp_id:      str,
    top_k:       int,
    golden_df:   pd.DataFrame,
    outputs_dir: str | Path = "outputs/experiments",
    batch_size:  int  = 5,
    force_rerun: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Run RAGAS evaluation for one experiment at one K value.

    Reads  : outputs/experiments/{exp_id}/k{top_k}/query_results.csv
    Saves  : outputs/experiments/{exp_id}/k{top_k}/ragas_scores.csv
    Returns: DataFrame with RAGAS scores, or None on failure.
    """
    outputs_dir = Path(outputs_dir)
    exp_dir     = outputs_dir / exp_id / f"k{top_k}"
    output_path = exp_dir / "ragas_scores.csv"
    is_no_rag   = (top_k == 0)

    if output_path.exists() and not force_rerun:
        existing = pd.read_csv(output_path)
        n_valid  = (
            int(existing["faithfulness"].notna().sum())
            if "faithfulness" in existing.columns else 0
        )
        total = len(existing)
        if n_valid >= total * 0.9:
            logger.info(
                "Scores already complete for %s k=%d (%d/%d). "
                "Use force_rerun=True to re-score.",
                exp_id, top_k, n_valid, total,
            )
            return existing
        logger.info("Partial scores (%d/%d) — re-running.", n_valid, total)

    print(f"\n{'='*60}")
    print(f"  RAGAS: {exp_id} | K={top_k}")
    print(f"  batch_size={batch_size} | delay={BATCH_DELAY:.0f}s | no_rag={is_no_rag}")
    print(f"{'='*60}")

    results_df = _load_experiment_results(exp_dir)
    ragas_rows, golden_ids, pipelines = _build_ragas_rows(results_df, golden_df)

    if not ragas_rows:
        logger.error("No valid rows for %s k=%d", exp_id, top_k)
        return None

    scores_df = _run_ragas_04x(
        ragas_rows=ragas_rows,
        golden_ids=golden_ids,
        pipelines=pipelines,
        output_path=output_path,
        is_no_rag=is_no_rag,
        batch_size=batch_size,
    )

    if scores_df is not None:
        _log_summary(scores_df, exp_id, top_k)
    return scores_df


def run_ragas_for_all_k(
    exp_id:      str,
    golden_df:   pd.DataFrame,
    k_values:    list[int] | None = None,
    outputs_dir: str | Path = "outputs/experiments",
    batch_size:  int  = 5,
    force_rerun: bool = False,
) -> dict[int, pd.DataFrame]:
    if k_values is None:
        k_values = EXP_DEFAULTS["top_k_values"]
    results = {}
    for k in k_values:
        scores = run_ragas_for_experiment(
            exp_id=exp_id, top_k=k, golden_df=golden_df,
            outputs_dir=outputs_dir, batch_size=batch_size,
            force_rerun=force_rerun,
        )
        if scores is not None:
            results[k] = scores
        time.sleep(5)
    return results


def run_ragas_for_experiments(
    exp_ids:     list[str],
    k_map:       dict[str, list[int]],
    golden_df:   pd.DataFrame,
    outputs_dir: str | Path = "outputs/experiments",
    batch_size:  int  = 5,
    force_rerun: bool = False,
) -> dict[str, dict[int, pd.DataFrame]]:
    all_results = {}
    for exp_id in exp_ids:
        k_values = k_map.get(exp_id, EXP_DEFAULTS["top_k_values"])
        all_results[exp_id] = run_ragas_for_all_k(
            exp_id=exp_id, golden_df=golden_df, k_values=k_values,
            outputs_dir=outputs_dir, batch_size=batch_size,
            force_rerun=force_rerun,
        )
        time.sleep(10)
    return all_results


def load_ragas_scores(
    exp_id:      str,
    top_k:       int,
    outputs_dir: str | Path = "outputs/experiments",
) -> Optional[pd.DataFrame]:
    path = Path(outputs_dir) / exp_id / f"k{top_k}" / "ragas_scores.csv"
    return pd.read_csv(path) if path.exists() else None


def summarise_ragas_scores(
    scores_df: pd.DataFrame,
    exp_id:    str = "",
    top_k:     int = 0,
) -> dict:
    summary: dict = {"exp_id": exp_id, "top_k": top_k}
    for col in RAGAS_METRIC_COLS:
        if col in scores_df.columns:
            vals = scores_df[col].dropna()
            summary[col] = round(float(vals.mean()), 4) if len(vals) > 0 else None
        else:
            summary[col] = None
    summary["n_valid_faithfulness"] = (
        int(scores_df["faithfulness"].notna().sum())
        if "faithfulness" in scores_df.columns else 0
    )
    summary["n_total"] = len(scores_df)
    return summary


def _log_summary(scores_df: pd.DataFrame, exp_id: str, top_k: int) -> None:
    print(f"\n{'='*60}")
    print(f"  RAGAS SCORES — {exp_id} | K={top_k}")
    print(f"{'='*60}")
    for col in RAGAS_METRIC_COLS:
        if col in scores_df.columns:
            vals    = scores_df[col].dropna()
            n_valid = len(vals)
            n_total = len(scores_df)
            mean    = vals.mean() if n_valid > 0 else float("nan")
            flag    = "WARN" if n_valid < n_total * 0.9 else "OK  "
            print(f"  [{flag}] {col:25s}: {mean:.4f}  ({n_valid}/{n_total} valid)")
        else:
            print(f"  [MISS] {col:25s}: not computed")
    print(f"{'='*60}\n")