"""Filesystem path configuration.

All paths used across the pipeline are defined here as pathlib.Path
objects. The base path is auto-detected based on environment:

    - Google Colab: uses /content/drive/MyDrive/LJMU_Thesis
    - Local machine: reads BASE_PATH from .env file

This makes the codebase portable across environments without manual
path edits in notebooks or modules.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

# Load environment variables from .env file in project root
load_dotenv()


def _detect_base_path() -> Path:
    """Auto-detect the project base path.

    Detection priority:
        1. Google Colab — uses Drive mount path
        2. Environment variable BASE_PATH from .env
        3. Current working directory as final fallback

    Returns:
        Absolute path to the project root directory.
    """
    if os.path.exists("/content/drive"):
        return Path("/content/drive/MyDrive/LJMU_Thesis")

    env_path = os.environ.get("BASE_PATH")
    if env_path:
        return Path(env_path)

    return Path.cwd()


BASE: Path = _detect_base_path()


# All pipeline directory paths defined relative to BASE
PATHS: Dict[str, Path] = {
    # ── Raw input data ───────────────────────────────────────────────────────
    "data_gefcom":        BASE / "data" / "gefcom",
    "data_household":     BASE / "data" / "household",
    # ── Processed intermediate files ─────────────────────────────────────────
    "proc_gefcom":        BASE / "outputs" / "knowledge_base" / "data_processed" / "gefcom",
    "proc_household":     BASE / "outputs" / "knowledge_base" / "data_processed" / "household",
    # ── Prompt inputs ────────────────────────────────────────────────────────
    "prompt_inputs":      BASE / "outputs" / "knowledge_base" / "prompt_inputs",
    # ── Generated KB summaries ───────────────────────────────────────────────
    "summaries_csv":      BASE / "outputs" / "knowledge_base" / "generated_summaries" / "csv",
    # ── Golden dataset outputs ───────────────────────────────────────────────
    "golden_dataset":     BASE / "outputs" / "golden_dataset",
    # ── Vector indexes ───────────────────────────────────────────────────────
    "indexes":            BASE / "outputs" / "indexes",
    "faiss_index":        BASE / "outputs" / "indexes" / "faiss",
    "chroma_index":       BASE / "outputs" / "indexes" / "chromadb",
    # ── Retrieval, RAG, and evaluation results ───────────────────────────────
    "retrieval_results":  BASE / "outputs" / "retrieval_results",
    "rag_results":        BASE / "outputs" / "rag_results",
    "evaluation_results": BASE / "outputs" / "evaluation_results",
    "charts":             BASE / "outputs" / "charts",
    # ── Logs ─────────────────────────────────────────────────────────────────
    "logs":               BASE / "logs",
}


def create_all_directories() -> None:
    """Create all pipeline directories if they do not already exist.

    Called once at notebook startup. Uses parents=True so intermediate
    directories are also created. Existing directories are left untouched.
    """
    for path in PATHS.values():
        path.mkdir(parents=True, exist_ok=True)
