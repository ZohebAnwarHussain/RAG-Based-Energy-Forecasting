"""Unit tests for config/paths.py, config/models.py, config/pipeline.py.

Verifies directory layout, model constants, experiment defaults, and
season mapping are correctly defined.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.paths import PATHS, BASE
from config.models import MODELS, EXP_DEFAULTS
from config.pipeline import SEASON_MAP, MAX_SUMMARIES_PER_TYPE


# -- config/paths.py ----------------------------------------------------------

def test_base_is_path():
    """BASE must be a pathlib.Path instance."""
    assert isinstance(BASE, Path)


def test_paths_dict_has_required_keys():
    """PATHS must contain all pipeline directory keys."""
    required = {
        "data_gefcom", "data_household", "summaries_csv",
        "golden_dataset", "faiss_index", "indexes",
    }
    assert required.issubset(set(PATHS.keys()))


def test_paths_values_are_path_objects():
    """Every value in PATHS must be a pathlib.Path."""
    for key, path in PATHS.items():
        assert isinstance(path, Path), f"PATHS['{key}'] is not a Path"


def test_faiss_index_under_indexes():
    """FAISS index path must be a child of the indexes directory."""
    assert str(PATHS["faiss_index"]).startswith(str(PATHS["indexes"]))


# -- config/models.py ---------------------------------------------------------

def test_models_dict_has_required_keys():
    """MODELS must define all model identifiers."""
    required = {"gemini_kb", "gemini_gd", "embedding", "groq_rag", "groq_judge"}
    assert required.issubset(set(MODELS.keys()))


def test_embedding_model_is_minilm():
    """Embedding model must be all-MiniLM-L6-v2 as specified by supervisor."""
    assert "all-MiniLM-L6-v2" in MODELS["embedding"]


def test_groq_rag_model_is_llama():
    """RAG model must be llama-3.3-70b-versatile as specified by supervisor."""
    assert "llama-3.3-70b" in MODELS["groq_rag"]


def test_exp_defaults_has_required_keys():
    """EXP_DEFAULTS must define all experiment parameters."""
    required = {"top_k_values", "temperature", "max_tokens", "min_docs"}
    assert required.issubset(set(EXP_DEFAULTS.keys()))


def test_top_k_values_are_3_5_10():
    """Default K values must be [3, 5, 10]."""
    assert EXP_DEFAULTS["top_k_values"] == [3, 5, 10]


def test_temperature_is_zero():
    """Temperature must be 0 for deterministic generation."""
    assert EXP_DEFAULTS["temperature"] == 0


def test_max_tokens_is_500():
    """Max tokens must be 500."""
    assert EXP_DEFAULTS["max_tokens"] == 500


def test_min_docs_is_200():
    """Minimum dataset size must be 200 per supervisor requirement."""
    assert EXP_DEFAULTS["min_docs"] == 200


# -- config/pipeline.py -------------------------------------------------------

def test_season_map_has_12_months():
    """SEASON_MAP must map all 12 calendar months."""
    assert set(SEASON_MAP.keys()) == set(range(1, 13))


def test_season_map_values_are_valid():
    """All season names must be one of the four meteorological seasons."""
    valid = {"Winter", "Spring", "Summer", "Autumn"}
    assert set(SEASON_MAP.values()) == valid


def test_december_is_winter():
    """December must be mapped to Winter."""
    assert SEASON_MAP[12] == "Winter"


def test_june_is_summer():
    """June must be mapped to Summer."""
    assert SEASON_MAP[6] == "Summer"


def test_max_summaries_per_type():
    """MAX_SUMMARIES_PER_TYPE must be a positive integer."""
    assert isinstance(MAX_SUMMARIES_PER_TYPE, int)
    assert MAX_SUMMARIES_PER_TYPE > 0
