"""Setup configuration for the LJMU thesis RAG package.

Enables editable installation via:
    pip install -e .

This allows clean imports like:
    from src.knowledge_base import generate_summaries
    from config import PATHS, MAX_SUMMARIES_PER_TYPE

regardless of which directory Python is run from.
"""

from setuptools import find_packages, setup

setup(
    name="ljmu-rag-energy-forecasting",
    version="0.1.0",
    description=(
        "RAG-based energy demand forecasting and natural language insight "
        "generation. MSc Thesis, Liverpool John Moores University, 2026."
    ),
    author="Zoheb Anwar Hussain",
    author_email="z.a.hussain@2026.ljmu.ac.uk",
    url="https://github.com/ZohebAnwarHussain/RAG-Based-Energy-Forecasting",
    packages=find_packages(include=["src", "src.*", "config", "config.*"]),
    python_requires=">=3.11",
    install_requires=[],  # Runtime dependencies are managed separately
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
