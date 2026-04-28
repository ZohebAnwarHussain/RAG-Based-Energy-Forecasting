"""LangChain CSVLoader integration for loading KB summaries as Documents.

This module uses LangChain's CSVLoader to load the master KB CSV into
Document objects. CSVLoader is the standard LangChain approach for
converting structured CSV data into retrievable documents.

Challenge:
    By default, CSVLoader puts ALL columns into page_content as a
    key-value dump (e.g. "kb_id: 42\nrow_id: gefcom_daily_4_...").
    This pollutes the embedding space — the embedding model would
    encode metadata strings alongside actual summary text, degrading
    retrieval quality.

Solution:
    CSVLoader's ``content_columns`` parameter restricts page_content
    to only the ``summary`` column. All other columns are routed to
    ``metadata_columns`` so they appear in doc.metadata for filtered
    retrieval via ChromaDB.

    If the installed LangChain version does not support content_columns,
    the fallback approach loads all columns and post-processes each
    Document to extract only the summary text into page_content.

Why CSVLoader over pd.read_csv():
    1. LangChain standard — reviewers and supervisors expect this pattern
    2. Produces Document objects that plug directly into
       Chroma.from_documents() and FAISS.from_documents()
    3. The source_column parameter sets doc.metadata["source"] to row_id,
       which maps directly to expected_summary_ids in the golden dataset
       for RAGAS Context Recall evaluation
    4. Metadata dict on each Document powers ChromaDB's filtered retrieval
       in Pipeline 2 (hybrid) and Pipeline 3 (hierarchical)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Metadata columns stored on each Document for ChromaDB filtered retrieval
METADATA_COLUMNS: List[str] = [
    "kb_id",
    "row_id",
    "dataset",
    "granularity",
    "zone_id",
    "year",
    "month",
    "season",
    "parent_id",
]


def load_kb_documents(csv_path: Path) -> List[Document]:
    """Load KB summaries as LangChain Document objects using CSVLoader.

    Each row of combined_master_summaries.csv becomes a Document with:

        page_content: The summary text only (clean, no metadata noise)
        metadata: {
            "source":      row_id (for RAGAS Context Recall matching),
            "row":         original CSV row number,
            "kb_id":       sequential integer,
            "row_id":      e.g. "gefcom_daily_4_2004-01-01",
            "dataset":     "gefcom" or "household",
            "granularity": "daily", "weekly", etc.,
            "zone_id":     zone identifier or "" for household,
            "year":        calendar year or "",
            "month":       month number or "",
            "season":      season name or "",
            "parent_id":   parent row_id for hierarchical retrieval,
        }

    The ``source`` metadata field is set to ``row_id`` via CSVLoader's
    ``source_column`` parameter. This means retrieved documents can be
    directly compared against ``expected_summary_ids`` in the golden
    dataset during RAGAS evaluation — no ID translation needed.

    Args:
        csv_path: Path to combined_master_summaries.csv.

    Returns:
        List of Document objects, one per KB summary row.
        Length matches the number of valid rows in the CSV (~480).

    Raises:
        FileNotFoundError: If the CSV file does not exist.
            Run notebooks/01_kb_generation.ipynb first.

    Example:
        >>> docs = load_kb_documents(
        ...     Path("outputs/knowledge_base/generated_summaries/csv/"
        ...          "combined_master_summaries.csv")
        ... )
        >>> len(docs)
        480
        >>> docs[0].page_content[:50]
        'On January 1st 2004, Zone 4 saw an average hourly'
        >>> docs[0].metadata["source"]
        'gefcom_daily_4_2004-01-01'
        >>> docs[0].metadata["dataset"]
        'gefcom'
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"KB CSV not found at {csv_path}. "
            "Run notebooks/01_kb_generation.ipynb first."
        )

    logger.info(
        "Loading KB documents from %s using LangChain CSVLoader.",
        csv_path.name,
    )

    # ── Attempt 1: Use content_columns if supported ──────────────────────────
    # content_columns restricts page_content to only the summary column.
    # metadata_columns routes all other fields to doc.metadata.
    try:
        loader = CSVLoader(
            file_path=str(csv_path),
            source_column="row_id",
            content_columns=["summary"],
            metadata_columns=METADATA_COLUMNS,
            encoding="utf-8",
        )
        documents = loader.load()
        logger.info(
            "CSVLoader loaded %d documents using content_columns='summary'.",
            len(documents),
        )
        return _clean_documents(documents)

    except TypeError:
        # content_columns not supported in this LangChain version
        logger.info(
            "content_columns not supported — falling back to "
            "post-processing approach."
        )

    # ── Attempt 2: Fallback — load all columns, then post-process ────────────
    # CSVLoader puts all columns into page_content as key-value pairs.
    # We extract only the summary text and move everything else to metadata.
    loader = CSVLoader(
        file_path=str(csv_path),
        source_column="row_id",
        encoding="utf-8",
    )
    raw_documents = loader.load()

    logger.info(
        "CSVLoader loaded %d raw documents. Post-processing to extract "
        "summary text into page_content.",
        len(raw_documents),
    )

    documents = _extract_summary_from_page_content(raw_documents)
    return _clean_documents(documents)


def _extract_summary_from_page_content(
    documents: List[Document],
) -> List[Document]:
    """Extract only the summary field from CSVLoader's key-value page_content.

    When CSVLoader does not support content_columns, it puts all CSV
    columns into page_content as:

        "kb_id: 42\\nrow_id: gefcom_daily_4_...\\n...\\nsummary: On Jan 1st..."

    This function parses that text to extract only the summary value
    and moves all other fields into metadata.

    Args:
        documents: Raw Document objects from CSVLoader where page_content
            contains all CSV columns as key-value pairs.

    Returns:
        Cleaned Document objects where page_content contains only the
        summary text and metadata contains all other fields.
    """
    cleaned: List[Document] = []

    for doc in documents:
        content = doc.page_content
        metadata = dict(doc.metadata)

        # Parse key-value pairs from page_content
        summary_text = ""
        for line in content.split("\n"):
            if ": " in line:
                key, _, value = line.partition(": ")
                key = key.strip()
                value = value.strip()

                if key == "summary":
                    summary_text = value
                elif key in METADATA_COLUMNS and key not in metadata:
                    metadata[key] = value

        if not summary_text:
            # If parsing failed, keep original page_content
            summary_text = content

        cleaned.append(
            Document(page_content=summary_text, metadata=metadata)
        )

    return cleaned


def _clean_documents(documents: List[Document]) -> List[Document]:
    """Remove documents with empty page_content and ensure metadata consistency.

    Filters out any documents where page_content is empty or whitespace-only.
    Ensures all metadata values are strings (ChromaDB requires string values).

    Args:
        documents: List of Document objects to clean.

    Returns:
        Cleaned list with empty documents removed and metadata normalised.
    """
    cleaned: List[Document] = []
    skipped = 0

    for doc in documents:
        if not doc.page_content or not doc.page_content.strip():
            skipped += 1
            continue

        # ChromaDB requires all metadata values to be strings
        for key in list(doc.metadata.keys()):
            if doc.metadata[key] is None:
                doc.metadata[key] = ""
            else:
                doc.metadata[key] = str(doc.metadata[key])

        cleaned.append(doc)

    if skipped > 0:
        logger.warning(
            "Removed %d documents with empty page_content.", skipped
        )

    logger.info(
        "Final document count: %d. "
        "page_content = summary text only. "
        "metadata fields: %s.",
        len(cleaned),
        list(cleaned[0].metadata.keys()) if cleaned else "N/A",
    )

    return cleaned
