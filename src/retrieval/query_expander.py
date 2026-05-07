"""
src/retrieval/query_expander.py
================================
LLM-based Query Expansion for EXP_02 and EXP_03.

WHY THIS EXISTS
---------------
Your golden queries are conceptual:
    "What was the average daily load across GEFCom zones?"

Your KB documents are statistical summaries keyed by zone + date:
    "gefcom_daily_zone_4_2004-01-07: mean=1842.3 MW, peak=2103.0 MW..."

These live in different semantic spaces. The bi-encoder embeds them
independently and often fails to bridge the gap — hence Recall@K of
0.035–0.059 (should be 0.3+).

Query expansion generates 2–3 reformulations of each query before
retrieval, fetches results for all variants, then unions them. This
dramatically increases the chance that at least one formulation
triggers the right KB documents.

STRATEGY
---------
Two complementary reformulations per query:

  1. Keyword form — extract the core domain terms, strip filler words.
     "What was the average daily electricity load across GEFCom zones?"
     → "GEFCom zone average daily load mean MW"

  2. Paraphrase — rephrase with different vocabulary to catch synonyms.
     "What was the average daily electricity load across GEFCom zones?"
     → "Typical daily electricity consumption per zone GEFCom dataset"

The union of (original + keyword + paraphrase) hits a much wider
semantic surface in the KB with no loss of precision (reranker handles
precision downstream).

EXPECTED IMPACT
---------------
  Recall@K:    0.035 → ~0.10–0.15 (3–4× improvement expected)
  MRR:         improvement from richer candidate pool
  Precision:   unchanged by expansion alone — handled by reranker

USAGE
-----
    expander = QueryExpander(groq_client)
    variants = expander.expand(query)
    # → ['original query', 'keyword form', 'paraphrase']

    # Retrieve for each variant, union results
    all_docs = expander.expand_and_retrieve(query, retriever, top_k=3)
    # → deduplicated list of docs from all variant retrievals
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)

# Prompt shown to the LLM to generate query variants
_EXPANSION_SYSTEM = (
    "You are a query reformulation assistant for an energy demand forecasting "
    "knowledge base. The KB contains daily and weekly statistical summaries of "
    "electricity load data from GEFCom zones and household power meters."
)

_EXPANSION_PROMPT = """\
Given the following energy domain query, produce exactly 2 reformulations:
1. KEYWORD: A short keyword-style search query (6-12 words) using domain terms
   like zone IDs, load/MW/demand, time periods, dataset names.
2. PARAPHRASE: A different phrasing of the same question using synonyms.

Original query: {query}

Respond in this exact format (no extra text):
KEYWORD: <keyword query here>
PARAPHRASE: <paraphrase here>"""


class QueryExpander:
    """
    Generates query variants using an LLM to improve retrieval recall.

    Uses your existing RotatingGroqClient so no new API key or model
    is needed — the same llama-3.3-70b-versatile call used for generation.

    Args:
        groq_client: Your RotatingGroqClient instance.
        model:       Groq model to use. Defaults to llama-3.3-70b-versatile.
        enabled:     Set to False to disable expansion (pass-through mode).
                     Useful for ablation — run with and without expansion
                     to measure the delta.

    Example:
        >>> expander = QueryExpander(groq_client)
        >>> variants = expander.expand("What was Zone 4 peak demand in winter 2006?")
        >>> # ['What was Zone 4 peak demand in winter 2006?',
        >>> #  'Zone 4 peak load winter 2006 MW GEFCom',
        >>> #  'What was the maximum electricity demand in Zone 4 during winter 2006?']
    """

    def __init__(
        self,
        groq_client: Any,
        model: str = "llama-3.3-70b-versatile",
        enabled: bool = True,
    ) -> None:
        self._groq = groq_client
        self._model = model
        self.enabled = enabled
        logger.info(
            "QueryExpander initialised (enabled=%s, model=%s).",
            enabled, model,
        )

    def expand(self, query: str) -> List[str]:
        """
        Expand a query into a list of variants including the original.

        Args:
            query: The original user query string.

        Returns:
            List of query strings: [original, keyword_form, paraphrase].
            Falls back to [original] alone if the LLM call fails — the
            experiment continues without expansion rather than crashing.
        """
        if not self.enabled:
            return [query]

        prompt = _EXPANSION_PROMPT.format(query=query)
        messages = [
            {"role": "system", "content": _EXPANSION_SYSTEM},
            {"role": "user",   "content": prompt},
        ]

        try:
            response = self._groq.chat(
                messages=messages,
                model=self._model,
                temperature=0,
                max_tokens=150,
            )
            raw = response.choices[0].message.content.strip()
            variants = self._parse_variants(raw, query)
            logger.debug(
                "Query expanded: '%s...' → %d variants.",
                query[:40], len(variants),
            )
            return variants

        except Exception as exc:
            logger.warning(
                "Query expansion failed for '%s...': %s — using original only.",
                query[:40], exc,
            )
            return [query]

    def expand_and_retrieve(
        self,
        query: str,
        retrieve_fn: Callable[[str], List[Any]],
        top_k: int,
        dedup: bool = True,
    ) -> List[Any]:
        """
        Expand the query, retrieve for each variant, union the results.

        Each variant is retrieved independently using retrieve_fn.
        Results are deduplicated by row_id / source metadata to avoid
        the same document appearing multiple times in the candidate pool.

        The returned list is *larger* than top_k — it is the union of
        all variant retrievals. Pass it to CrossEncoderReranker.rerank()
        to reduce back to top_k with optimal ranking.

        Args:
            query:       The original user query string.
            retrieve_fn: A callable that takes a query string and returns
                         a list of document dicts (with 'row_id' key)
                         or LangChain Document objects.
                         Example: lambda q: retriever.retrieve(q)
            top_k:       Used only for logging. The actual fetch size
                         per variant is controlled by the retriever.
            dedup:       Whether to deduplicate by row_id. Default True.

        Returns:
            Union of all variant retrievals, deduplicated.
        """
        variants = self.expand(query)
        seen_ids: set = set()
        all_docs: list = []

        for variant in variants:
            docs = retrieve_fn(variant)
            for doc in docs:
                # Handle both dict format (HybridRetriever) and
                # LangChain Document format (DenseRetriever)
                if isinstance(doc, dict):
                    doc_id = doc.get("row_id") or doc.get("id") or ""
                else:
                    doc_id = (
                        doc.metadata.get("row_id")
                        or doc.metadata.get("source")
                        or ""
                    )

                if dedup and doc_id and doc_id in seen_ids:
                    continue
                if doc_id:
                    seen_ids.add(doc_id)
                all_docs.append(doc)

        logger.info(
            "expand_and_retrieve: query='%s...' | %d variants | "
            "%d unique docs retrieved (target k=%d).",
            query[:40], len(variants), len(all_docs), top_k,
        )
        return all_docs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_variants(self, raw: str, original: str) -> List[str]:
        """
        Parse the LLM's KEYWORD/PARAPHRASE response into a list of strings.
        Always includes the original query as the first element.
        Falls back gracefully if parsing fails.
        """
        variants = [original]
        lines = raw.strip().splitlines()

        for line in lines:
            line = line.strip()
            if line.upper().startswith("KEYWORD:"):
                kw = line[len("KEYWORD:"):].strip()
                if kw and kw != original:
                    variants.append(kw)
            elif line.upper().startswith("PARAPHRASE:"):
                pp = line[len("PARAPHRASE:"):].strip()
                if pp and pp != original:
                    variants.append(pp)

        # If parsing found nothing useful, return original only
        if len(variants) == 1:
            logger.warning(
                "Could not parse expansion variants from LLM response: %s",
                raw[:100],
            )

        return variants
