"""LLM-based entity and relationship extraction for the knowledge graph.

Uses the configured LLM to extract (subject, predicate, object) triples
from text chunks during corpus ingestion.  The prompt is tuned for
corporate strategy documents — companies, markets, products, metrics.

Supports batch extraction via :func:`extract_triples_batch` which runs
multiple LLM calls concurrently using a thread pool.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM = """\
You are an expert at extracting structured knowledge from corporate \
strategy documents.  Your job is to identify entity-relationship triples.

Focus on:
- **Organizations**: companies, divisions, competitors, partners
- **Markets & geographies**: regions, segments, industries
- **Products & services**: product lines, offerings, platforms
- **Metrics**: revenue, market share, growth rates, margins
- **Strategies**: initiatives, plans, decisions, investments
- **People**: executives, board members, key hires
- **Relationships**: competes with, operates in, acquired, partnered with, \
  enables, depends on, threatens, grew by, invested in

Keep entity names normalized (e.g. always "Apple Inc." not sometimes \
"Apple" and sometimes "AAPL").  Only extract triples clearly supported \
by the text.  Return at most 20 triples per chunk.
"""

EXTRACTION_HUMAN = """\
Extract knowledge triples from this text:

{text}

Return ONLY a JSON array of objects, each with "subject", "predicate", \
and "object" keys.  No other text.
"""

extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", EXTRACTION_SYSTEM),
    ("human", EXTRACTION_HUMAN),
])


def extract_triples(text: str, llm: ChatOpenAI) -> list[tuple[str, str, str]]:
    """Extract entity-relationship triples from *text* using an LLM.

    Returns a list of (subject, predicate, object) tuples.
    Gracefully returns an empty list if parsing fails.
    """
    chain = extraction_prompt | llm | StrOutputParser()

    try:
        raw = chain.invoke({"text": text})

        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            first_nl = cleaned.index("\n")
            cleaned = cleaned[first_nl + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        data = json.loads(cleaned)
        if isinstance(data, list):
            keys = ("subject", "predicate", "object")
            return [
                (t["subject"].strip(), t["predicate"].strip(), t["object"].strip())
                for t in data
                if isinstance(t, dict)
                and all(k in t for k in keys)
                and all(isinstance(t[k], str) and t[k].strip() for k in keys)
            ]
    except Exception as e:
        logger.debug("Triple extraction failed: %s", e)

    return []


def extract_triples_batch(
    texts: list[str],
    llm: ChatOpenAI,
    *,
    max_workers: int = 10,
    on_progress: callable | None = None,
) -> list[list[tuple[str, str, str]]]:
    """Extract triples from multiple texts concurrently.

    Args:
        texts: List of text chunks to extract triples from.
        llm: The LLM to use for extraction.
        max_workers: Maximum number of concurrent LLM calls.
        on_progress: Optional callback ``(completed, total)`` called after
            each chunk finishes.

    Returns:
        A list of triple-lists, one per input text, in the same order as *texts*.
    """
    total = len(texts)
    results: list[list[tuple[str, str, str]]] = [[] for _ in range(total)]
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(extract_triples, text, llm): idx
            for idx, text in enumerate(texts)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.warning("Triple extraction failed for chunk %d: %s", idx, e)
                results[idx] = []
            completed += 1
            if on_progress:
                on_progress(completed, total)

    return results
