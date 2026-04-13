"""Intelligent chunking strategies for corporate strategy documents.

Strategy documents have specific structural patterns — section headers, bullet
lists, tables, executive summaries — that benefit from structure-aware splitting
rather than naive character counts.  This module layers a hierarchy of splitters
so that logical sections are preserved whenever possible, falling back to
recursive character splitting for unstructured prose.

Documents are chunked concurrently using a thread pool for faster ingestion.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from strategy_agent.config import Settings, get_settings

logger = logging.getLogger(__name__)


def _build_recursive_splitter(settings: Settings) -> RecursiveCharacterTextSplitter:
    """Return a general-purpose recursive splitter tuned for strategy prose."""
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=[
            "\n\n\n",   # triple newline (section breaks)
            "\n\n",      # paragraph breaks
            "\n",         # line breaks
            ". ",         # sentence boundaries
            ", ",         # clause boundaries
            " ",          # word boundaries
        ],
        length_function=len,
        is_separator_regex=False,
    )


def _build_markdown_splitter() -> MarkdownHeaderTextSplitter:
    """Split Markdown files on heading boundaries first."""
    return MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "heading_1"),
            ("##", "heading_2"),
            ("###", "heading_3"),
        ],
    )


def _chunk_single_doc(
    doc: Document,
    recursive: RecursiveCharacterTextSplitter,
    md_splitter: MarkdownHeaderTextSplitter,
) -> list[Document]:
    """Chunk a single document (safe to call from a thread)."""
    is_markdown = doc.metadata.get("file_type") in (".md",)

    if is_markdown:
        # Phase 1: heading-aware split
        heading_chunks = md_splitter.split_text(doc.page_content)
        # Propagate original metadata into heading chunks
        for hc in heading_chunks:
            hc.metadata = {**doc.metadata, **hc.metadata}
        # Phase 2: enforce size limit
        return recursive.split_documents(heading_chunks)

    return recursive.split_documents([doc])


def chunk_documents(
    docs: list[Document],
    settings: Settings | None = None,
    *,
    max_workers: int = 4,
) -> list[Document]:
    """Split *docs* into retrieval-friendly chunks.

    Markdown-sourced documents are first split on heading boundaries so each
    section stays together; everything then passes through the recursive
    character splitter so no chunk exceeds the configured token budget.

    Documents are chunked concurrently for faster processing on large corpora.
    """
    s = settings or get_settings()
    recursive = _build_recursive_splitter(s)
    md_splitter = _build_markdown_splitter()

    if len(docs) <= 1:
        # No benefit from threading for a single document
        chunks = _chunk_single_doc(docs[0], recursive, md_splitter) if docs else []
    else:
        workers = min(max_workers, len(docs))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            chunk_lists = list(
                executor.map(
                    lambda doc: _chunk_single_doc(doc, recursive, md_splitter),
                    docs,
                )
            )
        chunks = [c for sublist in chunk_lists for c in sublist]

    logger.info(
        "Chunked %d document(s) → %d chunks (target %d chars, %d overlap)",
        len(docs),
        len(chunks),
        s.chunk_size,
        s.chunk_overlap,
    )
    return chunks
