"""Corpus search tool — lets the research agent query the vector store.

This wraps the ChromaDB retriever as a LangChain ``Tool`` so it can be bound
to an agent that decides *what* to search for rather than relying on a single
static query derived from the brief.
"""

from __future__ import annotations

import threading

from langchain_core.tools import tool

from strategy_agent.memory.vector_store import CorpusStore

# Module-level store instance, lazily initialised on first call.
# Protected by _lock for thread safety.
_lock = threading.Lock()
_store: CorpusStore | None = None


def _get_store() -> CorpusStore:
    global _store
    with _lock:
        if _store is None:
            _store = CorpusStore()
        return _store


def set_store(store: CorpusStore) -> None:
    """Allow external code (e.g. the orchestrator) to inject a store."""
    global _store
    with _lock:
        _store = store


@tool
def search_corpus(query: str) -> str:
    """Search the corporate strategy document corpus for information relevant to the query.

    Use this tool to find specific data points, strategic precedents, market
    context, competitive intelligence, or any other information contained in
    the organization's strategy documents.  Formulate precise, targeted queries
    for best results.

    Args:
        query: A natural-language search query describing the information needed.

    Returns:
        Concatenated passages from the most relevant corpus documents, each
        prefixed with its source file name.
    """
    store = _get_store()
    docs = store.similarity_search(query, k=8)
    if not docs:
        return "No relevant passages found in the corpus for this query."

    passages: list[str] = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "unknown")
        passages.append(f"[{i}] (Source: {source})\n{doc.page_content}")

    return "\n\n---\n\n".join(passages)


@tool
def search_corpus_with_context(query: str, context: str) -> str:
    """Search the corpus with additional context to improve relevance.

    Use this when a simple keyword search isn't enough — provide both
    the specific query and broader context about *why* you need this
    information so the semantic search can find tangentially related passages.

    Args:
        query: The specific information needed.
        context: Broader context about why this information is relevant.

    Returns:
        Concatenated relevant passages with source attribution.
    """
    combined = f"{query} — Context: {context}"
    store = _get_store()
    docs = store.similarity_search(combined, k=10)
    if not docs:
        return "No relevant passages found in the corpus."

    passages: list[str] = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "unknown")
        passages.append(f"[{i}] (Source: {source})\n{doc.page_content}")

    return "\n\n---\n\n".join(passages)
