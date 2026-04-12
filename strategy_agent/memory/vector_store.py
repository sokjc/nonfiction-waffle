"""ChromaDB-backed vector store for the strategy document corpus.

This module owns the lifecycle of the persistent Chroma collection — creating
it, adding document chunks, and exposing a LangChain ``VectorStoreRetriever``
that the research agent can plug straight into a retrieval chain.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from strategy_agent.config import Settings, get_settings
from strategy_agent.models import build_embeddings

logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


class CorpusStore:
    """Thin wrapper around a persistent Chroma collection."""

    def __init__(self, settings: Settings | None = None, embeddings: Embeddings | None = None):
        self._settings = settings or get_settings()
        self._embeddings = embeddings or build_embeddings(self._settings)
        self._persist_dir = _ensure_dir(self._settings.chroma_persist_dir)
        self._collection_name = self._settings.chroma_collection

        self._store = Chroma(
            collection_name=self._collection_name,
            embedding_function=self._embeddings,
            persist_directory=str(self._persist_dir),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def add_documents(self, docs: list[Document]) -> int:
        """Embed and persist *docs*.  Returns the number of chunks stored."""
        if not docs:
            return 0
        self._store.add_documents(docs)
        logger.info("Stored %d chunks in collection '%s'", len(docs), self._collection_name)
        return len(docs)

    def as_retriever(self, top_k: int | None = None) -> VectorStoreRetriever:
        """Return a LangChain retriever over the corpus."""
        k = top_k or self._settings.retrieval_top_k
        return self._store.as_retriever(search_kwargs={"k": k})

    def similarity_search(self, query: str, k: int = 8) -> list[Document]:
        """Direct similarity search — useful inside agent tools."""
        return self._store.similarity_search(query, k=k)

    @property
    def count(self) -> int:
        """Number of chunks currently stored.

        Note: Chroma's public Python API doesn't expose ``count()`` directly
        on the LangChain wrapper — ``_collection`` is the underlying
        ``chromadb.Collection`` object, whose ``count()`` is a stable API.
        """
        return self._store._collection.count()

    def reset(self) -> None:
        """Delete the collection and recreate it (destructive)."""
        self._store.delete_collection()
        self._store = Chroma(
            collection_name=self._collection_name,
            embedding_function=self._embeddings,
            persist_directory=str(self._persist_dir),
        )
        logger.warning("Collection '%s' reset", self._collection_name)
