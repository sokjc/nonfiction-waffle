"""LlamaIndex-backed vector store for the strategy document corpus.

This module owns the lifecycle of the persistent LlamaIndex index — creating
it, adding document chunks, and exposing both chunk-level retrieval and
hybrid document-level context stuffing.

Hybrid retrieval strategy:
1. **Chunk retrieval**: Find top-k relevant chunks via embedding similarity
2. **Document expansion**: Identify the source documents those chunks came from
3. **Context stuffing**: Include the full text of the top N source documents
   in the LLM context window (leveraging modern 128K+ context models)

This gives the LLM both precision (relevant chunks) and completeness
(full document context around those chunks).
"""

from __future__ import annotations

import logging
from pathlib import Path

from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core import Settings as LISettings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from strategy_agent.config import Settings, get_settings
from strategy_agent.models import build_embeddings

logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


class CorpusStore:
    """LlamaIndex vector store with persistent storage and hybrid retrieval."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._persist_dir = _ensure_dir(self._settings.index_persist_dir)
        self._embed_model = build_embeddings(self._settings)

        # Configure LlamaIndex global settings
        LISettings.embed_model = self._embed_model
        LISettings.chunk_size = self._settings.chunk_size
        LISettings.chunk_overlap = self._settings.chunk_overlap

        self._index = self._load_or_create()

        # Tracks full source document text for context stuffing.
        # Keyed by source_file name.
        self._source_documents: dict[str, str] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def _load_or_create(self) -> VectorStoreIndex:
        """Load an existing index from disk, or create an empty one."""
        docstore_path = self._persist_dir / "docstore.json"
        if docstore_path.exists():
            logger.info("Loading existing index from %s", self._persist_dir)
            storage_context = StorageContext.from_defaults(
                persist_dir=str(self._persist_dir),
            )
            return load_index_from_storage(
                storage_context, embed_model=self._embed_model,
            )

        logger.info("Creating new index at %s", self._persist_dir)
        return VectorStoreIndex([], embed_model=self._embed_model)

    def _persist(self) -> None:
        """Save the index to disk."""
        self._index.storage_context.persist(persist_dir=str(self._persist_dir))

    # ── Ingestion ─────────────────────────────────────────────────────────

    def add_documents(self, docs: list) -> int:
        """Index a list of LangChain-style Document objects.

        Converts them to LlamaIndex Documents, chunks via SentenceSplitter,
        and stores the full source text for later context stuffing.
        """
        if not docs:
            return 0

        # Convert LangChain Documents to LlamaIndex Documents
        li_docs = []
        for doc in docs:
            source_file = doc.metadata.get("source_file", "unknown")
            li_doc = Document(
                text=doc.page_content,
                metadata={
                    "source_file": source_file,
                    "source_path": doc.metadata.get("source_path", ""),
                    "file_type": doc.metadata.get("file_type", ""),
                },
            )
            li_docs.append(li_doc)

            # Cache full text for context stuffing (concatenate by source)
            if source_file not in self._source_documents:
                self._source_documents[source_file] = ""
            self._source_documents[source_file] += doc.page_content + "\n\n"

        # Insert into index (LlamaIndex handles chunking internally)
        for doc in li_docs:
            self._index.insert(doc)

        self._persist()
        logger.info("Indexed %d document(s), persisted to %s", len(li_docs), self._persist_dir)
        return len(li_docs)

    # ── Retrieval ─────────────────────────────────────────────────────────

    def similarity_search(self, query: str, k: int = 8) -> list[dict]:
        """Return top-k chunk-level results as dicts with text and metadata."""
        retriever = self._index.as_retriever(similarity_top_k=k)
        nodes = retriever.retrieve(query)
        return [
            {
                "text": node.get_text(),
                "score": node.get_score(),
                "source_file": node.metadata.get("source_file", "unknown"),
            }
            for node in nodes
        ]

    def hybrid_retrieve(self, query: str, settings: Settings | None = None) -> dict:
        """Hybrid retrieval: chunk search → document expansion → context stuffing.

        Returns a dict with:
        - ``chunks``: The top-k most relevant chunks (for citation)
        - ``full_documents``: Full text of the top N source documents
          (for long-context stuffing into the LLM)
        - ``source_files``: Ordered list of source document names used
        """
        s = settings or self._settings

        # Step 1: Chunk-level retrieval
        chunks = self.similarity_search(query, k=s.retrieval_top_k)

        # Step 2: Rank source documents by frequency + relevance score
        doc_scores: dict[str, float] = {}
        for chunk in chunks:
            src = chunk["source_file"]
            doc_scores[src] = doc_scores.get(src, 0) + chunk["score"]

        ranked_sources = sorted(doc_scores, key=doc_scores.get, reverse=True)

        # Step 3: Expand top N sources to full document text
        n = s.context_stuffing_docs
        full_docs: list[dict] = []
        for src in ranked_sources[:n]:
            full_text = self._source_documents.get(src)
            if full_text:
                full_docs.append({"source_file": src, "text": full_text.strip()})

        return {
            "chunks": chunks,
            "full_documents": full_docs,
            "source_files": ranked_sources[:n],
        }

    # ── Inspection ──────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        """Number of nodes currently indexed."""
        try:
            docstore = self._index.storage_context.docstore
            return len(docstore.docs)
        except Exception:
            return 0

    def get_source_files(self) -> list[str]:
        """Return a sorted list of unique source_file names in the index."""
        sources: set[str] = set()
        try:
            docstore = self._index.storage_context.docstore
            for node in docstore.docs.values():
                src = node.metadata.get("source_file")
                if src:
                    sources.add(src)
        except Exception:
            pass
        return sorted(sources)

    def count_by_source(self) -> dict[str, int]:
        """Return ``{source_file: node_count}`` for every source document."""
        counts: dict[str, int] = {}
        try:
            docstore = self._index.storage_context.docstore
            for node in docstore.docs.values():
                src = node.metadata.get("source_file", "unknown")
                counts[src] = counts.get(src, 0) + 1
        except Exception:
            pass
        return dict(sorted(counts.items()))

    # ── Deletion ─────────────────────────────────────────────────────────

    def remove_document(self, source_file: str) -> int:
        """Remove all indexed nodes belonging to *source_file*.

        Returns the number of nodes removed.
        """
        docstore = self._index.storage_context.docstore
        node_ids_to_remove = [
            nid
            for nid, node in docstore.docs.items()
            if node.metadata.get("source_file") == source_file
        ]
        if not node_ids_to_remove:
            logger.info("No nodes found for source_file=%r", source_file)
            return 0

        for nid in node_ids_to_remove:
            self._index.delete_ref_doc(nid, delete_from_docstore=True)

        self._source_documents.pop(source_file, None)
        self._persist()
        logger.info("Removed %d node(s) for %r", len(node_ids_to_remove), source_file)
        return len(node_ids_to_remove)

    def deduplicate(self) -> int:
        """Remove duplicate nodes that share the same text and source_file.

        Keeps the first occurrence (by node ID sort order) and deletes the rest.
        Returns the number of duplicates removed.
        """
        docstore = self._index.storage_context.docstore
        seen: dict[tuple[str, str], str] = {}  # (source_file, text_hash) → kept node_id
        duplicates: list[str] = []

        for nid in sorted(docstore.docs):
            node = docstore.docs[nid]
            import hashlib

            text_hash = hashlib.sha256(node.get_content().encode()).hexdigest()
            key = (node.metadata.get("source_file", ""), text_hash)
            if key in seen:
                duplicates.append(nid)
            else:
                seen[key] = nid

        if not duplicates:
            logger.info("No duplicate nodes found")
            return 0

        for nid in duplicates:
            self._index.delete_ref_doc(nid, delete_from_docstore=True)

        self._persist()
        logger.info("Removed %d duplicate node(s)", len(duplicates))
        return len(duplicates)

    # ── Reset ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Delete the index and recreate it (destructive)."""
        import shutil

        if self._persist_dir.exists():
            shutil.rmtree(self._persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._index = VectorStoreIndex([], embed_model=self._embed_model)
        self._source_documents.clear()
        logger.warning("Index reset at %s", self._persist_dir)
