"""Unified document loader for the strategy corpus.

Supports PDF, DOCX, Markdown, plain text, and HTML.  Each loader normalises
its output into LangChain ``Document`` objects with consistent metadata so
downstream chunkers and retrievers can treat every source identically.

File loading is parallelized with a thread pool for faster corpus ingestion.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Map of file extension → (loader class path, extra kwargs).
# Lazy-imported so optional heavy deps are only pulled in when needed.
_LOADER_REGISTRY: dict[str, tuple[str, dict]] = {
    ".pdf": ("langchain_community.document_loaders.PyPDFLoader", {}),
    ".docx": ("langchain_community.document_loaders.Docx2txtLoader", {}),
    ".doc": ("langchain_community.document_loaders.Docx2txtLoader", {}),
    ".md": ("langchain_community.document_loaders.TextLoader", {"encoding": "utf-8"}),
    ".txt": ("langchain_community.document_loaders.TextLoader", {"encoding": "utf-8"}),
    ".html": ("langchain_community.document_loaders.BSHTMLLoader", {}),
    ".htm": ("langchain_community.document_loaders.BSHTMLLoader", {}),
    ".csv": ("langchain_community.document_loaders.CSVLoader", {}),
    ".pptx": ("strategy_agent.ingestion.pptx_loader.PptxLoader", {}),
}


def _import_loader(dotted_path: str):
    """Dynamically import a loader class by its fully-qualified name."""
    module_path, cls_name = dotted_path.rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def load_file(path: Path) -> list[Document]:
    """Load a single file and return a list of Documents."""
    suffix = path.suffix.lower()
    if suffix not in _LOADER_REGISTRY:
        logger.warning("Unsupported file type %s — skipping %s", suffix, path.name)
        return []

    dotted, kwargs = _LOADER_REGISTRY[suffix]
    loader_cls = _import_loader(dotted)
    loader = loader_cls(str(path), **kwargs)
    docs = loader.load()

    # Enrich metadata uniformly
    for doc in docs:
        doc.metadata["source_file"] = path.name
        doc.metadata["source_path"] = str(path)
        doc.metadata["file_type"] = suffix

    logger.info("Loaded %d page(s) from %s", len(docs), path.name)
    return docs


def load_corpus(corpus_dir: Path, *, max_workers: int = 8) -> list[Document]:
    """Recursively load every supported file under *corpus_dir*.

    Files are loaded concurrently using a thread pool (I/O-bound work).
    Results are returned in deterministic sorted-path order.
    """
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    paths = sorted(
        p for p in corpus_dir.rglob("*") if p.is_file() and not p.name.startswith(".")
    )

    if not paths:
        return []

    # Load files concurrently; collect results keyed by index to preserve order.
    all_docs: list[Document] = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(paths))) as executor:
        future_to_idx = {
            executor.submit(load_file, path): idx for idx, path in enumerate(paths)
        }
        results: list[list[Document]] = [[] for _ in paths]
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.warning("Failed to load %s: %s", paths[idx], e)

    for docs in results:
        all_docs.extend(docs)

    logger.info("Corpus loaded: %d document chunks from %s", len(all_docs), corpus_dir)
    return all_docs
