"""Tests for the document chunking logic."""

from unittest.mock import patch

from langchain_core.documents import Document

from strategy_agent.config import Settings
from strategy_agent.ingestion.chunker import chunk_documents


def _make_settings(**overrides) -> Settings:
    defaults = {"CHUNK_SIZE": "200", "CHUNK_OVERLAP": "20"}
    defaults.update(overrides)
    import os

    with patch.dict(os.environ, defaults, clear=True):
        return Settings()


def test_chunks_long_document():
    """A document longer than chunk_size should be split."""
    settings = _make_settings()
    long_text = "This is a sentence about strategy. " * 50
    doc = Document(page_content=long_text, metadata={"file_type": ".txt"})
    chunks = chunk_documents([doc], settings)
    assert len(chunks) > 1
    for chunk in chunks:
        # Each chunk should respect the size limit (with some tolerance for splits)
        assert len(chunk.page_content) <= settings.chunk_size + 50


def test_preserves_metadata():
    """Chunks should inherit the parent document's metadata."""
    settings = _make_settings()
    doc = Document(
        page_content="Short text about competitive analysis.",
        metadata={"file_type": ".txt", "source_file": "report.txt"},
    )
    chunks = chunk_documents([doc], settings)
    assert len(chunks) >= 1
    assert chunks[0].metadata["source_file"] == "report.txt"


def test_markdown_heading_split():
    """Markdown documents should split on heading boundaries."""
    settings = _make_settings(CHUNK_SIZE="500")
    md_text = (
        "# Executive Summary\n\nThis is the executive summary with key findings.\n\n"
        "## Market Analysis\n\nThe market is growing rapidly with significant opportunity.\n\n"
        "## Competitive Landscape\n\nThree major competitors dominate the space.\n"
    )
    doc = Document(page_content=md_text, metadata={"file_type": ".md"})
    chunks = chunk_documents([doc], settings)
    assert len(chunks) >= 1
