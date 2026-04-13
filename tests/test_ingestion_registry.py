"""Tests for the ingestion registry."""

import tempfile
from pathlib import Path

from strategy_agent.ingestion.registry import IngestionRegistry, compute_file_hash


def test_compute_file_hash():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("hello world")
        f.flush()
        h = compute_file_hash(Path(f.name))
    assert len(h) == 64  # SHA-256 hex digest
    assert h.isalnum()


def test_same_content_same_hash():
    with tempfile.TemporaryDirectory() as tmp:
        p1 = Path(tmp) / "a.txt"
        p2 = Path(tmp) / "b.txt"
        p1.write_text("identical content", encoding="utf-8")
        p2.write_text("identical content", encoding="utf-8")
        assert compute_file_hash(p1) == compute_file_hash(p2)


def test_different_content_different_hash():
    with tempfile.TemporaryDirectory() as tmp:
        p1 = Path(tmp) / "a.txt"
        p2 = Path(tmp) / "b.txt"
        p1.write_text("content A", encoding="utf-8")
        p2.write_text("content B", encoding="utf-8")
        assert compute_file_hash(p1) != compute_file_hash(p2)


def test_register_and_is_ingested():
    with tempfile.TemporaryDirectory() as tmp:
        reg_path = Path(tmp) / "registry.json"
        file_path = Path(tmp) / "doc.txt"
        file_path.write_text("some document", encoding="utf-8")

        registry = IngestionRegistry(reg_path)
        assert not registry.is_ingested(file_path)
        assert registry.count == 0

        registry.register(file_path, chunk_count=5)
        assert registry.is_ingested(file_path)
        assert registry.count == 1


def test_save_and_reload():
    with tempfile.TemporaryDirectory() as tmp:
        reg_path = Path(tmp) / "registry.json"
        file_path = Path(tmp) / "doc.txt"
        file_path.write_text("some document", encoding="utf-8")

        r1 = IngestionRegistry(reg_path)
        r1.register(file_path, chunk_count=3)
        r1.save()

        r2 = IngestionRegistry(reg_path)
        assert r2.is_ingested(file_path)
        assert r2.count == 1


def test_unregister_by_source():
    with tempfile.TemporaryDirectory() as tmp:
        reg_path = Path(tmp) / "registry.json"
        file_path = Path(tmp) / "report.pdf"
        file_path.write_text("pdf content", encoding="utf-8")

        registry = IngestionRegistry(reg_path)
        registry.register(file_path, chunk_count=10)
        assert registry.count == 1

        removed = registry.unregister_by_source("report.pdf")
        assert removed is True
        assert registry.count == 0


def test_unregister_nonexistent_source():
    with tempfile.TemporaryDirectory() as tmp:
        reg_path = Path(tmp) / "registry.json"
        registry = IngestionRegistry(reg_path)
        assert registry.unregister_by_source("nope.txt") is False


def test_list_files():
    with tempfile.TemporaryDirectory() as tmp:
        reg_path = Path(tmp) / "registry.json"
        p1 = Path(tmp) / "a.txt"
        p2 = Path(tmp) / "b.txt"
        p1.write_text("aaa", encoding="utf-8")
        p2.write_text("bbb", encoding="utf-8")

        registry = IngestionRegistry(reg_path)
        registry.register(p1, chunk_count=1)
        registry.register(p2, chunk_count=2)

        files = registry.list_files()
        assert len(files) == 2
        names = {f["source_file"] for f in files}
        assert names == {"a.txt", "b.txt"}


def test_get_entry_by_source():
    with tempfile.TemporaryDirectory() as tmp:
        reg_path = Path(tmp) / "registry.json"
        file_path = Path(tmp) / "doc.txt"
        file_path.write_text("document text", encoding="utf-8")

        registry = IngestionRegistry(reg_path)
        registry.register(file_path, chunk_count=7)

        result = registry.get_entry_by_source("doc.txt")
        assert result is not None
        h, entry = result
        assert entry["chunk_count"] == 7
        assert entry["source_file"] == "doc.txt"

        assert registry.get_entry_by_source("nope.txt") is None


def test_reset():
    with tempfile.TemporaryDirectory() as tmp:
        reg_path = Path(tmp) / "registry.json"
        file_path = Path(tmp) / "doc.txt"
        file_path.write_text("content", encoding="utf-8")

        registry = IngestionRegistry(reg_path)
        registry.register(file_path)
        registry.save()
        assert reg_path.exists()

        registry.reset()
        assert registry.count == 0
        assert not reg_path.exists()


def test_renamed_file_same_content_detected():
    """A file with the same content but different name is detected as already ingested."""
    with tempfile.TemporaryDirectory() as tmp:
        reg_path = Path(tmp) / "registry.json"
        p1 = Path(tmp) / "original.txt"
        p2 = Path(tmp) / "renamed.txt"
        p1.write_text("same content", encoding="utf-8")
        p2.write_text("same content", encoding="utf-8")

        registry = IngestionRegistry(reg_path)
        registry.register(p1)
        assert registry.is_ingested(p2)
