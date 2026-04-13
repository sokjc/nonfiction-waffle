"""Ingestion registry that tracks which files have been loaded.

Maintains a JSON file mapping content hashes to file metadata so the
ingestion pipeline can skip documents that have already been processed.
A file is considered a duplicate when its SHA-256 hash matches an
existing entry — this catches both exact re-ingestion and files that
have been moved/renamed but are otherwise identical.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Size of read buffer for hashing large files
_HASH_BUF_SIZE = 1 << 16  # 64 KiB


def compute_file_hash(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's contents."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(_HASH_BUF_SIZE)
            if not data:
                break
            sha.update(data)
    return sha.hexdigest()


class IngestionRegistry:
    """JSON-backed registry of ingested files.

    Each entry records the content hash, source file name, path,
    ingestion timestamp, and the number of chunks produced.
    """

    def __init__(self, registry_path: Path):
        self._path = registry_path
        self._entries: dict[str, dict] = self._load()

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def _load(self) -> dict[str, dict]:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                logger.info("Loaded ingestion registry with %d entries", len(data))
                return data
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load registry (%s) — starting fresh", exc)
        return {}

    def save(self) -> None:
        """Persist the registry to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._entries, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        logger.info("Ingestion registry saved (%d entries)", len(self._entries))

    def reset(self) -> None:
        """Clear all entries and delete the registry file."""
        self._entries.clear()
        if self._path.exists():
            self._path.unlink()
        logger.warning("Ingestion registry reset")

    # ── Queries ───────────────────────────────────────────────────────────

    def is_ingested(self, path: Path) -> bool:
        """Return True if a file with the same content hash is already registered."""
        file_hash = compute_file_hash(path)
        return file_hash in self._entries

    def get_entry_by_hash(self, file_hash: str) -> dict | None:
        """Return the registry entry for a given hash, or None."""
        return self._entries.get(file_hash)

    def get_entry_by_source(self, source_file: str) -> tuple[str, dict] | None:
        """Find a registry entry by source_file name.

        Returns ``(hash, entry)`` or ``None`` if not found.
        """
        for h, entry in self._entries.items():
            if entry.get("source_file") == source_file:
                return h, entry
        return None

    def list_files(self) -> list[dict]:
        """Return a list of all registered files with their metadata."""
        return [
            {"hash": h, **entry}
            for h, entry in sorted(
                self._entries.items(), key=lambda kv: kv[1].get("ingested_at", "")
            )
        ]

    @property
    def count(self) -> int:
        return len(self._entries)

    # ── Mutators ──────────────────────────────────────────────────────────

    def register(self, path: Path, chunk_count: int = 0) -> str:
        """Record a file as ingested.  Returns the content hash."""
        file_hash = compute_file_hash(path)
        self._entries[file_hash] = {
            "source_file": path.name,
            "source_path": str(path),
            "ingested_at": datetime.now(UTC).isoformat(),
            "chunk_count": chunk_count,
        }
        return file_hash

    def unregister_by_source(self, source_file: str) -> bool:
        """Remove registry entries matching *source_file*.  Returns True if any removed."""
        to_remove = [
            h for h, e in self._entries.items() if e.get("source_file") == source_file
        ]
        for h in to_remove:
            del self._entries[h]
        return len(to_remove) > 0

    def unregister_by_hash(self, file_hash: str) -> bool:
        """Remove a specific hash entry.  Returns True if removed."""
        if file_hash in self._entries:
            del self._entries[file_hash]
            return True
        return False
