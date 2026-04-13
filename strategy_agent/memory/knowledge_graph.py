"""Persistent knowledge graph backed by NetworkX and stored as GML.

The knowledge graph stores entity-relationship triples extracted from
corpus documents and conversations.  It complements the vector store:

- **Vector store** → "find passages semantically similar to X" (fuzzy)
- **Knowledge graph** → "what entities are connected to X and how?" (structured)
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.graphs import NetworkxEntityGraph
from langchain_community.graphs.networkx_graph import KnowledgeTriple

from strategy_agent.config import Settings, get_settings

logger = logging.getLogger(__name__)


class KnowledgeGraphStore:
    """Persistent wrapper around ``NetworkxEntityGraph`` with GML storage."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._path = self._settings.kg_gml_path
        self._graph = self._load_or_create()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _load_or_create(self) -> NetworkxEntityGraph:
        if self._path.exists():
            logger.info("Loading knowledge graph from %s", self._path)
            return NetworkxEntityGraph.from_gml(str(self._path))
        logger.info("Creating new knowledge graph")
        return NetworkxEntityGraph()

    def save(self) -> None:
        """Persist the graph to disk as GML."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._graph.write_to_gml(str(self._path))
        logger.info(
            "Knowledge graph saved (%d triples) to %s",
            self.num_triples,
            self._path,
        )

    def reset(self) -> None:
        """Delete the graph and start fresh (destructive)."""
        self._graph.clear()
        if self._path.exists():
            self._path.unlink()
        logger.warning("Knowledge graph reset")

    # ── Mutators ──────────────────────────────────────────────────────────────

    def add_triple(self, subject: str, predicate: str, obj: str) -> None:
        """Add a single (subject, predicate, object) relationship."""
        if not subject or not predicate or not obj:
            logger.debug("Skipping invalid triple: (%r, %r, %r)", subject, predicate, obj)
            return
        self._graph.add_triple(KnowledgeTriple(subject, predicate, obj))

    def add_triples(self, triples: list[tuple[str, str, str]]) -> int:
        """Add multiple triples.  Returns the count added."""
        for subj, pred, obj in triples:
            self.add_triple(subj, pred, obj)
        return len(triples)

    def has_triple(self, subject: str, predicate: str, obj: str) -> bool:
        """Return True if an exact (subject, predicate, object) triple exists."""
        s_low, p_low, o_low = subject.lower(), predicate.lower(), obj.lower()
        for s, p, o in self.get_all_triples():
            if s.lower() == s_low and p.lower() == p_low and o.lower() == o_low:
                return True
        return False

    def add_triple_if_new(self, subject: str, predicate: str, obj: str) -> bool:
        """Add a triple only if it does not already exist (case-insensitive).

        Returns True if the triple was added, False if it was a duplicate.
        """
        if not subject or not predicate or not obj:
            return False
        if self.has_triple(subject, predicate, obj):
            logger.debug("Skipping duplicate triple: (%r, %r, %r)", subject, predicate, obj)
            return False
        self._graph.add_triple(KnowledgeTriple(subject, predicate, obj))
        return True

    def add_triples_if_new(self, triples: list[tuple[str, str, str]]) -> tuple[int, int]:
        """Add multiple triples, skipping duplicates.

        Returns ``(added, skipped)`` counts.
        """
        added = skipped = 0
        for subj, pred, obj in triples:
            if self.add_triple_if_new(subj, pred, obj):
                added += 1
            else:
                skipped += 1
        return added, skipped

    def remove_entity(self, entity: str) -> int:
        """Remove all triples where *entity* appears as subject or object.

        Returns the number of triples removed.
        """
        triples_before = self.num_triples
        triples = self.get_all_triples()
        self._graph.clear()

        for s, p, o in triples:
            if s.lower() != entity.lower() and o.lower() != entity.lower():
                self._graph.add_triple(KnowledgeTriple(s, p, o))

        removed = triples_before - self.num_triples
        if removed:
            logger.info("Removed %d triple(s) involving %r", removed, entity)
        return removed

    def deduplicate(self) -> int:
        """Remove exact duplicate triples (case-insensitive comparison).

        Keeps the first occurrence and removes subsequent duplicates.
        Returns the number of duplicates removed.
        """
        triples = self.get_all_triples()
        seen: set[tuple[str, str, str]] = set()
        unique: list[tuple[str, str, str]] = []

        for s, p, o in triples:
            key = (s.lower(), p.lower(), o.lower())
            if key not in seen:
                seen.add(key)
                unique.append((s, p, o))

        removed = len(triples) - len(unique)
        if removed > 0:
            self._graph.clear()
            for s, p, o in unique:
                self._graph.add_triple(KnowledgeTriple(s, p, o))
            logger.info("Removed %d duplicate triple(s)", removed)
        else:
            logger.info("No duplicate triples found")
        return removed

    # ── Queries ───────────────────────────────────────────────────────────────

    def query_entity(self, entity: str) -> list[str]:
        """Return all known facts about *entity*."""
        return self._graph.get_entity_knowledge(entity)

    def get_all_triples(self) -> list[tuple[str, str, str]]:
        """Return every triple as ``(subject, predicate, object)``.

        Note: ``NetworkxEntityGraph.get_triples()`` returns
        ``(subject, object, predicate)`` — we re-order to the conventional
        ``(s, p, o)`` form so all downstream code uses a consistent layout.
        """
        return [(s, p, o) for s, o, p in self._graph.get_triples()]

    def get_neighbors(self, entity: str) -> list[str]:
        """Return entities directly connected to *entity*."""
        return list(self._graph.get_neighbors(entity))

    def search_entities(self, query: str) -> list[tuple[str, str, str]]:
        """Case-insensitive partial-match search across all triples.

        Returns ``(subject, predicate, object)`` tuples matching the query.
        """
        q = query.lower()
        return [
            (s, p, o) for s, p, o in self.get_all_triples()
            if q in s.lower() or q in p.lower() or q in o.lower()
        ]

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def num_triples(self) -> int:
        return len(self._graph.get_triples())

    @property
    def num_entities(self) -> int:
        """Count unique entities (subjects and objects) in the graph."""
        entities: set[str] = set()
        for s, _p, o in self.get_all_triples():
            entities.add(s)
            entities.add(o)
        return len(entities)
