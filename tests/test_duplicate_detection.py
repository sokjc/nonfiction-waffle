"""Tests for duplicate detection and deduplication in the knowledge graph."""

import os
import tempfile
from unittest.mock import patch

from strategy_agent.config import Settings
from strategy_agent.memory.knowledge_graph import KnowledgeGraphStore


def _make_settings(tmp_path: str) -> Settings:
    kg_path = os.path.join(tmp_path, "test_kg.gml")
    with patch.dict(os.environ, {"KNOWLEDGE_GRAPH_PATH": kg_path}, clear=True):
        return Settings()


def test_has_triple():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _make_settings(tmp)
        kg = KnowledgeGraphStore(settings)

        kg.add_triple("Acme", "competes with", "WidgetCo")
        assert kg.has_triple("Acme", "competes with", "WidgetCo")
        assert not kg.has_triple("Acme", "acquired", "WidgetCo")


def test_has_triple_case_insensitive():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _make_settings(tmp)
        kg = KnowledgeGraphStore(settings)

        kg.add_triple("Acme Corp", "COMPETES WITH", "WidgetCo")
        assert kg.has_triple("acme corp", "competes with", "widgetco")
        assert kg.has_triple("ACME CORP", "Competes With", "WIDGETCO")


def test_add_triple_if_new():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _make_settings(tmp)
        kg = KnowledgeGraphStore(settings)

        added = kg.add_triple_if_new("A", "relates to", "B")
        assert added is True
        assert kg.num_triples == 1

        added = kg.add_triple_if_new("A", "relates to", "B")
        assert added is False
        assert kg.num_triples == 1


def test_add_triple_if_new_skips_invalid():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _make_settings(tmp)
        kg = KnowledgeGraphStore(settings)

        assert kg.add_triple_if_new("", "relates to", "B") is False
        assert kg.add_triple_if_new("A", None, "B") is False
        assert kg.num_triples == 0


def test_add_triples_if_new():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _make_settings(tmp)
        kg = KnowledgeGraphStore(settings)

        triples = [
            ("A", "relates to", "B"),
            ("C", "relates to", "D"),
            ("A", "relates to", "B"),  # duplicate
        ]
        added, skipped = kg.add_triples_if_new(triples)
        assert added == 2
        assert skipped == 1
        assert kg.num_triples == 2


def test_deduplicate_case_variants():
    """Deduplicate should merge triples that differ only in case."""
    with tempfile.TemporaryDirectory() as tmp:
        settings = _make_settings(tmp)
        kg = KnowledgeGraphStore(settings)

        # These are considered duplicates under case-insensitive comparison
        kg.add_triple("Acme Corp", "competes with", "WidgetCo")
        kg.add_triple("acme corp", "Competes With", "widgetco")
        kg.add_triple("Alpha", "connects", "Beta")

        initial = kg.num_triples
        removed = kg.deduplicate()
        assert removed >= 1
        assert kg.num_triples == initial - removed
        # Should retain the first-seen casing
        triples = kg.get_all_triples()
        acme_triples = [t for t in triples if "acme" in t[0].lower()]
        assert len(acme_triples) == 1


def test_deduplicate_no_duplicates():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _make_settings(tmp)
        kg = KnowledgeGraphStore(settings)

        kg.add_triple("X", "links", "Y")
        kg.add_triple("A", "connects", "B")

        removed = kg.deduplicate()
        assert removed == 0
        assert kg.num_triples == 2


def test_remove_entity():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _make_settings(tmp)
        kg = KnowledgeGraphStore(settings)

        kg.add_triple("Acme", "competes with", "WidgetCo")
        kg.add_triple("Acme", "operates in", "USA")
        kg.add_triple("TechCo", "acquired", "SmallCo")

        removed = kg.remove_entity("Acme")
        assert removed == 2
        assert kg.num_triples == 1
        # The remaining triple should be TechCo → SmallCo
        triples = kg.get_all_triples()
        assert any("TechCo" in t[0] for t in triples)


def test_remove_entity_as_object():
    """Removing an entity also removes triples where it appears as object."""
    with tempfile.TemporaryDirectory() as tmp:
        settings = _make_settings(tmp)
        kg = KnowledgeGraphStore(settings)

        kg.add_triple("Acme", "competes with", "WidgetCo")
        kg.add_triple("TechCo", "acquired", "WidgetCo")

        removed = kg.remove_entity("WidgetCo")
        assert removed == 2
        assert kg.num_triples == 0


def test_remove_entity_case_insensitive():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _make_settings(tmp)
        kg = KnowledgeGraphStore(settings)

        kg.add_triple("Acme Corp", "competes with", "WidgetCo")
        removed = kg.remove_entity("acme corp")
        assert removed == 1
        assert kg.num_triples == 0


def test_remove_nonexistent_entity():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _make_settings(tmp)
        kg = KnowledgeGraphStore(settings)

        kg.add_triple("A", "links", "B")
        removed = kg.remove_entity("Z")
        assert removed == 0
        assert kg.num_triples == 1
