"""Tests for the knowledge graph memory layer."""

import os
import tempfile
from unittest.mock import patch

from strategy_agent.config import Settings
from strategy_agent.memory.knowledge_graph import KnowledgeGraphStore


def _make_settings(tmp_path: str) -> Settings:
    kg_path = os.path.join(tmp_path, "test_kg.gml")
    with patch.dict(os.environ, {"KNOWLEDGE_GRAPH_PATH": kg_path}, clear=True):
        return Settings()


def test_add_and_query_triple():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _make_settings(tmp)
        kg = KnowledgeGraphStore(settings)

        kg.add_triple("Acme Corp", "competes with", "WidgetCo")
        knowledge = kg.query_entity("Acme Corp")
        assert len(knowledge) >= 1
        assert any("WidgetCo" in k for k in knowledge)


def test_add_triples_batch():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _make_settings(tmp)
        kg = KnowledgeGraphStore(settings)

        triples = [
            ("Acme Corp", "operates in", "North America"),
            ("Acme Corp", "revenue is", "$500M"),
            ("WidgetCo", "acquired", "SmallCo"),
        ]
        count = kg.add_triples(triples)
        assert count == 3
        assert kg.num_triples >= 3


def test_save_and_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _make_settings(tmp)

        # Create and populate
        kg1 = KnowledgeGraphStore(settings)
        kg1.add_triple("Company A", "partnered with", "Company B")
        kg1.add_triple("Company A", "operates in", "EMEA")
        kg1.save()

        # Reload from disk
        kg2 = KnowledgeGraphStore(settings)
        assert kg2.num_triples >= 2
        knowledge = kg2.query_entity("Company A")
        assert len(knowledge) >= 1


def test_search_entities_partial_match():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _make_settings(tmp)
        kg = KnowledgeGraphStore(settings)

        kg.add_triple("European Market", "grew by", "12%")
        kg.add_triple("APAC Market", "grew by", "8%")

        results = kg.search_entities("market")
        assert len(results) >= 2


def test_get_neighbors():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _make_settings(tmp)
        kg = KnowledgeGraphStore(settings)

        kg.add_triple("Acme Corp", "competes with", "WidgetCo")
        kg.add_triple("Acme Corp", "partnered with", "TechCo")

        neighbors = kg.get_neighbors("Acme Corp")
        assert len(neighbors) >= 2


def test_reset():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _make_settings(tmp)
        kg = KnowledgeGraphStore(settings)

        kg.add_triple("A", "relates to", "B")
        assert kg.num_triples >= 1

        kg.reset()
        assert kg.num_triples == 0


def test_num_entities():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _make_settings(tmp)
        kg = KnowledgeGraphStore(settings)

        kg.add_triple("X", "connects to", "Y")
        kg.add_triple("Y", "connects to", "Z")
        # X, Y, Z = 3 unique entities
        assert kg.num_entities >= 3
