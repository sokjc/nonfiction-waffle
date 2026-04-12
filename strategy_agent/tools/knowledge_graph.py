"""Knowledge graph tools — let agents query and grow the KG during conversation."""

from __future__ import annotations

import threading

from langchain_core.tools import tool

from strategy_agent.memory.knowledge_graph import KnowledgeGraphStore

_lock = threading.Lock()
_store: KnowledgeGraphStore | None = None


def _get_store() -> KnowledgeGraphStore:
    global _store
    with _lock:
        if _store is None:
            _store = KnowledgeGraphStore()
        return _store


def set_store(store: KnowledgeGraphStore) -> None:
    """Allow external code to inject a store instance."""
    global _store
    with _lock:
        _store = store


@tool
def query_knowledge_graph(entity: str) -> str:
    """Query the knowledge graph for relationships involving a specific entity.

    Use this tool to discover structured relationships between companies,
    markets, strategies, and concepts.  Unlike corpus search (which finds
    similar passages), this returns explicit entity relationships: who
    competes with whom, what markets a company operates in, etc.

    Args:
        entity: The entity name to look up (e.g. "European market", "Product X").

    Returns:
        Known relationships for the entity, or a note if nothing was found.
    """
    store = _get_store()
    knowledge = store.query_entity(entity)

    if knowledge:
        return f"Knowledge about '{entity}':\n" + "\n".join(f"- {k}" for k in knowledge)

    # Fall back to partial match
    matches = store.search_entities(entity)
    if matches:
        lines = [f"- {s} --[{p}]--> {o}" for s, p, o in matches[:15]]
        return f"Partial matches for '{entity}':\n" + "\n".join(lines)

    return f"No relationships found for '{entity}' in the knowledge graph."


@tool
def add_knowledge(subject: str, predicate: str, obj: str) -> str:
    """Add a new fact to the knowledge graph during conversation.

    Use this when the user shares a new piece of strategic intelligence
    (e.g. "Acme Corp just acquired WidgetCo") that should be remembered
    across sessions.

    Args:
        subject: The source entity (e.g. "Acme Corp").
        predicate: The relationship (e.g. "acquired").
        obj: The target entity (e.g. "WidgetCo").

    Returns:
        Confirmation that the triple was stored.
    """
    store = _get_store()
    store.add_triple(subject, predicate, obj)
    store.save()
    return f"Added to knowledge graph: {subject} --[{predicate}]--> {obj}"
