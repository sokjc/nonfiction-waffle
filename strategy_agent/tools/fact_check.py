"""Fact cross-referencing tool — verifies claims against the corpus.

The fact-checker takes a specific claim from a draft and searches the corpus
for supporting or contradicting evidence.  This helps the evaluator and
rewriter ensure the document stays grounded in the source material.
"""

from __future__ import annotations

import threading

from langchain_core.tools import tool

from strategy_agent.memory.vector_store import CorpusStore

_lock = threading.Lock()
_store: CorpusStore | None = None


def _get_store() -> CorpusStore:
    global _store
    with _lock:
        if _store is None:
            _store = CorpusStore()
        return _store


def set_store(store: CorpusStore) -> None:
    """Allow external code to inject a store instance."""
    global _store
    with _lock:
        _store = store


@tool
def verify_claim(claim: str) -> str:
    """Verify a specific claim or data point against the strategy corpus.

    Use this tool to fact-check statements in a draft.  Provide the exact
    claim (including any numbers or percentages) and this tool will search
    the corpus for supporting or contradicting evidence.

    Args:
        claim: The specific claim or data point to verify.

    Returns:
        A verdict with supporting evidence from the corpus, or a note
        that the claim could not be verified.
    """
    store = _get_store()
    docs = store.similarity_search(claim, k=5)

    if not docs:
        return (
            f"UNVERIFIED: No corpus passages found that address the claim: \"{claim}\". "
            "This claim may need to be removed or qualified as an inference."
        )

    evidence_lines: list[str] = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "unknown")
        evidence_lines.append(f"[{i}] (Source: {source})\n{doc.page_content[:500]}")

    evidence = "\n\n".join(evidence_lines)
    return (
        f"EVIDENCE FOUND for claim: \"{claim}\"\n\n"
        f"Relevant passages:\n{evidence}\n\n"
        "Review these passages to confirm the claim is accurately represented."
    )
