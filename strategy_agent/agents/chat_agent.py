"""Conversational strategy agent with cross-session persistence.

Uses ``langgraph.prebuilt.create_react_agent`` with a ``SqliteSaver``
checkpointer so conversations survive across CLI sessions.  The agent
has access to the corpus, knowledge graph, style checker, and the full
document-generation pipeline as callable tools.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from strategy_agent.config import Settings, get_settings
from strategy_agent.memory.knowledge_graph import KnowledgeGraphStore
from strategy_agent.memory.vector_store import CorpusStore
from strategy_agent.models import build_writer_llm
from strategy_agent.prompts.chat import CHAT_SYSTEM_PROMPT
from strategy_agent.session import SessionManager
from strategy_agent.tools.corpus_search import (
    search_corpus,
    search_corpus_with_context,
    set_store as set_corpus_store,
)
from strategy_agent.tools.fact_check import (
    verify_claim,
    set_store as set_fact_store,
)
from strategy_agent.tools.knowledge_graph import (
    add_knowledge,
    query_knowledge_graph,
    set_store as set_kg_store,
)
from strategy_agent.tools.style_check import check_style

logger = logging.getLogger(__name__)


def _build_generate_tool(settings: Settings):
    """Create the generate_document tool with a closure over settings."""

    @tool
    def generate_document(brief: str, document_type: str = "strategy_memo") -> str:
        """Generate a full strategy document using the multi-agent pipeline.

        Use this when asked to write, draft, or produce a complete strategy
        document.  Runs the research → write → evaluate → rewrite loop and
        returns the finished document.

        Args:
            brief: What the document should cover.
            document_type: One of strategy_memo, white_paper, board_presentation,
                          competitive_analysis, market_assessment.

        Returns:
            The generated document text with evaluation scores.
        """
        from strategy_agent.orchestrator import run_pipeline

        memory = run_pipeline(
            brief=brief,
            document_type=document_type,
            settings=settings,
        )

        # Save to output dir
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = settings.output_dir / f"{document_type}_{timestamp}.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(memory.latest_draft, encoding="utf-8")

        score_info = memory.latest_evaluation.summary if memory.latest_evaluation else "no evaluation"
        doc_preview = memory.latest_draft[:4000]
        truncated = len(memory.latest_draft) > 4000

        return (
            f"Document generated and saved to {output_path}\n"
            f"Iterations: {memory.current_iteration} | Final score: {score_info}\n\n"
            f"--- DOCUMENT ---\n\n{doc_preview}"
            + ("\n\n[...truncated — full text saved to file]" if truncated else "")
        )

    return generate_document


def build_chat_agent(
    session_manager: SessionManager,
    settings: Settings | None = None,
):
    """Construct the conversational ReAct agent.

    Returns a compiled LangGraph that accepts::

        agent.invoke(
            {"messages": [("human", "your question")]},
            config={"configurable": {"thread_id": "..."}},
        )
    """
    s = settings or get_settings()
    llm = build_writer_llm(s)

    # Wire up tool stores
    corpus_store = CorpusStore(s)
    set_corpus_store(corpus_store)
    set_fact_store(corpus_store)

    kg_store = KnowledgeGraphStore(s)
    set_kg_store(kg_store)

    tools = [
        search_corpus,
        search_corpus_with_context,
        query_knowledge_graph,
        add_knowledge,
        check_style,
        verify_claim,
        _build_generate_tool(s),
    ]

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=CHAT_SYSTEM_PROMPT,
        checkpointer=session_manager.checkpointer,
        name="strategy-chat-agent",
    )

    return agent
