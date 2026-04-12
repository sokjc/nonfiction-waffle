"""Research Agent — mines the corpus and produces a structured research brief.

The researcher operates in two modes:

1. **Retrieval mode** (default): Uses the vector-store retriever to pull
   relevant passages based on the brief, then synthesizes them via LLM.
2. **Agentic mode**: Given the search tools, the LLM decides what queries
   to run, iterating until it has enough material.  This mode is better for
   complex briefs that span multiple topics.
"""

from __future__ import annotations

import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from strategy_agent.config import Settings, get_settings
from strategy_agent.errors import invoke_llm
from strategy_agent.memory.vector_store import CorpusStore
from strategy_agent.memory.working_memory import WorkingMemory
from strategy_agent.models import build_writer_llm
from strategy_agent.prompts.researcher import research_prompt
from strategy_agent.tools.corpus_search import (
    search_corpus,
    search_corpus_with_context,
    set_store as set_corpus_store,
)
from strategy_agent.tools.knowledge_graph import (
    query_knowledge_graph,
    set_store as set_kg_store,
)

logger = logging.getLogger(__name__)


class ResearchAgent:
    """Retrieves and synthesizes corpus knowledge for a writing brief."""

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        store: CorpusStore | None = None,
        settings: Settings | None = None,
    ):
        self._settings = settings or get_settings()
        self._llm = llm or build_writer_llm(self._settings)
        self._store = store or CorpusStore(self._settings)

        # Wire the tools to use our store instance
        set_corpus_store(self._store)

        # Build the retrieval-mode chain (simpler, faster, deterministic)
        self._chain = research_prompt | self._llm | StrOutputParser()

    # ── Retrieval mode ────────────────────────────────────────────────────────

    def run(self, memory: WorkingMemory) -> WorkingMemory:
        """Execute retrieval-mode research and update working memory."""
        # Step 1: Retrieve relevant passages from the corpus
        retriever = self._store.as_retriever(top_k=self._settings.retrieval_top_k)
        query = f"{memory.brief} {memory.document_type} {memory.additional_instructions}"
        docs = retriever.invoke(query)

        context_passages = []
        for doc in docs:
            source = doc.metadata.get("source_file", "unknown")
            context_passages.append(f"[Source: {source}]\n{doc.page_content}")

        memory.retrieved_context = context_passages
        retrieved_text = "\n\n---\n\n".join(context_passages) if context_passages else (
            "No documents found in the corpus.  The writer will need to rely "
            "on general knowledge and the brief alone."
        )

        # Step 2: Synthesize via LLM
        logger.info("Synthesizing research from %d retrieved passages", len(docs))
        synthesis = invoke_llm(
            self._chain,
            {
                "brief": memory.brief,
                "document_type": memory.document_type,
                "additional_instructions": memory.additional_instructions,
                "retrieved_context": retrieved_text,
            },
            endpoint_url=self._settings.llm_base_url,
        )

        memory.research_synthesis = synthesis
        logger.info("Research synthesis complete (%d chars)", len(synthesis))
        return memory

    # ── Agentic mode ──────────────────────────────────────────────────────────

    def run_agentic(self, memory: WorkingMemory) -> WorkingMemory:
        """Use tool-calling to let the LLM drive its own research queries.

        This mode binds ``search_corpus`` and ``search_corpus_with_context``
        as tools and lets the model iteratively search until it's satisfied.
        Falls back to retrieval mode if the model doesn't support tool calling.
        """
        tools = [search_corpus, search_corpus_with_context, query_knowledge_graph]
        agent_llm = self._llm.bind_tools(tools)

        from langchain_core.messages import HumanMessage, SystemMessage

        from strategy_agent.prompts.researcher import RESEARCH_SYSTEM

        messages = [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(content=(
                f"Brief: {memory.brief}\n"
                f"Document type: {memory.document_type}\n"
                f"Additional instructions: {memory.additional_instructions}\n\n"
                "Use the search tools to find all relevant information in the "
                "corpus, then produce a structured research synthesis."
            )),
        ]

        # Iterative tool-calling loop (max 10 rounds to prevent runaway)
        for _round in range(10):
            response = agent_llm.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                # Model is done searching — final message is the synthesis
                memory.research_synthesis = response.content
                break

            # Execute each tool call and feed results back
            from langchain_core.messages import ToolMessage

            tool_map = {
                "search_corpus": search_corpus,
                "search_corpus_with_context": search_corpus_with_context,
                "query_knowledge_graph": query_knowledge_graph,
            }
            for tc in response.tool_calls:
                tool_fn = tool_map.get(tc["name"], search_corpus)
                result = tool_fn.invoke(tc["args"])
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        else:
            # Fallback: extract whatever the model produced
            memory.research_synthesis = messages[-1].content if messages else ""

        logger.info("Agentic research complete (%d chars)", len(memory.research_synthesis))
        return memory
