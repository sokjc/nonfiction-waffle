"""Research Agent — mines the corpus and produces a structured research brief.

The researcher uses **hybrid retrieval**: LlamaIndex finds the most relevant
chunks, then expands to full source documents for long-context stuffing.
The LLM sees both precision (specific relevant passages) and completeness
(full documents so it doesn't miss context around those passages).
"""

from __future__ import annotations

import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from strategy_agent.config import Settings, get_settings
from strategy_agent.errors import invoke_llm
from strategy_agent.memory.vector_store import CorpusStore
from strategy_agent.memory.working_memory import WorkingMemory
from strategy_agent.models import build_agent_llm
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
    """Retrieves and synthesizes corpus knowledge for a writing brief.

    Uses hybrid retrieval: chunk-level similarity search to find relevant
    passages, then document-level expansion to stuff full source documents
    into the LLM context window.
    """

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        store: CorpusStore | None = None,
        settings: Settings | None = None,
    ):
        self._settings = settings or get_settings()
        self._llm = llm or build_agent_llm(self._settings)
        self._store = store or CorpusStore(self._settings)

        # Wire the tools to use our store instance
        set_corpus_store(self._store)

        # Build the synthesis chain
        self._chain = research_prompt | self._llm | StrOutputParser()

    def run(self, memory: WorkingMemory) -> WorkingMemory:
        """Execute hybrid retrieval research and update working memory.

        1. Run chunk-level similarity search for the brief
        2. Expand top-scoring chunks to full source documents
        3. Build a context package with both chunks (for citation) and
           full documents (for completeness)
        4. Synthesize via LLM into a research brief
        """
        query = f"{memory.brief} {memory.document_type} {memory.additional_instructions}"

        # Hybrid retrieval: chunks + full document expansion
        result = self._store.hybrid_retrieve(query, self._settings)
        chunks = result["chunks"]
        full_docs = result["full_documents"]

        # Build the context for the LLM
        context_parts: list[str] = []

        # Full source documents (long-context stuffing)
        if full_docs:
            context_parts.append("## Full Source Documents\n")
            for doc in full_docs:
                context_parts.append(
                    f"### {doc['source_file']}\n\n{doc['text']}\n"
                )
            context_parts.append("\n---\n")

        # Chunk-level passages (for precision + citation)
        if chunks:
            context_parts.append("## Most Relevant Passages (ranked by similarity)\n")
            for i, chunk in enumerate(chunks, 1):
                source = chunk["source_file"]
                score = chunk["score"]
                context_parts.append(
                    f"[{i}] (Source: {source}, relevance: {score:.2f})\n{chunk['text']}\n"
                )

        memory.retrieved_context = [c["text"] for c in chunks]
        retrieved_text = "\n\n".join(context_parts) if context_parts else (
            "No documents found in the corpus.  The writer will need to rely "
            "on general knowledge and the brief alone."
        )

        # Synthesize via LLM
        logger.info(
            "Synthesizing research from %d chunks + %d full documents",
            len(chunks),
            len(full_docs),
        )
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

        This mode binds ``search_corpus``, ``search_corpus_with_context``,
        and ``query_knowledge_graph`` as tools and lets the model iteratively
        search until it's satisfied.
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
                memory.research_synthesis = response.content
                break

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
            memory.research_synthesis = messages[-1].content if messages else ""

        logger.info("Agentic research complete (%d chars)", len(memory.research_synthesis))
        return memory
