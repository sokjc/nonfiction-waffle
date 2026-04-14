"""Strategy Writer Agent — produces drafts guided by the style guide and research.

The writer is a straightforward chain: it takes the brief, research synthesis,
and style guide (baked into the system prompt) and produces a complete
strategy document in a single pass.
"""

from __future__ import annotations

import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from strategy_agent.config import Settings, get_settings
from strategy_agent.errors import invoke_llm
from strategy_agent.memory.working_memory import WorkingMemory
from strategy_agent.models import build_writer_llm
from strategy_agent.prompts.writer import writer_prompt

logger = logging.getLogger(__name__)


class WriterAgent:
    """Produces strategy document drafts from research and a brief."""

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        settings: Settings | None = None,
    ):
        self._settings = settings or get_settings()
        self._llm = llm or build_writer_llm(self._settings)
        self._chain = writer_prompt | self._llm | StrOutputParser()

    def run(self, memory: WorkingMemory) -> WorkingMemory:
        """Generate a draft and append it to working memory."""
        logger.info(
            "Writing %s draft (iteration %d)",
            memory.document_type,
            memory.current_iteration + 1,
        )

        draft = invoke_llm(
            self._chain,
            {
                "brief": memory.brief,
                "document_type": memory.document_type,
                "research_synthesis": memory.research_synthesis,
                "additional_instructions": memory.additional_instructions,
            },
            endpoint_url=self._settings.writer_base_url,
        )

        memory.drafts.append(draft)
        memory.current_iteration += 1
        memory.model_used = self._settings.writer_model

        logger.info(
            "Draft %d complete (%d chars)",
            memory.current_iteration,
            len(draft),
        )
        return memory
