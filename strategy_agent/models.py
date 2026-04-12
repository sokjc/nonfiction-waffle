"""Factory functions for LLM and embedding instances.

Every model is accessed through OpenAI-compatible endpoints, making this
framework portable across vLLM, Ollama, LM Studio, text-generation-inference,
and any other server that exposes the ``/v1/chat/completions`` contract.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from strategy_agent.config import Settings, get_settings


# ---------------------------------------------------------------------------
# LLM factories
# ---------------------------------------------------------------------------

def build_writer_llm(settings: Settings | None = None) -> ChatOpenAI:
    """Return the primary LLM used for research, writing, and rewriting."""
    s = settings or get_settings()
    return ChatOpenAI(
        base_url=s.llm_base_url,
        api_key=s.llm_api_key,
        model=s.llm_model,
        temperature=s.llm_temperature,
        max_tokens=s.llm_max_tokens,
    )


def build_eval_llm(settings: Settings | None = None) -> ChatOpenAI:
    """Return the evaluation LLM (defaults to primary if not configured)."""
    s = settings or get_settings()
    return ChatOpenAI(
        base_url=s.eval_base_url_resolved,
        api_key=s.eval_api_key_resolved,
        model=s.eval_model_resolved,
        temperature=s.eval_temperature,
        max_tokens=s.llm_max_tokens,
    )


# ---------------------------------------------------------------------------
# Embedding factory
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def build_embeddings(settings: Settings | None = None) -> OpenAIEmbeddings:
    """Return an embeddings model.

    When ``EMBEDDING_BASE_URL`` is set the framework uses a remote
    OpenAI-compatible embedding endpoint.  Otherwise it falls back to
    a local ``sentence-transformers`` model wrapped through LangChain's
    HuggingFace integration.
    """
    s = settings or get_settings()

    if s.embedding_base_url:
        return OpenAIEmbeddings(
            base_url=s.embedding_base_url,
            api_key=s.embedding_api_key or "not-needed",
            model=s.embedding_model,
        )

    # Local sentence-transformers via the community wrapper
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=s.embedding_model)  # type: ignore[return-value]
