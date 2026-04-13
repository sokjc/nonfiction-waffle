"""Factory functions for LLM and embedding instances.

Three model roles, each backed by an OpenAI-compatible endpoint:

- **Writer** (``build_writer_llm``) — prose generation, rewriting
- **Agent** (``build_agent_llm``) — research, chat, tool-calling
- **Evaluator** (``build_eval_llm``) — narrative scoring

Embeddings default to an OpenAI-compatible API endpoint (``EMBEDDING_BASE_URL``).
For local embeddings via sentence-transformers, install the ``local`` extra:
``uv pip install -e ".[local]"``
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from strategy_agent.config import Settings, get_settings


# ---------------------------------------------------------------------------
# LLM factories (LangChain — used by agents and LangGraph)
# ---------------------------------------------------------------------------

def build_writer_llm(settings: Settings | None = None) -> ChatOpenAI:
    """Return the writer LLM (gpt-oss by default) — optimized for prose."""
    s = settings or get_settings()
    return ChatOpenAI(
        base_url=s.writer_base_url,
        api_key=s.writer_api_key,
        model=s.writer_model,
        temperature=s.writer_temperature,
        max_tokens=s.writer_max_tokens,
    )


def build_agent_llm(settings: Settings | None = None) -> ChatOpenAI:
    """Return the agent LLM (gemma4-31b by default) — for tool-calling and chat."""
    s = settings or get_settings()
    return ChatOpenAI(
        base_url=s.agent_base_url,
        api_key=s.agent_api_key,
        model=s.agent_model,
        temperature=s.agent_temperature,
        max_tokens=s.agent_max_tokens,
    )


def build_eval_llm(settings: Settings | None = None) -> ChatOpenAI:
    """Return the evaluator LLM (nemotron-120b by default) — heavier model for judgment."""
    s = settings or get_settings()
    return ChatOpenAI(
        base_url=s.eval_base_url,
        api_key=s.eval_api_key,
        model=s.eval_model,
        temperature=s.eval_temperature,
        max_tokens=s.eval_max_tokens,
    )


# ---------------------------------------------------------------------------
# Embedding factory (LlamaIndex)
# ---------------------------------------------------------------------------

_embeddings_cache: dict[tuple[str, str | None], object] = {}


def build_embeddings(settings: Settings | None = None):
    """Return a LlamaIndex embedding model, cached by (model, base_url).

    When ``EMBEDDING_BASE_URL`` is set (recommended), uses a remote
    OpenAI-compatible embedding endpoint — no local GPU or PyTorch needed.

    Without ``EMBEDDING_BASE_URL``, falls back to local inference via
    sentence-transformers, which requires the ``local`` extra::

        uv pip install -e ".[local]"
    """
    s = settings or get_settings()
    cache_key = (s.embedding_model, s.embedding_base_url)

    if cache_key in _embeddings_cache:
        return _embeddings_cache[cache_key]

    if s.embedding_base_url:
        from llama_index.embeddings.openai import OpenAIEmbedding

        emb = OpenAIEmbedding(
            api_base=s.embedding_base_url,
            api_key=s.embedding_api_key or "not-needed",
            model_name=s.embedding_model,
        )
    else:
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        except ImportError:
            raise ImportError(
                "Local embeddings require the 'local' extra (sentence-transformers + PyTorch). "
                "Install with: uv pip install -e \".[local]\"\n"
                "Or set EMBEDDING_BASE_URL to use an API-based embedding endpoint instead."
            ) from None

        emb = HuggingFaceEmbedding(
            model_name=s.embedding_model,
            trust_remote_code=True,
        )

    _embeddings_cache[cache_key] = emb
    return emb
