"""Factory functions for LLM and embedding instances.

All models share a single OpenAI-compatible endpoint (``LLM_BASE_URL`` +
``LLM_API_KEY``) and are distinguished only by model name:

- **Writer** (``build_writer_llm``) — prose generation, rewriting
- **Agent** (``build_agent_llm``) — research, chat, tool-calling
- **Evaluator** (``build_eval_llm``) — narrative scoring

Embeddings default to the same endpoint.  Set ``EMBEDDING_LOCAL=true`` to
use local sentence-transformers inference instead (requires the ``local``
extra: ``uv pip install -e ".[local]"``).
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
        base_url=s.llm_base_url,
        api_key=s.llm_api_key,
        model=s.writer_model,
        temperature=s.writer_temperature,
        max_tokens=s.writer_max_tokens,
    )


def build_agent_llm(settings: Settings | None = None) -> ChatOpenAI:
    """Return the agent LLM (gemma4-31b by default) — for tool-calling and chat."""
    s = settings or get_settings()
    return ChatOpenAI(
        base_url=s.llm_base_url,
        api_key=s.llm_api_key,
        model=s.agent_model,
        temperature=s.agent_temperature,
        max_tokens=s.agent_max_tokens,
    )


def build_eval_llm(settings: Settings | None = None) -> ChatOpenAI:
    """Return the evaluator LLM (nemotron-120b by default) — heavier model for judgment."""
    s = settings or get_settings()
    return ChatOpenAI(
        base_url=s.llm_base_url,
        api_key=s.llm_api_key,
        model=s.eval_model,
        temperature=s.eval_temperature,
        max_tokens=s.eval_max_tokens,
    )


# ---------------------------------------------------------------------------
# Embedding factory (LlamaIndex)
# ---------------------------------------------------------------------------

_embeddings_cache: dict[tuple[str, str, bool], object] = {}


def build_embeddings(settings: Settings | None = None):
    """Return a LlamaIndex embedding model, cached by (model, base_url, local).

    By default, uses the shared ``LLM_BASE_URL`` OpenAI-compatible endpoint —
    no local GPU or PyTorch needed.

    Set ``EMBEDDING_LOCAL=true`` to run embeddings locally via
    sentence-transformers, which requires the ``local`` extra::

        uv pip install -e ".[local]"
    """
    s = settings or get_settings()
    cache_key = (s.embedding_model, s.llm_base_url, s.embedding_local)

    if cache_key in _embeddings_cache:
        return _embeddings_cache[cache_key]

    if not s.embedding_local:
        from llama_index.embeddings.openai import OpenAIEmbedding

        emb = OpenAIEmbedding(
            api_base=s.llm_base_url,
            api_key=s.llm_api_key,
            model_name=s.embedding_model,
        )
    else:
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        except ImportError:
            raise ImportError(
                "Local embeddings require the 'local' extra (sentence-transformers + PyTorch). "
                "Install with: uv pip install -e \".[local]\"\n"
                "Or unset EMBEDDING_LOCAL to use the shared LLM_BASE_URL endpoint instead."
            ) from None

        emb = HuggingFaceEmbedding(
            model_name=s.embedding_model,
            trust_remote_code=True,
        )

    _embeddings_cache[cache_key] = emb
    return emb
