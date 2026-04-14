"""Centralized configuration driven by environment variables / .env file.

All LLMs and the embedding model share a single OpenAI-compatible endpoint
(``LLM_BASE_URL`` + ``LLM_API_KEY``).  The framework distinguishes three
model roles by ``*_MODEL`` name only:

- **Writer** (``WRITER_MODEL``): Primary content generation — drafts and
  rewrites.  Optimized for fluency and style adherence.  Default: ``gpt-oss``.
- **Agent** (``AGENT_MODEL``): Orchestration, research, tool-calling, chat.
  Needs strong instruction-following and tool use.  Default: ``gemma4-31b``.
- **Evaluator** (``EVAL_MODEL``): Narrative scoring and quality assessment.
  Benefits from a heavier model for better judgment.  Default: ``nemotron-120b``.

Per-role temperature and max-token caps are also configurable independently.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All tunables for the strategy-agent framework.

    Values are loaded from environment variables first, then from a ``.env``
    file in the project root.  Every field has a sensible default so the
    framework can be explored immediately after cloning the repo.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Shared LLM endpoint (auth used by all roles + embeddings) ────────
    llm_base_url: str = Field("http://localhost:8000/v1", alias="LLM_BASE_URL")
    llm_api_key: str = Field("not-needed", alias="LLM_API_KEY")

    # ── Writer LLM (drafts, rewrites — optimized for prose quality) ──────
    writer_model: str = Field("gpt-oss", alias="WRITER_MODEL")
    writer_temperature: float = Field(0.7, alias="WRITER_TEMPERATURE")
    writer_max_tokens: int = Field(4096, alias="WRITER_MAX_TOKENS")

    # ── Agent LLM (research, chat, tool-calling — needs tool use) ────────
    agent_model: str = Field("gemma4-31b", alias="AGENT_MODEL")
    agent_temperature: float = Field(0.4, alias="AGENT_TEMPERATURE")
    agent_max_tokens: int = Field(4096, alias="AGENT_MAX_TOKENS")

    # ── Evaluator LLM (scoring — heavier model for better judgment) ──────
    eval_model: str = Field("nemotron-120b", alias="EVAL_MODEL")
    eval_temperature: float = Field(0.3, alias="EVAL_TEMPERATURE")
    eval_max_tokens: int = Field(4096, alias="EVAL_MAX_TOKENS")

    # ── Embeddings (share the LLM endpoint by default) ───────────────────
    # Set ``EMBEDDING_LOCAL=true`` to use HuggingFace sentence-transformers
    # locally instead of the remote endpoint (requires the ``local`` extra).
    embedding_model: str = Field(
        "nomic-ai/nomic-embed-text-v1.5", alias="EMBEDDING_MODEL"
    )
    embedding_local: bool = Field(False, alias="EMBEDDING_LOCAL")

    # ── LlamaIndex vector store ───────────────────────────────────────────
    index_persist_dir: Path = Field(
        Path("./data/index"), alias="INDEX_PERSIST_DIR"
    )

    # ── Paths ─────────────────────────────────────────────────────────────
    corpus_dir: Path = Field(Path("./corpus"), alias="CORPUS_DIR")
    output_dir: Path = Field(Path("./output"), alias="OUTPUT_DIR")

    # ── Knowledge graph ───────────────────────────────────────────────────
    kg_gml_path: Path = Field(
        Path("./data/knowledge_graph.gml"), alias="KNOWLEDGE_GRAPH_PATH"
    )

    # ── Ingestion registry ────────────────────────────────────────────────
    ingestion_registry_path: Path = Field(
        Path("./data/ingestion_registry.json"), alias="INGESTION_REGISTRY_PATH"
    )

    # ── Chat & session persistence ────────────────────────────────────────
    session_db_path: Path = Field(
        Path("./data/sessions.db"), alias="SESSION_DB_PATH"
    )

    # ── Agent tuning ──────────────────────────────────────────────────────
    max_rewrite_loops: int = Field(3, alias="MAX_REWRITE_LOOPS")
    chunk_size: int = Field(1500, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(300, alias="CHUNK_OVERLAP")
    retrieval_top_k: int = 12
    # Number of top-scoring chunks to expand into full source documents
    # for long-context stuffing (0 = skip stuffing, just use chunks)
    context_stuffing_docs: int = Field(5, alias="CONTEXT_STUFFING_DOCS")


def get_settings() -> Settings:
    """Return a Settings instance."""
    return Settings()
