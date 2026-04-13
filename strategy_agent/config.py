"""Centralized configuration driven by environment variables / .env file.

The framework uses three distinct model roles:

- **Writer** (``WRITER_*``): Primary content generation — drafts and rewrites.
  Optimized for fluency and style adherence.  Default: ``gpt-oss``.
- **Agent** (``AGENT_*``): Orchestration, research, tool-calling, chat.
  Needs strong instruction-following and tool use.  Default: ``gemma4-31b``.
- **Evaluator** (``EVAL_*``): Narrative scoring and quality assessment.
  Benefits from a heavier model for better judgment.  Default: ``nemotron-120b``.

All three connect via OpenAI-compatible endpoints and can be swapped freely.
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

    # ── Writer LLM (drafts, rewrites — optimized for prose quality) ──────
    writer_base_url: str = Field("http://localhost:8000/v1", alias="WRITER_BASE_URL")
    writer_api_key: str = Field("not-needed", alias="WRITER_API_KEY")
    writer_model: str = Field("gpt-oss", alias="WRITER_MODEL")
    writer_temperature: float = 0.7
    writer_max_tokens: int = 4096

    # ── Agent LLM (research, chat, tool-calling — needs tool use) ────────
    agent_base_url: str = Field("http://localhost:8000/v1", alias="AGENT_BASE_URL")
    agent_api_key: str = Field("not-needed", alias="AGENT_API_KEY")
    agent_model: str = Field("gemma4-31b", alias="AGENT_MODEL")
    agent_temperature: float = 0.4
    agent_max_tokens: int = 4096

    # ── Evaluator LLM (scoring — heavier model for better judgment) ──────
    eval_base_url: str = Field("http://localhost:8000/v1", alias="EVAL_BASE_URL")
    eval_api_key: str = Field("not-needed", alias="EVAL_API_KEY")
    eval_model: str = Field("nemotron-120b", alias="EVAL_MODEL")
    eval_temperature: float = 0.3
    eval_max_tokens: int = 4096

    # ── Embeddings (LlamaIndex + Nomic) ───────────────────────────────────
    embedding_model: str = Field(
        "nomic-ai/nomic-embed-text-v1.5", alias="EMBEDDING_MODEL"
    )
    embedding_base_url: str | None = Field(None, alias="EMBEDDING_BASE_URL")
    embedding_api_key: str | None = Field(None, alias="EMBEDDING_API_KEY")

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
