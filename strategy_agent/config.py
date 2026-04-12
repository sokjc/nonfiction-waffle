"""Centralized configuration driven by environment variables / .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All tunables for the strategy-agent framework.

    Values are loaded from environment variables first, then from a `.env` file
    in the project root.  Every field has a sensible default so the framework
    can be explored immediately after cloning the repo.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Primary LLM (writer / researcher / rewriter) ─────────────────────────
    llm_base_url: str = Field("http://localhost:8000/v1", alias="LLM_BASE_URL")
    llm_api_key: str = Field("not-needed", alias="LLM_API_KEY")
    llm_model: str = Field("gemma4-31b", alias="LLM_MODEL")
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096

    # ── Evaluation LLM (narrative evaluator — can be a heavier model) ────────
    eval_base_url: str | None = Field(None, alias="EVAL_BASE_URL")
    eval_api_key: str | None = Field(None, alias="EVAL_API_KEY")
    eval_model: str | None = Field(None, alias="EVAL_MODEL")
    eval_temperature: float = 0.3

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = Field("all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    embedding_base_url: str | None = Field(None, alias="EMBEDDING_BASE_URL")
    embedding_api_key: str | None = Field(None, alias="EMBEDDING_API_KEY")

    # ── ChromaDB vector store ─────────────────────────────────────────────────
    chroma_persist_dir: Path = Field(Path("./data/vectorstore"), alias="CHROMA_PERSIST_DIR")
    chroma_collection: str = Field("strategy_corpus", alias="CHROMA_COLLECTION")

    # ── Paths ─────────────────────────────────────────────────────────────────
    corpus_dir: Path = Field(Path("./corpus"), alias="CORPUS_DIR")
    output_dir: Path = Field(Path("./output"), alias="OUTPUT_DIR")

    # ── Agent tuning ──────────────────────────────────────────────────────────
    max_rewrite_loops: int = Field(3, alias="MAX_REWRITE_LOOPS")
    chunk_size: int = Field(1500, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(300, alias="CHUNK_OVERLAP")
    retrieval_top_k: int = 12

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def eval_base_url_resolved(self) -> str:
        return self.eval_base_url or self.llm_base_url

    @property
    def eval_api_key_resolved(self) -> str:
        return self.eval_api_key or self.llm_api_key

    @property
    def eval_model_resolved(self) -> str:
        return self.eval_model or self.llm_model


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
