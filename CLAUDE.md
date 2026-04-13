# CLAUDE.md

## Project Overview

**Strategy Agent** is a LangChain/LangGraph-based multi-agent framework that generates high-quality corporate strategy documents. It ingests a corpus of strategy documents, builds dual semantic (vector) and structured (knowledge graph) memory, then runs a multi-agent pipeline — research, write, evaluate, rewrite — to produce polished prose in an *Economist*-inspired voice.

Repository: `sokjc/nonfiction-waffle`

## Tech Stack

- **Language:** Python 3.11+ (3.11, 3.12)
- **Agent framework:** LangChain 0.3.x, LangGraph 0.3.x
- **Retrieval:** LlamaIndex 0.14+, ChromaDB (via LlamaIndex)
- **Embeddings:** sentence-transformers 3.0+ (Nomic v1.5 by default)
- **Knowledge graph:** NetworkX 3.2+ (persisted as GML)
- **CLI:** Typer 0.15+ with Rich formatting
- **Config:** Pydantic 2.9+ / pydantic-settings 2.6+ (env vars + `.env`)
- **Session persistence:** SQLite via langgraph-checkpoint-sqlite
- **Package manager:** [uv](https://docs.astral.sh/uv/)
- **Build system:** Hatchling (pyproject.toml)
- **Linting:** Ruff 0.8+
- **Testing:** Pytest 8.3+ with pytest-asyncio

## Quick Reference Commands

```bash
# Install (development) — uv is the primary package manager
uv sync --dev

# Alternative: pip install -e ".[dev]"

# Run tests
uv run pytest tests/ -v

# Run tests with coverage
uv run pytest tests/ --cov

# Lint
uv run ruff check strategy_agent/ tests/

# Lint with auto-fix
uv run ruff check --fix strategy_agent/ tests/

# Add a dependency
uv add <package>

# Add a dev dependency
uv add --dev <package>

# CLI commands (via uv run)
uv run strategy-agent ingest ./corpus              # Ingest documents
uv run strategy-agent ingest ./corpus --build-kg   # Ingest + build knowledge graph
uv run strategy-agent generate -b "brief text"     # Generate a strategy document
uv run strategy-agent chat                         # Interactive chat mode
uv run strategy-agent chat --list                  # List chat sessions
uv run strategy-agent corpus-info                  # Show corpus statistics
uv run strategy-agent kg-info                      # Show knowledge graph stats
uv run strategy-agent style-check <file>           # Lint a document for style
```

## Project Structure

```
strategy_agent/               # Main package
├── __init__.py
├── cli.py                    # Typer CLI entry point
├── config.py                 # Pydantic settings (all env vars)
├── errors.py                 # Custom exceptions (LLMConnectionError)
├── models.py                 # LLM/embedding factory functions
├── orchestrator.py           # LangGraph pipeline (research→write→evaluate→rewrite)
├── session.py                # SQLite-backed session management
├── agents/                   # Specialist agents
│   ├── chat_agent.py         # Interactive ReAct chat agent
│   ├── evaluator.py          # Scores drafts on 4 rubrics (0-10)
│   ├── researcher.py         # Corpus mining via hybrid retrieval
│   ├── rewriter.py           # Revises drafts based on eval feedback
│   └── writer.py             # Drafts in house style
├── ingestion/                # Document processing
│   ├── chunker.py            # Intelligent document chunking
│   ├── kg_extractor.py       # Entity-relation triple extraction
│   └── loader.py             # Multi-format loader (PDF, DOCX, MD, HTML, TXT)
├── memory/                   # Dual memory system
│   ├── knowledge_graph.py    # NetworkX persistent KG (GML format)
│   ├── vector_store.py       # LlamaIndex vector store with hybrid retrieval
│   └── working_memory.py     # Session-scoped pipeline state
├── prompts/                  # All LLM prompt templates
│   ├── chat.py               # Chat agent system prompt
│   ├── evaluator.py          # Evaluation rubric prompt
│   ├── researcher.py         # Research synthesis prompt
│   ├── rewriter.py           # Rewrite instructions prompt
│   ├── style_guide.py        # The Economist voice guide + forbidden phrases
│   └── writer.py             # Draft generation prompt
└── tools/                    # LangChain @tool-decorated functions
    ├── corpus_search.py      # Vector store search
    ├── fact_check.py         # Claim verification
    ├── knowledge_graph.py    # KG query/add operations
    └── style_check.py        # House style linting

tests/                        # 14 test modules (pytest)
corpus/                       # Input documents (user-provided)
data/                         # Persistent stores (auto-created)
├── index/                    # LlamaIndex vector index
├── knowledge_graph.gml       # NetworkX KG
└── sessions.db               # SQLite session DB
output/                       # Generated documents
```

## Architecture

### Three-Model System

The framework uses three distinct LLM roles, each configurable to a different model/endpoint:

| Role | Env Prefix | Default Model | Purpose |
|------|-----------|---------------|---------|
| **Writer** | `WRITER_*` | gpt-oss | Prose generation, rewrites (temp=0.7) |
| **Agent** | `AGENT_*` | gemma4-31b | Research, chat, tool-calling (temp=0.4) |
| **Evaluator** | `EVAL_*` | nemotron-120b | Quality scoring (temp=0.3) |

Using a different evaluator model avoids the "grading your own homework" problem. All connect via OpenAI-compatible endpoints.

### Multi-Agent Pipeline (LangGraph)

```
Research → Write → Evaluate → [score < 8?] → Rewrite → Evaluate → ... → Finalize
```

- Pipeline state flows as `PipelineState` (TypedDict with `WorkingMemory` + `Settings`)
- Conditional edges loop evaluate→rewrite until score >= 8 or `MAX_REWRITE_LOOPS` (default 3)
- Entry point: `run_pipeline()` in `orchestrator.py`

### Dual Memory System

1. **Vector Store** (LlamaIndex) — semantic similarity search, chunk-level retrieval expanded to full source documents
2. **Knowledge Graph** (NetworkX) — structured entity-relation triples, relational queries

### Hybrid Retrieval Strategy

1. Chunk-level similarity search (precision)
2. Expand to full source documents (completeness)
3. Stuff full docs into LLM context (leverages large context windows)
4. Configurable via `CONTEXT_STUFFING_DOCS` (default 5)

## Code Conventions

### Style Rules (enforced by Ruff)

- **Line length:** 100 characters max
- **Target:** Python 3.11+
- **Lint rules:** E (errors), F (pyflakes), I (isort), N (naming), W (warnings), UP (pyupgrade)
- **Imports:** sorted via isort rules (stdlib → third-party → local)

### Naming & Patterns

- Classes: `PascalCase` (e.g., `ResearchAgent`, `CorpusStore`, `WorkingMemory`)
- Functions/methods: `snake_case`
- Private members: prefixed with `_`
- Tool functions: decorated with `@tool` from LangChain
- Type hints: modern PEP 604 syntax (`str | None` not `Optional[str]`)
- `from __future__ import annotations` used for forward references

### Architectural Patterns

- **Factory pattern** in `models.py` for LLM/embedding construction
- **TypedDict** for graph state (`PipelineState`)
- **Dataclass** for structured data (`WorkingMemory`, `EvaluationResult`)
- **Singleton with thread safety** for module-level store instances (lock-protected)
- **Dependency injection** via constructor args with `get_settings()` fallback

### Error Handling

- Custom `LLMConnectionError` in `errors.py` for endpoint failures
- `invoke_llm()` wrapper catches connection issues and provides actionable messages
- JSON parse failures in evaluator are non-fatal — draft accepted with `parse_failed` flag
- Retry logic uses `tenacity`

## Testing

- **Framework:** Pytest with pytest-asyncio (asyncio_mode = "auto")
- **Location:** `tests/` directory
- **14 test modules** covering config, ingestion, memory, agents, orchestration, errors, and integration
- All tests are runnable with `uv run pytest tests/ -v`
- Tests mock LLM calls — no external services required to run the test suite

### Test Naming Pattern

Files: `test_<module>.py` (e.g., `test_config.py`, `test_knowledge_graph.py`)

## Configuration

All configuration is via environment variables or `.env` file. Every setting has a sensible default — no env vars are strictly required.

Copy `.env.example` to `.env` and adjust as needed:
```bash
cp .env.example .env
```

Key environment variables:
- `WRITER_BASE_URL`, `WRITER_API_KEY`, `WRITER_MODEL` — Writer LLM
- `AGENT_BASE_URL`, `AGENT_API_KEY`, `AGENT_MODEL` — Agent LLM
- `EVAL_BASE_URL`, `EVAL_API_KEY`, `EVAL_MODEL` — Evaluator LLM
- `EMBEDDING_MODEL` — HuggingFace embedding model name
- `INDEX_PERSIST_DIR`, `CORPUS_DIR`, `OUTPUT_DIR` — File paths
- `MAX_REWRITE_LOOPS`, `CHUNK_SIZE`, `CHUNK_OVERLAP` — Tuning parameters

See `strategy_agent/config.py` for the full `Settings` class with all defaults.

## Data Directories

| Directory | Purpose | Committed |
|-----------|---------|-----------|
| `corpus/` | Input strategy documents | No (user-provided) |
| `data/index/` | LlamaIndex vector index | No (generated) |
| `data/knowledge_graph.gml` | NetworkX KG | No (generated) |
| `data/sessions.db` | SQLite session store | No (generated) |
| `output/` | Generated strategy documents | No (generated) |

## Guidelines for AI Assistants

- Run `uv run ruff check strategy_agent/ tests/` before committing to catch lint issues
- Run `uv run pytest tests/ -v` to verify changes don't break existing tests
- Use `uv add <package>` / `uv add --dev <package>` to manage dependencies (not `pip install`)
- When adding new modules, follow the existing package structure (agents/, memory/, tools/, etc.)
- When adding new agents, follow the pattern in existing agents: constructor takes `llm` + `settings`, exposes a `run(memory)` method
- New tools should use the `@tool` decorator and go in `strategy_agent/tools/`
- New prompts go in `strategy_agent/prompts/` as module-level string constants
- Keep the three-model separation — writer for prose, agent for tool-calling/research, evaluator for scoring
- Configuration changes go through `Settings` in `config.py` with env var aliases
- All LLM endpoints are OpenAI-compatible — do not add provider-specific code
- The `.env` file should never be committed; use `.env.example` for documentation
