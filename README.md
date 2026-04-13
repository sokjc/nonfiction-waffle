# Strategy Agent

A LangChain deep agent framework that reads a corpus of corporate strategy documents, builds semantic and structured memory, then writes detailed, contextually-aware strategy documents in a warm, witty, data-driven voice inspired by *The Economist*.

## Architecture

```
                         CLI REPL  (strategy-agent chat)
                              |
                    +---------+---------+
                    |  ReAct Chat Agent |---- SqliteSaver (cross-session memory)
                    +---------+---------+
                              |
             +----------------+----------------+
             |                |                |
       search_corpus   query_knowledge   generate_document
             |           _graph               |
             v                v               v
      +----------+    +-----------+    +--------------------------+
      | ChromaDB |    | NetworkX  |    |   Document Pipeline      |
      | Vector   |    | Knowledge |    |                          |
      | Store    |    | Graph     |    |  Research --> Write -->   |
      | (fuzzy   |    | (struct-  |    |  Evaluate --> Rewrite    |
      |  search) |    |  ured)    |    |       (loop until 8/10)  |
      +----------+    +-----------+    +--------------------------+
```

**Dual memory** — the vector store finds passages semantically similar to a query; the knowledge graph answers structured questions about entity relationships (who competes with whom, what markets a company operates in).

**Four specialist agents** run the document pipeline: a Researcher mines the corpus, a Writer drafts in the house style, an Evaluator scores on a 0-10 rubric (storytelling, cohesion, data integration, style compliance), and a Rewriter revises based on feedback. The loop continues until the draft scores 8+/10 or hits the iteration cap.

## Quick Start

```bash
# 1. Install
pip install -e ".[dev]"          # or: uv sync --dev

# 2. Configure
cp .env.example .env             # edit model endpoints for your servers

# 3. Ingest your strategy documents
strategy-agent ingest ./corpus

# 4. Chat with the agent
strategy-agent chat
```

### Three-Model Architecture

The framework uses three distinct model roles, each connectable to any OpenAI-compatible endpoint:

| Role | Default Model | Purpose | Config Prefix |
|------|--------------|---------|---------------|
| **Writer** | `gpt-oss` | Prose generation and rewriting | `WRITER_*` |
| **Agent** | `gemma4-31b` | Research, chat, tool-calling | `AGENT_*` |
| **Evaluator** | `nemotron-120b` | Narrative scoring and quality assessment | `EVAL_*` |

This avoids the "grading your own homework" problem — the evaluator is a *different model* than the writer, producing more honest quality assessments.

### Model Server Setup

All three roles connect via OpenAI-compatible endpoints. Common setups:

| Server | Command | Base URL |
|--------|---------|----------|
| **Ollama** | `ollama serve` | `http://localhost:11434/v1` |
| **vLLM** | `vllm serve gemma4-31b` | `http://localhost:8000/v1` |
| **LM Studio** | Start from GUI | `http://localhost:1234/v1` |

All three models can run on the same server (same URL, different model names) or on separate GPU servers (different URLs).

## CLI Reference

| Command | Description |
|---------|-------------|
| `strategy-agent ingest <dir>` | Load documents into the vector store |
| `strategy-agent ingest <dir> --build-kg` | Also extract a knowledge graph (requires LLM) |
| `strategy-agent generate -b "brief"` | Generate a strategy document (batch mode) |
| `strategy-agent chat` | Interactive chat session |
| `strategy-agent chat --list` | List all saved sessions |
| `strategy-agent chat -s <id>` | Resume a previous session |
| `strategy-agent corpus-info` | Show vector store statistics |
| `strategy-agent kg-info` | Show knowledge graph statistics |
| `strategy-agent style-check <file>` | Lint a document against the house style |

### Generate Options

```bash
strategy-agent generate \
    --brief "Evaluate whether we should expand into the European market" \
    --type strategy_memo \          # or: white_paper, board_presentation,
                                    #     competitive_analysis, market_assessment
    --instructions "Focus on regulatory risk" \
    --output ./output/eu_memo.md
```

## Chat Mode

The chat agent remembers your entire conversation across sessions. It can:

- **Answer questions** by searching the corpus and knowledge graph
- **Write full documents** by invoking the multi-agent pipeline as a tool
- **Learn new facts** and persist them in the knowledge graph

### Slash Commands

| Command | Action |
|---------|--------|
| `/help` | Show available commands |
| `/sessions` | List all saved sessions |
| `/export` | Export conversation to markdown |
| `/quit` | End the session |

Resume any session later with `strategy-agent chat --session <thread_id>`.

## Knowledge Graph

The hybrid memory system gives agents two complementary ways to query the corpus:

- **Vector store** (LlamaIndex) — *"Find passages about European market entry"* — fuzzy, semantic
- **Knowledge graph** (NetworkX) — *"What entities are connected to Acme Corp?"* — structured, relational

Build the knowledge graph during ingestion:

```bash
strategy-agent ingest ./corpus --build-kg
```

This uses the LLM to extract entity-relationship triples (companies, markets, strategies, metrics) from every document chunk. The graph persists as a GML file and grows during chat conversations when new strategic intelligence is shared.

## The Voice

The house style guide encodes an *Economist*-inspired voice adapted for American English:

- **Warm authority** — confident without condescension, precise without pedantry
- **Wit with purpose** — dry observations that crystallize a point, never at the expense of clarity
- **Data as narrative fuel** — every statistic advances the argument ("14% margin in an industry averaging 6%")
- **No corporate fog** — *synergy*, *leverage*, and *unlock value* are banned; specificity is required
- **Narrative arc** — even a two-page memo has a hook, a build, and a quotable close

The evaluator agent scores every draft against this rubric and sends it back for revision until it passes.

## Configuration

All settings are controlled via environment variables or a `.env` file. See [`.env.example`](.env.example) for the full list with descriptions. Key tunables:

| Variable | Default | Description |
|----------|---------|-------------|
| `WRITER_MODEL` | `gpt-oss` | Model for prose generation |
| `AGENT_MODEL` | `gemma4-31b` | Model for research and chat |
| `EVAL_MODEL` | `nemotron-120b` | Model for quality evaluation |
| `MAX_REWRITE_LOOPS` | `3` | Max revision passes |
| `CHUNK_SIZE` | `1500` | Document chunk size (characters) |

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check strategy_agent/ tests/
```

### Project Structure

```
strategy_agent/
  agents/          # Researcher, Writer, Evaluator, Rewriter, Chat Agent
  ingestion/       # Document loaders, chunking, KG extraction
  memory/          # Vector store (ChromaDB), knowledge graph (NetworkX), working memory
  prompts/         # Style guide + prompt templates for every agent
  tools/           # Corpus search, KG query, style check, fact verification
  cli.py           # Typer CLI entry point
  config.py        # Pydantic settings from env vars
  errors.py        # Shared error handling for LLM calls
  models.py        # LLM and embedding factory functions
  orchestrator.py  # LangGraph pipeline (research -> write -> evaluate -> rewrite)
  session.py       # SQLite-backed session management
```

## License

MIT
