"""CLI entry point for the strategy-agent framework.

Usage examples::

    # Ingest a corpus of strategy documents
    strategy-agent ingest ./corpus

    # Ingest with knowledge graph extraction
    strategy-agent ingest ./corpus --build-kg

    # Generate a strategy memo (batch mode)
    strategy-agent generate --type strategy_memo \\
        --brief "Evaluate whether we should enter the European market"

    # Interactive chat (new session)
    strategy-agent chat

    # Resume a previous session
    strategy-agent chat --session <thread_id>

    # List all chat sessions
    strategy-agent chat --list

    # Show corpus / KG stats
    strategy-agent corpus-info
    strategy-agent kg-info
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="strategy-agent",
    help="LangChain deep agent for corporate strategy document generation.",
    no_args_is_help=True,
)
console = Console()

VALID_DOCUMENT_TYPES = {
    "strategy_memo",
    "white_paper",
    "board_presentation",
    "competitive_analysis",
    "market_assessment",
}


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=True)],
    )


# ── Ingest command ────────────────────────────────────────────────────────────

@app.command()
def ingest(
    corpus_dir: Path = typer.Argument(
        ...,
        help="Path to the directory containing strategy documents to ingest.",
        exists=True,
        dir_okay=True,
        file_okay=False,
    ),
    reset: bool = typer.Option(
        False,
        "--reset",
        help="Delete existing vector store, knowledge graph, and registry before ingesting.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-ingest files even if they are already in the registry.",
    ),
    build_kg: bool = typer.Option(
        False,
        "--build-kg",
        help="Extract entities/relationships into the knowledge graph (requires LLM calls).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Ingest a corpus of strategy documents into the vector store and optionally the knowledge graph."""
    _setup_logging(verbose)

    from strategy_agent.config import get_settings
    from strategy_agent.ingestion.chunker import chunk_documents
    from strategy_agent.ingestion.loader import load_corpus, load_file
    from strategy_agent.ingestion.registry import IngestionRegistry
    from strategy_agent.memory.vector_store import CorpusStore

    settings = get_settings()
    registry = IngestionRegistry(settings.ingestion_registry_path)

    store = CorpusStore(settings)
    if reset:
        console.print("[yellow]Resetting existing vector store...")
        store.reset()
        console.print("[yellow]Resetting ingestion registry...")
        registry.reset()

    # ── Load with duplicate detection ────────────────────────────────────
    if force or reset:
        with console.status("[bold green]Loading documents..."):
            docs = load_corpus(corpus_dir)
        skipped_files: list[str] = []
    else:
        docs = []
        skipped_files = []
        supported_paths = sorted(
            p for p in corpus_dir.rglob("*")
            if p.is_file() and not p.name.startswith(".")
        )
        with console.status("[bold green]Loading documents (checking for duplicates)..."):
            for path in supported_paths:
                if registry.is_ingested(path):
                    skipped_files.append(path.name)
                    continue
                docs.extend(load_file(path))

    if skipped_files:
        console.print(
            f"[dim]Skipped [bold]{len(skipped_files)}[/bold] already-ingested file(s): "
            f"{', '.join(skipped_files[:5])}"
            f"{'...' if len(skipped_files) > 5 else ''}"
        )

    if not docs:
        if skipped_files:
            console.print("[green]All files already ingested. Use --force to re-ingest.")
        else:
            console.print("[yellow]No supported documents found in the corpus directory.")
        raise typer.Exit(0 if skipped_files else 1)

    console.print(f"Loaded [bold]{len(docs)}[/bold] new document(s)")

    with console.status("[bold green]Chunking..."):
        chunks = chunk_documents(docs, settings)

    console.print(f"Split into [bold]{len(chunks)}[/bold] chunks")

    with console.status("[bold green]Embedding and storing..."):
        stored = store.add_documents(chunks)

    # ── Register ingested files ──────────────────────────────────────────
    ingested_sources: set[str] = set()
    for chunk in chunks:
        src_path = chunk.metadata.get("source_path")
        if src_path and src_path not in ingested_sources:
            ingested_sources.add(src_path)
    for src_path in ingested_sources:
        p = Path(src_path)
        chunk_count = sum(
            1 for c in chunks if c.metadata.get("source_path") == src_path
        )
        if p.exists():
            registry.register(p, chunk_count)
    registry.save()

    console.print(
        Panel(
            f"[green]Successfully indexed {stored} document(s) in "
            f"{settings.index_persist_dir}\n"
            f"Registry: {registry.count} file(s) tracked",
            title="Vector Store Ingestion Complete",
        )
    )

    # ── Optional knowledge graph extraction ───────────────────────────────
    if build_kg:
        from strategy_agent.ingestion.kg_extractor import extract_triples_batch
        from strategy_agent.memory.knowledge_graph import KnowledgeGraphStore
        from strategy_agent.models import build_writer_llm

        kg = KnowledgeGraphStore(settings)
        if reset:
            console.print("[yellow]Resetting existing knowledge graph...")
            kg.reset()

        llm = build_writer_llm(settings)
        total_added = 0
        total_skipped = 0

        with console.status("[bold green]Extracting knowledge graph triples...") as status:
            def _update_progress(done: int, total: int) -> None:
                status.update(
                    f"[bold green]Extracting triples... chunk {done}/{total}"
                )

            texts = [chunk.page_content for chunk in chunks]
            all_triples = extract_triples_batch(
                texts, llm, max_workers=10, on_progress=_update_progress,
            )
            for triples in all_triples:
                added, skipped = kg.add_triples_if_new(triples)
                total_added += added
                total_skipped += skipped

        kg.save()
        console.print(
            Panel(
                f"[green]Added {total_added} new triples ({total_skipped} duplicates skipped, "
                f"{kg.num_entities} total entities) into knowledge graph",
                title="Knowledge Graph Extraction Complete",
            )
        )


# ── Generate command ──────────────────────────────────────────────────────────

@app.command()
def generate(
    brief: str | None = typer.Option(
        None,
        "--brief", "-b",
        help="The writing brief describing what strategy document to produce.",
    ),
    brief_file: Path | None = typer.Option(
        None,
        "--brief-file", "-f",
        help="Path to a file containing the writing brief.",
        exists=True,
    ),
    document_type: str = typer.Option(
        "strategy_memo",
        "--type", "-t",
        help="Document type: strategy_memo, white_paper, board_presentation, competitive_analysis, market_assessment.",
    ),
    instructions: str = typer.Option(
        "",
        "--instructions", "-i",
        help="Additional instructions for the writing agents.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output", "-o",
        help="Output file path.  Defaults to ./output/<type>_<timestamp>.md",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Generate a strategy document from the ingested corpus."""
    _setup_logging(verbose)

    # Validate document type
    if document_type not in VALID_DOCUMENT_TYPES:
        console.print(f"[red]Invalid document type: '{document_type}'")
        console.print(f"Valid types: {', '.join(sorted(VALID_DOCUMENT_TYPES))}")
        raise typer.Exit(1)

    # Resolve the brief
    if brief_file:
        brief_text = brief_file.read_text(encoding="utf-8")
    elif brief:
        brief_text = brief
    else:
        console.print("[red]Provide either --brief or --brief-file")
        raise typer.Exit(1)

    from strategy_agent.config import get_settings
    from strategy_agent.orchestrator import run_pipeline

    settings = get_settings()

    console.print(Panel(brief_text, title="Writing Brief", subtitle=document_type))
    console.print()

    with console.status("[bold green]Running strategy pipeline..."):
        memory = run_pipeline(
            brief=brief_text,
            document_type=document_type,
            additional_instructions=instructions,
            settings=settings,
        )

    # Display evaluation history
    if memory.evaluations:
        table = Table(title="Evaluation History")
        table.add_column("Iter", style="cyan", justify="center")
        table.add_column("Overall", style="bold", justify="center")
        table.add_column("Story", justify="center")
        table.add_column("Cohesion", justify="center")
        table.add_column("Data", justify="center")
        table.add_column("Style", justify="center")
        table.add_column("Accepted", justify="center")

        for i, ev in enumerate(memory.evaluations, 1):
            table.add_row(
                str(i),
                f"{ev.overall_score:.1f}",
                f"{ev.storytelling_score:.1f}",
                f"{ev.narrative_cohesion_score:.1f}",
                f"{ev.data_integration_score:.1f}",
                f"{ev.style_compliance_score:.1f}",
                "[green]Yes" if ev.passes_threshold else "[red]No",
            )

        console.print(table)

        # Warn if any evaluation had a parse failure
        if any(ev.parse_failed for ev in memory.evaluations):
            console.print(
                "[yellow]Warning: One or more evaluation passes failed to parse "
                "the LLM's JSON output.  The document was accepted but may benefit "
                "from manual review."
            )

        console.print()

    # Write output
    if output is None:
        from datetime import datetime, timezone

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output = settings.output_dir / f"{document_type}_{timestamp}.md"

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(memory.latest_draft, encoding="utf-8")

    console.print(
        Panel(
            f"[green]Document written to: {output}\n"
            f"Iterations: {memory.current_iteration}\n"
            f"Final length: {len(memory.latest_draft):,} characters",
            title="Generation Complete",
        )
    )


# ── Corpus info command ───────────────────────────────────────────────────────

@app.command(name="corpus-info")
def corpus_info(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Show information about the ingested corpus."""
    _setup_logging(verbose)

    from strategy_agent.config import get_settings
    from strategy_agent.memory.vector_store import CorpusStore

    settings = get_settings()
    store = CorpusStore(settings)

    count = store.count
    console.print(
        Panel(
            f"Documents indexed: [bold]{count:,}[/bold]\n"
            f"Persist directory: {settings.index_persist_dir}\n"
            f"Embedding model: {settings.embedding_model}",
            title="Corpus Info",
        )
    )


# ── Corpus remove command ────────────────────────────────────────────────────

@app.command(name="corpus-remove")
def corpus_remove(
    source_file: str = typer.Argument(
        ...,
        help="Name of the source file to remove (e.g. 'report.pdf').",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Remove a specific document from the vector store, knowledge graph, and registry."""
    _setup_logging(verbose)

    from strategy_agent.config import get_settings
    from strategy_agent.ingestion.registry import IngestionRegistry
    from strategy_agent.memory.knowledge_graph import KnowledgeGraphStore
    from strategy_agent.memory.vector_store import CorpusStore

    settings = get_settings()

    # ── Vector store ──────────────────────────────────────────────────────
    store = CorpusStore(settings)
    available = store.get_source_files()
    if source_file not in available:
        console.print(f"[yellow]'{source_file}' not found in vector store.")
        if available:
            console.print(f"[dim]Available files: {', '.join(available)}")
        raise typer.Exit(1)

    removed_nodes = store.remove_document(source_file)
    console.print(f"Removed [bold]{removed_nodes}[/bold] node(s) from vector store")

    # ── Knowledge graph (remove entity matching file stem) ────────────────
    kg = KnowledgeGraphStore(settings)
    if kg.num_triples > 0:
        # Remove triples mentioning the file name (without extension)
        removed_triples = kg.remove_entity(Path(source_file).stem)
        if removed_triples:
            kg.save()
            console.print(
                f"Removed [bold]{removed_triples}[/bold] triple(s) from knowledge graph"
            )

    # ── Registry ──────────────────────────────────────────────────────────
    registry = IngestionRegistry(settings.ingestion_registry_path)
    if registry.unregister_by_source(source_file):
        registry.save()
        console.print("Removed from ingestion registry")

    console.print(Panel(f"[green]'{source_file}' removed successfully", title="Done"))


# ── Corpus dedup command ─────────────────────────────────────────────────────

@app.command(name="corpus-dedup")
def corpus_dedup(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Report duplicates without removing them.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Find and remove duplicate entries in the vector store and knowledge graph."""
    _setup_logging(verbose)

    from strategy_agent.config import get_settings
    from strategy_agent.memory.knowledge_graph import KnowledgeGraphStore
    from strategy_agent.memory.vector_store import CorpusStore

    settings = get_settings()

    # ── Vector store dedup ────────────────────────────────────────────────
    store = CorpusStore(settings)
    if dry_run:
        # Count duplicates without removing
        import hashlib

        docstore = store._index.storage_context.docstore
        seen: dict[tuple[str, str], str] = {}
        dup_count = 0
        for nid in sorted(docstore.docs):
            node = docstore.docs[nid]
            text_hash = hashlib.sha256(node.get_content().encode()).hexdigest()
            key = (node.metadata.get("source_file", ""), text_hash)
            if key in seen:
                dup_count += 1
            else:
                seen[key] = nid
        console.print(f"Vector store: [bold]{dup_count}[/bold] duplicate node(s) found")
    else:
        removed_vdb = store.deduplicate()
        console.print(f"Vector store: removed [bold]{removed_vdb}[/bold] duplicate node(s)")

    # ── Knowledge graph dedup ─────────────────────────────────────────────
    kg = KnowledgeGraphStore(settings)
    if kg.num_triples > 0:
        if dry_run:
            triples = kg.get_all_triples()
            seen_triples: set[tuple[str, str, str]] = set()
            dup_triple_count = 0
            for s, p, o in triples:
                key = (s.lower(), p.lower(), o.lower())
                if key in seen_triples:
                    dup_triple_count += 1
                else:
                    seen_triples.add(key)
            console.print(
                f"Knowledge graph: [bold]{dup_triple_count}[/bold] duplicate triple(s) found"
            )
        else:
            removed_kg = kg.deduplicate()
            if removed_kg:
                kg.save()
            console.print(
                f"Knowledge graph: removed [bold]{removed_kg}[/bold] duplicate triple(s)"
            )
    else:
        console.print("[dim]Knowledge graph is empty — nothing to deduplicate")

    if dry_run:
        console.print("\n[dim]Dry run — no changes made. Remove --dry-run to clean up.")


# ── Style check command ──────────────────────────────────────────────────────

@app.command(name="style-check")
def style_check_cmd(
    file: Path = typer.Argument(
        ...,
        help="Path to a document to check against the house style.",
        exists=True,
    ),
) -> None:
    """Run the house style checker on an existing document."""
    text = file.read_text(encoding="utf-8")

    from strategy_agent.tools.style_check import check_style

    result = check_style.invoke({"text": text})
    console.print(Panel(result, title=f"Style Check: {file.name}"))


# ── Chat command ──────────────────────────────────────────────────────────────

@app.command()
def chat(
    session: str | None = typer.Option(
        None,
        "--session", "-s",
        help="Resume a previous session by thread ID.",
    ),
    list_sessions: bool = typer.Option(
        False,
        "--list", "-l",
        help="List all saved chat sessions and exit.",
    ),
    title: str | None = typer.Option(
        None,
        "--title",
        help="Title for a new session.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Start an interactive chat session with the strategy agent."""
    _setup_logging(verbose)

    from strategy_agent.agents.chat_agent import build_chat_agent
    from strategy_agent.config import get_settings
    from strategy_agent.session import SessionManager

    settings = get_settings()
    sm = SessionManager(settings)

    # ── List mode ─────────────────────────────────────────────────────────
    if list_sessions:
        sessions = sm.list_sessions()
        if not sessions:
            console.print("[dim]No saved sessions.")
            raise typer.Exit(0)

        table = Table(title="Chat Sessions")
        table.add_column("Thread ID", style="cyan")
        table.add_column("Title", style="bold")
        table.add_column("Created", style="dim")
        table.add_column("Last Active", style="dim")

        for s in sessions:
            table.add_row(s["thread_id"], s["title"], s["created_at"][:19], s["updated_at"][:19])

        console.print(table)
        sm.close()
        raise typer.Exit(0)

    # ── Resolve or create session ─────────────────────────────────────────
    if session:
        thread_id = session
        if not sm.session_exists(thread_id):
            console.print(f"[yellow]Session '{thread_id}' not found — creating it.")
            thread_id = sm.create_session(title or f"Session {thread_id[:8]}")
        label = "resumed"
    else:
        thread_id = sm.create_session(title)
        label = "new"

    config = sm.get_config(thread_id)
    agent = build_chat_agent(sm, settings)

    # Reuse the stores already created by build_chat_agent (avoid double init)
    from strategy_agent.tools.corpus_search import _get_store as _get_corpus
    from strategy_agent.tools.knowledge_graph import _get_store as _get_kg

    corpus_count = _get_corpus().count
    kg_count = _get_kg().num_entities

    console.print(
        Panel(
            f"Session: [bold]{thread_id}[/bold] ({label})\n"
            f"Corpus: {corpus_count:,} chunks  |  KG: {kg_count:,} entities\n"
            f"Type [bold]/help[/bold] for commands, [bold]/quit[/bold] to exit",
            title="Strategy Agent",
        )
    )

    # ── REPL loop ─────────────────────────────────────────────────────────
    try:
        while True:
            try:
                user_input = console.input("\n[bold cyan]You:[/bold cyan] ")
            except (EOFError, KeyboardInterrupt):
                break

            stripped = user_input.strip()
            if not stripped:
                continue

            # Slash commands
            if stripped.lower() in ("/quit", "/exit", "/q"):
                break
            if stripped.lower() == "/help":
                console.print(
                    Panel(
                        "[bold]/quit[/bold]     — End the session\n"
                        "[bold]/sessions[/bold] — List all saved sessions\n"
                        "[bold]/export[/bold]   — Export this conversation to a file\n"
                        "[bold]/help[/bold]     — Show this help",
                        title="Commands",
                    )
                )
                continue
            if stripped.lower() == "/sessions":
                for s in sm.list_sessions():
                    marker = " [bold green]<-- current[/bold green]" if s["thread_id"] == thread_id else ""
                    console.print(f"  {s['thread_id']}  {s['title']}{marker}")
                continue
            if stripped.lower() == "/export":
                try:
                    state = agent.get_state(config)
                    messages = state.values.get("messages", [])
                    if not messages:
                        console.print("[yellow]No messages to export.")
                        continue
                    export_path = settings.output_dir / f"chat_{thread_id}.md"
                    export_path.parent.mkdir(parents=True, exist_ok=True)
                    lines: list[str] = []
                    for m in messages:
                        role = getattr(m, "type", "unknown").title()
                        lines.append(f"### {role}\n\n{m.content}\n")
                    export_path.write_text("\n---\n\n".join(lines), encoding="utf-8")
                    console.print(f"[green]Exported {len(messages)} messages to {export_path}")
                except Exception as e:
                    console.print(f"[red]Export failed: {e}")
                continue

            # Reject unknown slash commands
            if stripped.startswith("/"):
                console.print(f"[yellow]Unknown command: {stripped}. Type /help for available commands.")
                continue

            # Send to agent
            console.print()
            sm.touch_session(thread_id)

            try:
                with console.status("[bold green]Thinking..."):
                    result = agent.invoke(
                        {"messages": [("human", stripped)]},
                        config=config,
                    )

                # Extract and display the AI response
                ai_message = result["messages"][-1]
                console.print(
                    Panel(
                        ai_message.content,
                        title="[bold green]Strategy Agent[/bold green]",
                        border_style="green",
                    )
                )
            except Exception as e:
                console.print(Panel(f"[red]{e}", title="Error", border_style="red"))
    finally:
        sm.close()
        console.print(f"\n[dim]Session saved.  Resume with: strategy-agent chat --session {thread_id}[/dim]")


# ── KG info command ───────────────────────────────────────────────────────────

@app.command(name="kg-info")
def kg_info(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Show knowledge graph statistics."""
    _setup_logging(verbose)

    from strategy_agent.config import get_settings
    from strategy_agent.memory.knowledge_graph import KnowledgeGraphStore

    settings = get_settings()
    kg = KnowledgeGraphStore(settings)

    console.print(
        Panel(
            f"Triples: [bold]{kg.num_triples:,}[/bold]\n"
            f"Entities: [bold]{kg.num_entities:,}[/bold]\n"
            f"Graph file: {settings.kg_gml_path}",
            title="Knowledge Graph Info",
        )
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
