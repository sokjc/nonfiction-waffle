"""Session-scoped working memory for the multi-agent pipeline.

Working memory tracks the evolving state of a single document-generation run:
the brief, retrieved context, successive drafts, evaluation scores, and the
final output.  It is passed through the LangGraph state so every agent can
read prior outputs and append its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class EvaluationResult:
    """Structured scorecard produced by the narrative evaluator."""

    storytelling_score: float = 0.0        # 0-10
    narrative_cohesion_score: float = 0.0  # 0-10
    data_integration_score: float = 0.0    # 0-10
    style_compliance_score: float = 0.0    # 0-10
    overall_score: float = 0.0             # 0-10
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    rewrite_instructions: str = ""
    passes_threshold: bool = False

    @property
    def summary(self) -> str:
        return (
            f"Overall {self.overall_score:.1f}/10 — "
            f"Story {self.storytelling_score:.1f} · "
            f"Cohesion {self.narrative_cohesion_score:.1f} · "
            f"Data {self.data_integration_score:.1f} · "
            f"Style {self.style_compliance_score:.1f}"
        )


@dataclass
class WorkingMemory:
    """Mutable state carried through the LangGraph pipeline."""

    # Inputs
    brief: str = ""
    document_type: str = "strategy_memo"
    additional_instructions: str = ""

    # Research phase
    retrieved_context: list[str] = field(default_factory=list)
    research_synthesis: str = ""

    # Drafting / rewriting
    drafts: list[str] = field(default_factory=list)
    evaluations: list[EvaluationResult] = field(default_factory=list)
    current_iteration: int = 0

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_used: str = ""

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def latest_draft(self) -> str:
        return self.drafts[-1] if self.drafts else ""

    @property
    def latest_evaluation(self) -> EvaluationResult | None:
        return self.evaluations[-1] if self.evaluations else None

    @property
    def is_accepted(self) -> bool:
        ev = self.latest_evaluation
        return ev.passes_threshold if ev else False
