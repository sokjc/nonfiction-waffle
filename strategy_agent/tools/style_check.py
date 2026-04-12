"""Style-checking tool — flags common violations of the house style guide.

This is a fast, rule-based pre-filter that catches obvious issues before
the evaluator LLM does its deeper assessment.  It is deliberately opinionated:
the style guide bans certain words and patterns outright.
"""

from __future__ import annotations

import re

from langchain_core.tools import tool

# ── Banned corporate fog ──────────────────────────────────────────────────────
_BANNED_PHRASES = [
    "synergy", "synergies", "synergistic",
    "leverage",  # as a verb — "leverage our capabilities"
    "unlock value",
    "move the needle",
    "best in class", "best-in-class",
    "world class", "world-class",  # ironic, but the style guide is specific
    "paradigm shift",
    "low-hanging fruit",
    "circle back",
    "take it offline",
    "at the end of the day",
    "going forward",
    "holistic approach",
    "deep dive",  # as a noun
    "value proposition",
    "core competency", "core competencies",
    "bleeding edge",
    "disruptive innovation",
    "thought leader", "thought leadership",
    "north star",
    "boil the ocean",
    "move the goalposts",
]

# ── British → American spelling catches ───────────────────────────────────────
_BRITISH_AMERICAN = {
    "analyse": "analyze",
    "analysing": "analyzing",
    "organisation": "organization",
    "organisations": "organizations",
    "organise": "organize",
    "organising": "organizing",
    "optimise": "optimize",
    "optimising": "optimizing",
    "recognise": "recognize",
    "recognising": "recognizing",
    "colour": "color",
    "colours": "colors",
    "favour": "favor",
    "favours": "favors",
    "behaviour": "behavior",
    "behaviours": "behaviors",
    "labour": "labor",
    "centre": "center",
    "centres": "centers",
    "defence": "defense",
    "licence": "license",
    "practise": "practice",
    "programme": "program",
    "programmes": "programs",
    "catalogue": "catalog",
    "dialogue": "dialog",
    "travelled": "traveled",
    "travelling": "traveling",
    "modelling": "modeling",
    "focussed": "focused",
    "focussing": "focusing",
    "judgement": "judgment",
    "ageing": "aging",
    "amongst": "among",
    "whilst": "while",
}

# ── Passive voice detector (simple heuristic) ────────────────────────────────
_PASSIVE_RE = re.compile(
    r"\b(is|are|was|were|been|being|be)\s+"
    r"(being\s+)?"
    r"(\w+ed|built|made|done|seen|known|given|taken|found|shown)\b",
    re.IGNORECASE,
)


@tool
def check_style(text: str) -> str:
    """Check a draft for house style violations.

    Scans for banned corporate jargon, British spellings, excessive passive
    voice, and overly long paragraphs.  Returns a structured report of
    violations found.

    Args:
        text: The draft text to check.

    Returns:
        A plain-text report listing style violations, or a confirmation
        that no major issues were found.
    """
    issues: list[str] = []

    # 1. Banned phrases
    text_lower = text.lower()
    for phrase in _BANNED_PHRASES:
        count = text_lower.count(phrase)
        if count:
            issues.append(f"BANNED PHRASE: \"{phrase}\" appears {count}x — replace with specific language")

    # 2. British spellings
    for british, american in _BRITISH_AMERICAN.items():
        pattern = re.compile(rf"\b{british}\b", re.IGNORECASE)
        matches = pattern.findall(text)
        if matches:
            issues.append(
                f"BRITISH SPELLING: \"{british}\" found {len(matches)}x — use \"{american}\""
            )

    # 3. Passive voice density
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    passive_count = sum(1 for s in sentences if _PASSIVE_RE.search(s))
    if sentences:
        passive_pct = passive_count / len(sentences) * 100
        if passive_pct > 25:
            issues.append(
                f"PASSIVE VOICE: {passive_pct:.0f}% of sentences use passive constructions "
                f"(target: <25%) — rewrite in active voice"
            )

    # 4. Long paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    for i, para in enumerate(paragraphs, 1):
        sentence_count = len([s for s in re.split(r"[.!?]+", para) if s.strip()])
        if sentence_count > 6:
            issues.append(
                f"LONG PARAGRAPH: Paragraph {i} has {sentence_count} sentences "
                f"(max 5) — split or tighten"
            )

    if not issues:
        return "Style check passed — no major violations detected."

    header = f"Found {len(issues)} style issue(s):\n"
    return header + "\n".join(f"  • {issue}" for issue in issues)
