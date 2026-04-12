"""Tests for the rule-based style checker."""

from strategy_agent.tools.style_check import check_style


def test_detects_banned_phrases():
    text = "We need to leverage our synergies to unlock value and move the needle."
    result = check_style.invoke({"text": text})
    assert "BANNED PHRASE" in result
    assert "synerg" in result.lower()
    assert "leverage" in result.lower()
    assert "unlock value" in result.lower()
    assert "move the needle" in result.lower()


def test_detects_british_spellings():
    text = "The organisation must analyse its behaviour and optimise accordingly."
    result = check_style.invoke({"text": text})
    assert "BRITISH SPELLING" in result
    assert "organization" in result
    assert "analyze" in result
    assert "behavior" in result
    assert "optimize" in result


def test_detects_long_paragraphs():
    # 8 sentences in one paragraph
    text = ". ".join([f"This is sentence number {i}" for i in range(8)]) + "."
    result = check_style.invoke({"text": text})
    assert "LONG PARAGRAPH" in result


def test_passes_clean_text():
    text = (
        "Revenue grew 14% year-over-year. That performance exceeded the "
        "industry average by eight points. Three factors drove the gain."
    )
    result = check_style.invoke({"text": text})
    assert "passed" in result.lower()


def test_detects_passive_voice():
    sentences = [
        "The report was written by the team.",
        "Costs were reduced by management.",
        "The strategy was approved by the board.",
        "Results were analyzed by the team.",
        "The total was done correctly.",
    ]
    text = " ".join(sentences)
    result = check_style.invoke({"text": text})
    assert "PASSIVE VOICE" in result
