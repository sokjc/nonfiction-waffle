"""Tests for the KG triple extraction logic."""

import json

from strategy_agent.ingestion.kg_extractor import extract_triples


class MockLLM:
    """Fake LLM that returns canned JSON for extraction tests."""

    def __init__(self, response: str):
        self._response = response

    def invoke(self, *args, **kwargs):
        return self._response


class MockChain:
    """Fake chain that returns canned output."""

    def __init__(self, output: str):
        self._output = output

    def invoke(self, *args, **kwargs):
        return self._output


def test_extract_triples_valid_json(monkeypatch):
    """Should parse well-formed JSON triples from LLM output."""
    triples_data = [
        {"subject": "Acme Corp", "predicate": "competes with", "object": "WidgetCo"},
        {"subject": "Acme Corp", "predicate": "revenue is", "object": "$500M"},
    ]
    raw_output = json.dumps(triples_data)

    # Patch the chain to return our canned output
    from strategy_agent.ingestion import kg_extractor

    original_prompt = kg_extractor.extraction_prompt

    class FakeChain:
        def invoke(self, inputs):
            return raw_output

    monkeypatch.setattr(
        kg_extractor,
        "extraction_prompt",
        type("FakePrompt", (), {"__or__": lambda self, other: type("Mid", (), {"__or__": lambda s, o: FakeChain()})()})(),
    )

    result = extract_triples("Some text about Acme Corp competing.", None)

    # Restore
    kg_extractor.extraction_prompt = original_prompt

    assert len(result) == 2
    assert result[0] == ("Acme Corp", "competes with", "WidgetCo")
    assert result[1] == ("Acme Corp", "revenue is", "$500M")


def test_extract_triples_with_code_fences(monkeypatch):
    """Should handle JSON wrapped in markdown code fences."""
    triples_data = [
        {"subject": "TechCo", "predicate": "acquired", "object": "StartupX"},
    ]
    raw_output = f"```json\n{json.dumps(triples_data)}\n```"

    from strategy_agent.ingestion import kg_extractor

    original_prompt = kg_extractor.extraction_prompt

    class FakeChain:
        def invoke(self, inputs):
            return raw_output

    monkeypatch.setattr(
        kg_extractor,
        "extraction_prompt",
        type("FakePrompt", (), {"__or__": lambda self, other: type("Mid", (), {"__or__": lambda s, o: FakeChain()})()})(),
    )

    result = extract_triples("TechCo acquired StartupX.", None)
    kg_extractor.extraction_prompt = original_prompt

    assert len(result) == 1
    assert result[0] == ("TechCo", "acquired", "StartupX")


def test_extract_triples_invalid_json_returns_empty(monkeypatch):
    """Should return empty list on unparseable output."""
    from strategy_agent.ingestion import kg_extractor

    original_prompt = kg_extractor.extraction_prompt

    class FakeChain:
        def invoke(self, inputs):
            return "This is not JSON at all."

    monkeypatch.setattr(
        kg_extractor,
        "extraction_prompt",
        type("FakePrompt", (), {"__or__": lambda self, other: type("Mid", (), {"__or__": lambda s, o: FakeChain()})()})(),
    )

    result = extract_triples("Some text.", None)
    kg_extractor.extraction_prompt = original_prompt

    assert result == []


def test_extract_triples_filters_none_values(monkeypatch):
    """Should skip triples where subject, predicate, or object is null."""
    triples_data = [
        {"subject": "Acme Corp", "predicate": "competes with", "object": "WidgetCo"},
        {"subject": None, "predicate": "acquired", "object": "StartupX"},
        {"subject": "TechCo", "predicate": None, "object": "DataInc"},
        {"subject": "TechCo", "predicate": "partners with", "object": None},
    ]
    raw_output = json.dumps(triples_data)

    from strategy_agent.ingestion import kg_extractor

    original_prompt = kg_extractor.extraction_prompt

    class FakeChain:
        def invoke(self, inputs):
            return raw_output

    monkeypatch.setattr(
        kg_extractor,
        "extraction_prompt",
        type(
            "FakePrompt", (),
            {"__or__": lambda self, other: type(
                "Mid", (), {"__or__": lambda s, o: FakeChain()}
            )()},
        )(),
    )

    result = extract_triples("Text with null triples.", None)
    kg_extractor.extraction_prompt = original_prompt

    assert len(result) == 1
    assert result[0] == ("Acme Corp", "competes with", "WidgetCo")


def test_extract_triples_filters_empty_strings(monkeypatch):
    """Should skip triples where any value is an empty string."""
    triples_data = [
        {"subject": "Acme Corp", "predicate": "competes with", "object": "WidgetCo"},
        {"subject": "", "predicate": "acquired", "object": "StartupX"},
        {"subject": "TechCo", "predicate": "partners with", "object": "  "},
    ]
    raw_output = json.dumps(triples_data)

    from strategy_agent.ingestion import kg_extractor

    original_prompt = kg_extractor.extraction_prompt

    class FakeChain:
        def invoke(self, inputs):
            return raw_output

    monkeypatch.setattr(
        kg_extractor,
        "extraction_prompt",
        type(
            "FakePrompt", (),
            {"__or__": lambda self, other: type(
                "Mid", (), {"__or__": lambda s, o: FakeChain()}
            )()},
        )(),
    )

    result = extract_triples("Text with empty triples.", None)
    kg_extractor.extraction_prompt = original_prompt

    assert len(result) == 1
    assert result[0] == ("Acme Corp", "competes with", "WidgetCo")
