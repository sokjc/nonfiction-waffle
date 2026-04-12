"""Tests for CLI input validation and edge cases."""

from strategy_agent.cli import VALID_DOCUMENT_TYPES


def test_valid_document_types_complete():
    """All five document types should be in the validation set."""
    expected = {
        "strategy_memo",
        "white_paper",
        "board_presentation",
        "competitive_analysis",
        "market_assessment",
    }
    assert VALID_DOCUMENT_TYPES == expected


def test_invalid_type_not_in_set():
    """An invalid type should not pass validation."""
    assert "quarterly_report" not in VALID_DOCUMENT_TYPES
    assert "memo" not in VALID_DOCUMENT_TYPES
    assert "" not in VALID_DOCUMENT_TYPES
