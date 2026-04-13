"""Tests for the PPTX document loader."""

from unittest.mock import MagicMock, patch

from strategy_agent.ingestion.loader import load_file
from strategy_agent.ingestion.pptx_loader import PptxLoader


def _make_shape(text: str, has_table: bool = False) -> MagicMock:
    """Create a mock shape with a text frame."""
    shape = MagicMock()
    shape.has_text_frame = bool(text)
    shape.has_table = has_table
    shape.shape_type = None

    if text:
        paragraph = MagicMock()
        paragraph.text = text
        shape.text_frame.paragraphs = [paragraph]
    else:
        shape.text_frame.paragraphs = []

    # Group shape iteration should raise AttributeError (not a group)
    del shape.shapes

    return shape


def _make_table_shape(rows: list[list[str]]) -> MagicMock:
    """Create a mock shape with a table."""
    shape = MagicMock()
    shape.has_text_frame = False
    shape.text_frame.paragraphs = []
    shape.has_table = True
    shape.shape_type = None

    mock_rows = []
    for row_texts in rows:
        row = MagicMock()
        cells = []
        for cell_text in row_texts:
            cell = MagicMock()
            cell.text = cell_text
            cells.append(cell)
        row.cells = cells
        mock_rows.append(row)
    shape.table.rows = mock_rows

    del shape.shapes
    return shape


def _make_slide(shapes: list[MagicMock]) -> MagicMock:
    slide = MagicMock()
    slide.shapes = shapes
    return slide


@patch("strategy_agent.ingestion.pptx_loader.Presentation")
def test_load_extracts_text_from_slides(mock_prs_cls):
    """Each slide with text should become a Document."""
    mock_prs = MagicMock()
    mock_prs.slides = [
        _make_slide([_make_shape("Slide 1 Title"), _make_shape("Slide 1 Body")]),
        _make_slide([_make_shape("Slide 2 Content")]),
    ]
    mock_prs_cls.return_value = mock_prs

    loader = PptxLoader("/fake/path/deck.pptx")
    docs = loader.load()

    assert len(docs) == 2
    assert "Slide 1 Title" in docs[0].page_content
    assert "Slide 1 Body" in docs[0].page_content
    assert "Slide 2 Content" in docs[1].page_content


@patch("strategy_agent.ingestion.pptx_loader.Presentation")
def test_load_sets_metadata(mock_prs_cls):
    """Each Document should have slide_number and source_file metadata."""
    mock_prs = MagicMock()
    mock_prs.slides = [
        _make_slide([_make_shape("Content")]),
    ]
    mock_prs_cls.return_value = mock_prs

    loader = PptxLoader("/fake/path/deck.pptx")
    docs = loader.load()

    assert docs[0].metadata["slide_number"] == 1
    assert docs[0].metadata["source_file"] == "deck.pptx"
    assert docs[0].metadata["source"] == "/fake/path/deck.pptx"


@patch("strategy_agent.ingestion.pptx_loader.Presentation")
def test_load_skips_empty_slides(mock_prs_cls):
    """Slides with no extractable text should be skipped."""
    mock_prs = MagicMock()
    mock_prs.slides = [
        _make_slide([_make_shape("")]),
        _make_slide([_make_shape("Real content")]),
    ]
    mock_prs_cls.return_value = mock_prs

    loader = PptxLoader("/fake/path/deck.pptx")
    docs = loader.load()

    assert len(docs) == 1
    assert "Real content" in docs[0].page_content


@patch("strategy_agent.ingestion.pptx_loader.Presentation")
def test_load_extracts_table_text(mock_prs_cls):
    """Tables in slides should be extracted as pipe-delimited rows."""
    table_shape = _make_table_shape([
        ["Header A", "Header B"],
        ["Value 1", "Value 2"],
    ])
    mock_prs = MagicMock()
    mock_prs.slides = [_make_slide([table_shape])]
    mock_prs_cls.return_value = mock_prs

    loader = PptxLoader("/fake/path/deck.pptx")
    docs = loader.load()

    assert len(docs) == 1
    assert "Header A | Header B" in docs[0].page_content
    assert "Value 1 | Value 2" in docs[0].page_content


def test_pptx_registered_in_loader():
    """The .pptx extension should be recognized by the loader registry."""
    from strategy_agent.ingestion.loader import _LOADER_REGISTRY

    assert ".pptx" in _LOADER_REGISTRY


@patch("strategy_agent.ingestion.pptx_loader.Presentation")
def test_load_file_pptx_integration(mock_prs_cls, tmp_path):
    """load_file should dispatch .pptx files to PptxLoader."""
    mock_prs = MagicMock()
    mock_prs.slides = [
        _make_slide([_make_shape("Integration test slide")]),
    ]
    mock_prs_cls.return_value = mock_prs

    pptx_file = tmp_path / "test.pptx"
    pptx_file.touch()  # Create an empty file; PptxLoader is mocked

    docs = load_file(pptx_file)

    assert len(docs) == 1
    assert "Integration test slide" in docs[0].page_content
    assert docs[0].metadata["file_type"] == ".pptx"
    assert docs[0].metadata["source_file"] == "test.pptx"
