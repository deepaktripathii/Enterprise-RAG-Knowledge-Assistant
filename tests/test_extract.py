"""PDF extraction: cleaning and page provenance."""

from __future__ import annotations

import pymupdf
import pytest

from sourceiq.documents.extract import PdfExtractionError, clean_text, extract_pdf


def test_clean_text_joins_hyphenated_lines():
    assert clean_text("This is manda-\ntory.") == "This is mandatory."


def test_clean_text_drops_page_number_lines():
    cleaned = clean_text("Policy text.\nPage 3 of 10\nMore text.")
    assert "Page 3" not in cleaned
    assert cleaned == "Policy text.\nMore text."


def test_clean_text_collapses_whitespace():
    assert clean_text("too    many\tspaces") == "too many spaces"


def test_extract_pdf_preserves_page_numbers(pdf_library):
    result = extract_pdf(pdf_library["remote"])

    assert result.page_count == 2
    assert [page.page_number for page in result.pages] == [1, 2]
    assert "two days per week" in result.pages[0].text
    assert "outside India" in result.pages[1].text
    assert result.metadata.get("title") == "Remote_Work_Policy"


def test_extract_pdf_missing_file(tmp_path):
    with pytest.raises(PdfExtractionError):
        extract_pdf(tmp_path / "missing.pdf")


def test_extract_pdf_without_text_raises(tmp_path):
    blank = tmp_path / "blank.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(blank)
    document.close()

    with pytest.raises(PdfExtractionError, match="no extractable text"):
        extract_pdf(blank)
