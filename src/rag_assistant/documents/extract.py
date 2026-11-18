"""PDF text extraction with page-level provenance (Feature 2).

PyMuPDF is used because it is fast, has no external binary dependency and
returns text per page - which is exactly what a citation-aware RAG system
needs. Scanned PDFs without a text layer are rejected with an actionable error
instead of silently producing empty chunks.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf

from rag_assistant.logging_config import get_logger

logger = get_logger(__name__)


class PdfExtractionError(RuntimeError):
    """Raised when a PDF cannot be read or contains no extractable text."""


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str


@dataclass
class ExtractionResult:
    pages: list[PageText]
    page_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def character_count(self) -> int:
        return sum(len(page.text) for page in self.pages)


_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
_PAGE_NUMBER_LINE = re.compile(r"^\s*(?:page\s*)?\d+\s*(?:of\s*\d+)?\s*$", re.IGNORECASE)
_MULTIPLE_NEWLINES = re.compile(r"\n{3,}")
_INLINE_SPACES = re.compile(r"[ \t]+")


def clean_text(raw: str) -> str:
    """Normalise extracted text while keeping paragraph structure.

    * de-hyphenates words broken across lines ("manda-\\ntory" -> "mandatory")
    * removes control characters and zero-width spaces
    * drops standalone page-number lines ("Page 4 of 12")
    * collapses repeated spaces/newlines
    """
    text = unicodedata.normalize("NFKC", raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)

    cleaned_lines: list[str] = []
    for line in text.split("\n"):
        line = _INLINE_SPACES.sub(" ", line).strip()
        if not line:
            cleaned_lines.append("")
            continue
        if _PAGE_NUMBER_LINE.match(line):
            continue
        if any(unicodedata.category(char) == "Cc" for char in line):
            line = "".join(char for char in line if unicodedata.category(char) != "Cc")
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = _MULTIPLE_NEWLINES.sub("\n\n", text)
    return text.strip()


def extract_pdf(path: str | Path) -> ExtractionResult:
    """Extract page-wise text from a PDF file."""
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise PdfExtractionError(f"PDF not found: {pdf_path}")

    try:
        document = pymupdf.open(pdf_path)
    except Exception as exc:  # noqa: BLE001 - pymupdf raises several types
        raise PdfExtractionError(f"Could not open PDF '{pdf_path.name}': {exc}") from exc

    try:
        page_count = int(document.page_count)
        pages: list[PageText] = []
        for index in range(page_count):
            page = document.load_page(index)
            cleaned = clean_text(page.get_text("text"))
            if cleaned:
                pages.append(PageText(page_number=index + 1, text=cleaned))

        raw_metadata = document.metadata or {}
        metadata = {
            str(key).lower(): value
            for key, value in raw_metadata.items()
            if value and str(value).strip()
        }
    finally:
        document.close()

    if not pages:
        raise PdfExtractionError(
            f"'{pdf_path.name}' has no extractable text. It is probably a scanned PDF; "
            "run OCR before uploading."
        )

    logger.info(
        "Extracted %s: %d pages, %d characters",
        pdf_path.name,
        len(pages),
        sum(len(page.text) for page in pages),
    )
    return ExtractionResult(pages=pages, page_count=page_count, metadata=metadata)


__all__ = ["ExtractionResult", "PageText", "PdfExtractionError", "clean_text", "extract_pdf"]
