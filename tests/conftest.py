"""Shared fixtures.

Tests run fully offline with the deterministic hashing embedder and the
extractive answerer, so no model download, API key or Qdrant server is needed.
"""

from __future__ import annotations

import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import pymupdf
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sourceiq.config import get_settings  # noqa: E402
from sourceiq.services import build_services  # noqa: E402


def write_pdf(path: Path, pages: list[str]) -> Path:
    """Write a small PDF with one explicit page per string."""
    document = pymupdf.open()
    for text in pages:
        page = document.new_page(width=595, height=842)
        y = 72
        for line in textwrap.wrap(text, width=80):
            page.insert_text((72, y), line, fontsize=11, fontname="helv")
            y += 16
    document.set_metadata({"title": path.stem})
    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(path)
    document.close()
    return path


@pytest.fixture()
def tmp_settings(tmp_path):
    base = get_settings().with_storage(tmp_path)
    settings = replace(
        base,
        embedding_backend="hashing",
        llm_provider="extractive",
        chunk_size=500,
        chunk_overlap=80,
        retrieval_top_k=4,
        min_relevance_score=0.02,
        max_upload_mb=5,
        max_history_messages=6,
    )
    settings.ensure_directories()
    return settings


@pytest.fixture()
def pdf_library(tmp_settings):
    library = {
        "remote": write_pdf(
            tmp_settings.sample_docs_dir / "Remote_Work_Policy.pdf",
            [
                "Remote Work Policy. Employees may work remotely for two days per week. "
                "Tuesday and Thursday are mandatory office days.",
                "Working from outside India requires prior approval from HR and the "
                "security team.",
            ],
        ),
        "security": write_pdf(
            tmp_settings.sample_docs_dir / "IT_Security_Policy.pdf",
            [
                "IT Security Policy. Passwords must be at least 14 characters long. "
                "Multi-factor authentication is mandatory.",
                "Report suspected phishing or malware within one hour to "
                "security@example.com.",
            ],
        ),
    }
    return library


@pytest.fixture()
def services(tmp_settings, pdf_library):
    container = build_services(tmp_settings)
    try:
        yield container
    finally:
        container.close()


@pytest.fixture()
def ingested(services):
    """Index both sample documents and return their ids keyed by short name."""
    ids = {}
    for path in services.settings.sample_docs_dir.glob("*.pdf"):
        document = services.document_service.ingest_file(path)
        assert document["status"] == "completed", document.get("error_message")
        ids[path.stem] = document["id"]
    return ids
