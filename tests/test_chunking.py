"""Chunking: page preservation, size limits and overlap."""

from __future__ import annotations

import pytest

from sourceiq.documents.chunking import chunk_pages
from sourceiq.documents.extract import PageText


def _page(number: int, sentences: int = 40) -> PageText:
    text = " ".join(
        f"Sentence {index} explains policy item number {index} in detail."
        for index in range(sentences)
    )
    return PageText(page_number=number, text=text)


def test_short_page_produces_single_chunk():
    chunks = chunk_pages(
        [PageText(page_number=1, text="A short policy paragraph.")],
        document_id="doc-1",
        filename="policy.pdf",
        chunk_size=500,
        overlap=50,
    )

    assert len(chunks) == 1
    assert chunks[0].page == 1
    assert chunks[0].chunk_index == 0


def test_long_page_is_split_and_pages_are_preserved():
    pages = [_page(1), _page(2)]
    chunks = chunk_pages(
        pages, document_id="doc-1", filename="policy.pdf", chunk_size=400, overlap=80
    )

    assert len(chunks) > 2
    assert {chunk.page for chunk in chunks} == {1, 2}
    assert all(len(chunk.text) <= 400 for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_overlap_increases_total_characters():
    page = _page(1)
    with_overlap = chunk_pages(
        [page], document_id="d", filename="f.pdf", chunk_size=400, overlap=100
    )
    without_overlap = chunk_pages(
        [page], document_id="d", filename="f.pdf", chunk_size=400, overlap=0
    )

    assert sum(len(chunk.text) for chunk in with_overlap) > sum(
        len(chunk.text) for chunk in without_overlap
    )


def test_chunks_do_not_cut_words():
    chunks = chunk_pages(
        [_page(1)], document_id="d", filename="f.pdf", chunk_size=300, overlap=50
    )
    for chunk in chunks:
        assert chunk.text == chunk.text.strip()
        assert not chunk.text.endswith("-")


def test_invalid_configuration_raises():
    page = PageText(page_number=1, text="text")

    with pytest.raises(ValueError):
        chunk_pages([page], document_id="d", filename="f.pdf", chunk_size=100, overlap=100)
    with pytest.raises(ValueError):
        chunk_pages([page], document_id="d", filename="f.pdf", chunk_size=0, overlap=0)
    with pytest.raises(ValueError):
        chunk_pages([page], document_id="d", filename="f.pdf", chunk_size=100, overlap=-1)


def test_empty_pages_produce_no_chunks():
    assert chunk_pages([], document_id="d", filename="f.pdf") == []
