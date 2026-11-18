"""Page-preserving text chunking (Feature 2).

Design choices
---------------
* **Characters, not tokens.** Chunk size is expressed in characters because it
  is model independent: a token budget that fits ``all-MiniLM`` may overflow a
  different embedder or a larger LLM context.
* **Page scoped.** A chunk never spans two pages, so every retrieved passage
  maps to one page number and citations stay exact.
* **Sentence aware.** Windows are cut at the last sentence boundary when
  possible, and the next window slides back into a word boundary. The overlap
  keeps pronouns and multi-sentence definitions retrievable.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag_assistant.documents.extract import PageText
from rag_assistant.vectorstore.base import ChunkRecord

_SENTENCE_ENDS = (". ", ".\n", "! ", "?\n", "! ", "? ")


def _slice_page(text: str, chunk_size: int, overlap: int) -> list[tuple[str, int, int]]:
    """Split one page into ``(text, start, end)`` windows."""
    if len(text) <= chunk_size:
        return [(text, 0, len(text))]

    windows: list[tuple[str, int, int]] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            window = text[start:end]
            cut = max(window.rfind(marker) for marker in _SENTENCE_ENDS)
            if cut >= int(chunk_size * 0.55):
                end = start + cut + 1  # keep the sentence terminator

        piece = text[start:end].strip()
        if piece:
            windows.append((piece, start, end))
        if end >= length:
            break

        next_start = max(end - overlap, start + 1)
        space = text.find(" ", next_start)
        if space != -1 and space < end:
            next_start = space + 1
        if next_start <= start:  # safety net, should not happen
            next_start = start + 1
        start = next_start

    return windows


def chunk_pages(
    pages: Sequence[PageText],
    *,
    document_id: str,
    filename: str,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[ChunkRecord]:
    """Convert extracted pages into overlapping, page-tagged chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be greater than or equal to 0 and smaller than chunk_size.")

    chunks: list[ChunkRecord] = []
    index = 0
    for page in pages:
        for text, _start, _end in _slice_page(page.text, chunk_size, overlap):
            chunks.append(
                ChunkRecord(
                    document_id=document_id,
                    filename=filename,
                    chunk_index=index,
                    page=page.page_number,
                    text=text,
                )
            )
            index += 1
    return chunks


__all__ = ["chunk_pages"]
