"""Citation parsing and source mapping (Feature 6).

The LLM only emits numeric markers (``[1]``). This module validates those markers
against the chunks that were actually retrieved, drops anything fabricated and
turns the surviving markers into user-facing sources with document, page and a
text snippet. Invalid markers are removed from the answer text, so a model cannot
cite a document that was never retrieved.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from sourceiq.vectorstore.base import RetrievedChunk

_MARKER = re.compile(r"\[(\d{1,2})\]")


def citation_from_chunk(index: int, chunk: RetrievedChunk) -> dict[str, object]:
    return {
        "index": index,
        "document_id": chunk.document_id,
        "filename": chunk.filename,
        "page": chunk.page,
        "chunk_index": chunk.chunk_index,
        "score": round(float(chunk.score), 4),
        "snippet": chunk.text.strip()[:320],
    }


def extract_citations(
    answer: str,
    chunks: Sequence[RetrievedChunk],
) -> tuple[str, list[dict[str, object]], list[str]]:
    """Return ``(sanitised_answer, citations, warnings)``."""
    warnings: list[str] = []
    if not chunks:
        return answer, [], warnings

    referenced: list[int] = []
    invalid_seen = False
    for match in _MARKER.finditer(answer):
        index = int(match.group(1))
        if 1 <= index <= len(chunks):
            if index not in referenced:
                referenced.append(index)
        else:
            invalid_seen = True

    if invalid_seen:
        warnings.append("invalid_citation_markers_removed")
        answer = _MARKER.sub(
            lambda match: match.group(0)
            if 1 <= int(match.group(1)) <= len(chunks)
            else "",
            answer,
        )

    if not referenced:
        warnings.append("answer_missing_citations")
        citations = [citation_from_chunk(1, chunks[0])]
        return answer, citations, warnings

    citations = [citation_from_chunk(index, chunks[index - 1]) for index in referenced]
    return answer, citations, warnings


__all__ = ["citation_from_chunk", "extract_citations"]
