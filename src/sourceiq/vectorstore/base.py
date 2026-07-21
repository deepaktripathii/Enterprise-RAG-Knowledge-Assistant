"""Vector store abstractions and payload types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class ChunkRecord:
    """A chunk ready to be embedded and stored."""

    document_id: str
    filename: str
    chunk_index: int
    page: int
    text: str

    def payload(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("text")
        data["text"] = self.text
        return data


@dataclass
class RetrievedChunk:
    document_id: str
    filename: str
    chunk_index: int
    page: int
    text: str
    score: float

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        data = {
            "document_id": self.document_id,
            "filename": self.filename,
            "chunk_index": self.chunk_index,
            "page": self.page,
            "score": round(float(self.score), 4),
        }
        if include_text:
            data["text"] = self.text
        return data


class VectorStore(ABC):
    """Minimal contract the RAG pipeline needs from a vector database."""

    @abstractmethod
    def upsert_chunks(self, chunks: Sequence[ChunkRecord], vectors: np.ndarray) -> int: ...

    @abstractmethod
    def search(
        self,
        vector: np.ndarray,
        *,
        top_k: int = 5,
        document_ids: Sequence[str] | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]: ...

    @abstractmethod
    def delete_document(self, document_id: str) -> None: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def health(self) -> bool: ...

    def close(self) -> None:
        """Release the underlying client (no-op for in-process stores)."""
        return None


__all__ = ["ChunkRecord", "RetrievedChunk", "VectorStore"]
