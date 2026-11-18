"""Query embedding + Qdrant similarity search (Features 3 and 4)."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from rag_assistant.config import Settings, get_settings
from rag_assistant.embeddings.base import Embedder
from rag_assistant.logging_config import get_logger
from rag_assistant.vectorstore.base import RetrievedChunk, VectorStore

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    latency_ms: float = 0.0
    top_score: float = 0.0

    @property
    def is_empty(self) -> bool:
        return not self.chunks


class Retriever:
    def __init__(
        self,
        *,
        embedder: Embedder,
        vector_store: VectorStore,
        settings: Settings | None = None,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.settings = settings or get_settings()

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        vector = self.embedder.encode_one(query)
        chunks = self.vector_store.search(
            vector,
            top_k=top_k or self.settings.retrieval_top_k,
            document_ids=document_ids,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        top_score = chunks[0].score if chunks else 0.0
        logger.debug(
            "Retrieved %d chunks in %.1f ms (top score %.3f)",
            len(chunks),
            latency_ms,
            top_score,
        )
        return RetrievalResult(
            query=query,
            chunks=chunks,
            latency_ms=latency_ms,
            top_score=top_score,
        )


__all__ = ["RetrievalResult", "Retriever"]
