"""Qdrant vector store with local (embedded) and server modes.

Local mode is used by default so the whole platform runs on a laptop without a
Docker daemon. Point ``SOURCEIQ_QDRANT_URL`` at a Qdrant server (or the compose
service) for a shared deployment - the application code does not change.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

from sourceiq.config import Settings, get_settings
from sourceiq.logging_config import get_logger
from sourceiq.vectorstore.base import ChunkRecord, RetrievedChunk, VectorStore

logger = get_logger(__name__)

_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        *,
        dimensions: int,
        settings: Settings | None = None,
        client: QdrantClient | None = None,
        collection: str | None = None,
        path: str | Path | None = None,
    ) -> None:
        cfg = settings or get_settings()
        self.settings = cfg
        self.dimensions = int(dimensions)
        self.collection = collection or cfg.qdrant_collection
        self._owns_client = client is None

        if client is not None:
            self.client = client
        elif cfg.qdrant_url:
            logger.info("Connecting to Qdrant server at %s", cfg.qdrant_url)
            self.client = QdrantClient(
                url=cfg.qdrant_url, api_key=cfg.qdrant_api_key, timeout=30
            )
        else:
            location = ":memory:" if path == ":memory:" or cfg.qdrant_path == Path(":memory:") else None
            if location:
                logger.info("Using in-memory Qdrant instance")
                self.client = QdrantClient(location=location)
            else:
                qdrant_path = Path(path) if path is not None else cfg.qdrant_path
                qdrant_path.mkdir(parents=True, exist_ok=True)
                logger.info("Using local Qdrant storage at %s", qdrant_path)
                self.client = QdrantClient(path=str(qdrant_path))

        self.ensure_collection()

    # ------------------------------------------------------------------ #
    # Schema
    # ------------------------------------------------------------------ #
    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            info = self.client.get_collection(self.collection)
            existing = self._vector_size(info)
            if existing is not None and existing != self.dimensions:
                raise ValueError(
                    f"Collection '{self.collection}' stores {existing}-dim vectors but the "
                    f"embedder produces {self.dimensions}. Delete the collection or change "
                    "SOURCEIQ_EMBEDDING_MODEL."
                )
            return

        logger.info("Creating Qdrant collection '%s' (%d dims)", self.collection, self.dimensions)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.dimensions, distance=Distance.COSINE),
        )

    @staticmethod
    def _vector_size(info: Any) -> int | None:
        try:
            vectors = info.config.params.vectors
            if isinstance(vectors, dict):  # named vectors
                return int(next(iter(vectors.values())).size)
            return int(vectors.size)
        except Exception:  # pragma: no cover - defensive across client versions
            return None

    @staticmethod
    def _point_id(document_id: str, chunk_index: int) -> str:
        return str(uuid.uuid5(_NAMESPACE, f"{document_id}:{chunk_index}"))

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    def upsert_chunks(self, chunks: Sequence[ChunkRecord], vectors: np.ndarray) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("Every chunk needs exactly one embedding vector.")
        if not chunks:
            return 0

        points = [
            PointStruct(
                id=self._point_id(chunk.document_id, chunk.chunk_index),
                vector=np.asarray(vector, dtype=float).tolist(),
                payload=chunk.payload(),
            )
            for chunk, vector in zip(chunks, vectors, strict=False)
        ]
        self.client.upsert(collection_name=self.collection, points=points, wait=True)
        return len(points)

    def delete_document(self, document_id: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            ),
            wait=True,
        )

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    def search(
        self,
        vector: np.ndarray,
        *,
        top_k: int = 5,
        document_ids: Sequence[str] | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        query_filter = None
        if document_ids:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id", match=MatchAny(any=[str(value) for value in document_ids])
                    )
                ]
            )

        response = self.client.query_points(
            collection_name=self.collection,
            query=np.asarray(vector, dtype=float).tolist(),
            limit=int(top_k),
            query_filter=query_filter,
            score_threshold=score_threshold,
            with_payload=True,
        )

        results: list[RetrievedChunk] = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                RetrievedChunk(
                    document_id=str(payload.get("document_id", "")),
                    filename=str(payload.get("filename", "unknown")),
                    chunk_index=int(payload.get("chunk_index", 0)),
                    page=int(payload.get("page", 1)),
                    text=str(payload.get("text", "")),
                    score=float(point.score),
                )
            )
        return results

    def count(self) -> int:
        try:
            return int(self.client.count(self.collection, exact=True).count)
        except Exception:  # pragma: no cover - collection removed underneath us
            return 0

    def health(self) -> bool:
        try:
            self.client.get_collection(self.collection)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Housekeeping
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Remove every point from the collection (tests, admin tooling).

        Deleting and recreating the collection is not enough in Qdrant's local
        mode - the embedded segment cache keeps the old points until the client
        restarts - so we explicitly delete all points with a match-all filter.
        """
        self.client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(filter=Filter(must=[])),
            wait=True,
        )

    def close(self) -> None:
        if self._owns_client:
            try:
                self.client.close()
            except Exception:  # pragma: no cover - best effort on shutdown
                logger.debug("Qdrant client close failed", exc_info=True)


__all__ = ["QdrantVectorStore"]
