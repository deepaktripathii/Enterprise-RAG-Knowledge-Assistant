"""Qdrant vector store: upsert, search, filters and deletion."""

from __future__ import annotations

import numpy as np
import pytest

from rag_assistant.vectorstore.base import ChunkRecord
from rag_assistant.vectorstore.qdrant_store import QdrantVectorStore

DIMENSIONS = 16


def _chunk(document_id: str, index: int, page: int, text: str) -> ChunkRecord:
    return ChunkRecord(
        document_id=document_id,
        filename=f"{document_id}.pdf",
        chunk_index=index,
        page=page,
        text=text,
    )


def _vector(seed: int, dimensions: int = DIMENSIONS) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dimensions)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


@pytest.fixture()
def store(tmp_settings):
    vector_store = QdrantVectorStore(dimensions=DIMENSIONS, settings=tmp_settings)
    try:
        yield vector_store
    finally:
        vector_store.close()


def test_upsert_and_count(store):
    chunks = [_chunk("doc-a", 0, 1, "alpha"), _chunk("doc-a", 1, 2, "beta")]

    stored = store.upsert_chunks(chunks, np.vstack([_vector(1), _vector(2)]))

    assert stored == 2
    assert store.count() == 2
    assert store.health() is True


def test_upsert_is_idempotent_per_chunk(store):
    chunk = _chunk("doc-a", 0, 1, "alpha")

    store.upsert_chunks([chunk], _vector(1).reshape(1, -1))
    store.upsert_chunks([chunk], _vector(2).reshape(1, -1))

    assert store.count() == 1
    results = store.search(_vector(2), top_k=1)
    assert results[0].text == "alpha"


def test_search_returns_best_match(store):
    chunks = [
        _chunk("doc-a", 0, 1, "annual leave policy"),
        _chunk("doc-b", 0, 1, "password rotation policy"),
    ]
    store.upsert_chunks(chunks, np.vstack([_vector(1), _vector(2)]))

    results = store.search(_vector(1), top_k=1)

    assert len(results) == 1
    assert results[0].document_id == "doc-a"
    assert results[0].score == pytest.approx(1.0, abs=1e-4)


def test_search_filters_by_document(store):
    chunks = [
        _chunk("doc-a", 0, 1, "alpha"),
        _chunk("doc-b", 0, 1, "alpha"),
    ]
    store.upsert_chunks(chunks, np.vstack([_vector(1), _vector(1)]))

    results = store.search(_vector(1), top_k=5, document_ids=["doc-b"])

    assert len(results) == 1
    assert results[0].document_id == "doc-b"


def test_score_threshold(store):
    store.upsert_chunks([_chunk("doc-a", 0, 1, "alpha")], _vector(1).reshape(1, -1))

    assert store.search(_vector(2), top_k=3, score_threshold=0.99) == []


def test_delete_document_removes_only_its_points(store):
    chunks = [
        _chunk("doc-a", 0, 1, "alpha"),
        _chunk("doc-b", 0, 1, "beta"),
    ]
    store.upsert_chunks(chunks, np.vstack([_vector(1), _vector(2)]))

    store.delete_document("doc-a")

    assert store.count() == 1
    assert store.search(_vector(1), top_k=5)[0].document_id == "doc-b"


def test_reset_clears_collection(store):
    store.upsert_chunks([_chunk("doc-a", 0, 1, "alpha")], _vector(1).reshape(1, -1))

    store.reset()

    assert store.count() == 0
    assert store.health() is True


def test_dimension_mismatch_is_rejected(tmp_settings):
    first = QdrantVectorStore(dimensions=DIMENSIONS, settings=tmp_settings)
    first.upsert_chunks([_chunk("doc-a", 0, 1, "alpha")], _vector(1).reshape(1, -1))
    first.close()

    with pytest.raises(ValueError, match="dim vectors"):
        QdrantVectorStore(dimensions=8, settings=tmp_settings)
