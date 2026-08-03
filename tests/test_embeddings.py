"""Embeddings: determinism, normalisation and semantic ordering."""

from __future__ import annotations

import numpy as np

from sourceiq.embeddings.hashing import HashingEmbedder


def test_dimensions_and_shape():
    embedder = HashingEmbedder(128)

    vectors = embedder.encode(["first chunk", "second chunk"])

    assert vectors.shape == (2, 128)
    assert vectors.dtype == np.float32


def test_vectors_are_normalised():
    embedder = HashingEmbedder(64)

    vector = embedder.encode_one("annual leave policy for employees")

    assert np.linalg.norm(vector) == np.float32(1.0) or np.isclose(
        np.linalg.norm(vector), 1.0, atol=1e-6
    )


def test_encoding_is_deterministic():
    embedder = HashingEmbedder(64)

    first = embedder.encode_one("work from home policy")
    second = embedder.encode_one("work from home policy")

    np.testing.assert_array_equal(first, second)


def test_related_text_scores_higher_than_unrelated():
    embedder = HashingEmbedder(512)
    query = embedder.encode_one("How many annual leave days do employees get?")
    related = embedder.encode_one("Employees receive 24 days of paid annual leave per year.")
    unrelated = embedder.encode_one("Passwords must be rotated every 90 days.")

    assert float(np.dot(query, related)) > float(np.dot(query, unrelated))


def test_empty_input_returns_empty_matrix():
    embedder = HashingEmbedder(32)

    assert embedder.encode([]).shape == (0, 32)


def test_minimum_dimensions_enforced():
    import pytest

    with pytest.raises(ValueError):
        HashingEmbedder(8)
