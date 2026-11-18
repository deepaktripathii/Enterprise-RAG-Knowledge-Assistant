"""Deterministic hashing embedder (bag of words + bigrams).

This is not a replacement for a trained sentence encoder, but it is honest
about what it is: a lexical vectoriser that lets the whole pipeline - ingestion,
Qdrant storage, retrieval, evaluation - run in CI and on machines without torch.
Vectors are L2-normalised so cosine similarity behaves exactly like the
sentence-transformer backend.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from rag_assistant.embeddings.base import Embedder

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


class HashingEmbedder(Embedder):
    def __init__(self, dimensions: int = 768) -> None:
        if dimensions < 16:
            raise ValueError("Hashing embedder needs at least 16 dimensions.")
        self.dimensions = int(dimensions)
        self.name = f"hashing-{dimensions}"

    @staticmethod
    def _tokens(text: str) -> list[str]:
        tokens = [token.lower() for token in _TOKEN_RE.findall(text)]
        bigrams = [f"{first}__{second}" for first, second in zip(tokens, tokens[1:], strict=False)]
        return tokens + bigrams

    def _weights(self, text: str) -> dict[int, float]:
        vector: dict[int, float] = {}
        for token in self._tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            index = value % self.dimensions
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vector[index] = vector.get(index, 0.0) + sign
        return vector

    def _encode_one(self, text: str) -> np.ndarray:
        output = np.zeros(self.dimensions, dtype=np.float32)
        for index, weight in self._weights(text).items():
            output[index] = weight
        norm = float(np.linalg.norm(output))
        if norm > 0:
            output /= norm
        return output

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return np.vstack([self._encode_one(text) for text in texts]).astype(np.float32)


__all__ = ["HashingEmbedder"]
