"""Sentence Transformer embeddings (production backend)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from rag_assistant.embeddings.base import Embedder


class SentenceTransformerEmbedder(Embedder):
    """Wraps a pre-trained Sentence Transformer model.

    The import is local so the rest of the application keeps working when torch
    is not installed; :func:`rag_assistant.embeddings.build_embedder` catches the
    failure and falls back to the hashing backend.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        dimension_getter = getattr(self._model, "get_embedding_dimension", None) or (
            self._model.get_sentence_embedding_dimension
        )
        self.dimensions = int(dimension_getter())
        self.name = f"sentence-transformers:{model_name}"

    @property
    def is_semantic(self) -> bool:
        return True

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)


__all__ = ["SentenceTransformerEmbedder"]
