"""Embedder protocol shared by every backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class Embedder(ABC):
    """Turns text into L2-normalised float vectors."""

    name: str
    dimensions: int

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(n, dimensions)`` float32 array."""

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    @property
    def is_semantic(self) -> bool:
        return False

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r} dim={self.dimensions}>"


__all__ = ["Embedder"]
