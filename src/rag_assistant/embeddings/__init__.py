"""Embedder abstraction and implementations.

Two backends ship with the project:

* ``sentence-transformers`` - production semantic embeddings (MiniLM by default).
* ``hashing`` - deterministic, dependency-light lexical embeddings used for CI,
  offline demos and as an automatic fallback when torch is unavailable.

Both expose the same ``encode`` contract, so the ingestion and retrieval code is
backend agnostic.
"""

from rag_assistant.embeddings.base import Embedder
from rag_assistant.embeddings.factory import build_embedder
from rag_assistant.embeddings.hashing import HashingEmbedder
from rag_assistant.embeddings.sentence_transformer import SentenceTransformerEmbedder

__all__ = [
    "Embedder",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "build_embedder",
]
