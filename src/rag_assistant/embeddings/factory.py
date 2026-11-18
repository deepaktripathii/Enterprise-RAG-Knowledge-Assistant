"""Embedder factory with a graceful production -> offline fallback."""

from __future__ import annotations

from rag_assistant.config import Settings, get_settings
from rag_assistant.embeddings.base import Embedder
from rag_assistant.embeddings.hashing import HashingEmbedder
from rag_assistant.embeddings.sentence_transformer import SentenceTransformerEmbedder
from rag_assistant.logging_config import get_logger

logger = get_logger(__name__)


def build_embedder(settings: Settings | None = None, *, backend: str | None = None) -> Embedder:
    """Create the configured embedder.

    ``auto`` tries Sentence Transformers first and silently degrades to the
    deterministic hashing embedder, which keeps the API usable in air-gapped
    environments. The chosen backend is always logged and exposed by
    ``/health``.
    """
    cfg = settings or get_settings()
    selected = (backend or cfg.embedding_backend).lower()

    if selected == "hashing":
        return HashingEmbedder(cfg.hashing_dimensions)

    if selected in {"sentence-transformers", "sentence_transformers", "st"}:
        return SentenceTransformerEmbedder(cfg.embedding_model)

    if selected != "auto":
        raise ValueError(
            f"Unknown embedding backend '{selected}'. "
            "Use 'auto', 'sentence-transformers' or 'hashing'."
        )

    try:
        embedder = SentenceTransformerEmbedder(cfg.embedding_model)
        logger.info("Embedding backend: %s (%d dims)", embedder.name, embedder.dimensions)
        return embedder
    except Exception as exc:  # noqa: BLE001 - import/model download failures
        logger.warning(
            "Sentence Transformers unavailable (%s). Falling back to the hashing embedder.",
            exc,
        )
        embedder = HashingEmbedder(cfg.hashing_dimensions)
        logger.info("Embedding backend: %s (%d dims)", embedder.name, embedder.dimensions)
        return embedder


__all__ = ["build_embedder"]
