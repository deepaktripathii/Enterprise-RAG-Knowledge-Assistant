"""Retrieval-augmented generation pipeline."""

from sourceiq.rag.citations import citation_from_chunk, extract_citations
from sourceiq.rag.generator import AnswerGenerator
from sourceiq.rag.retriever import RetrievalResult, Retriever
from sourceiq.rag.service import (
    FALLBACK_ANSWER,
    ConversationNotFoundError,
    RagAnswer,
    RagGenerationError,
    RagService,
    expand_query,
)

__all__ = [
    "FALLBACK_ANSWER",
    "AnswerGenerator",
    "ConversationNotFoundError",
    "RagAnswer",
    "RagGenerationError",
    "RagService",
    "RetrievalResult",
    "Retriever",
    "citation_from_chunk",
    "expand_query",
    "extract_citations",
]
