"""Retrieval-augmented generation pipeline."""

from rag_assistant.rag.citations import citation_from_chunk, extract_citations
from rag_assistant.rag.generator import AnswerGenerator
from rag_assistant.rag.retriever import RetrievalResult, Retriever
from rag_assistant.rag.service import (
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
