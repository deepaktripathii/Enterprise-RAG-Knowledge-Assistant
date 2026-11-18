"""RAG behaviour: grounded answers, citations, history and fallbacks."""

from __future__ import annotations

import pytest

from rag_assistant.llm.base import LLMClient, LLMError, LLMResult
from rag_assistant.llm.extractive import ExtractiveLLM
from rag_assistant.rag.citations import extract_citations
from rag_assistant.rag.retriever import Retriever
from rag_assistant.rag.service import (
    ConversationNotFoundError,
    RagService,
    expand_query,
)
from rag_assistant.vectorstore.base import RetrievedChunk


def _chunk(text: str, page: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        document_id="doc-1",
        filename="policy.pdf",
        chunk_index=0,
        page=page,
        text=text,
        score=0.8,
    )


def test_answer_is_grounded_and_cited(services, ingested):
    answer = services.rag_service.ask("How many days per week can I work remotely?")

    assert answer.fallback is False
    assert answer.grounded is True
    assert "two days" in answer.answer.lower()
    assert answer.citations[0]["filename"] == "Remote_Work_Policy.pdf"
    assert answer.citations[0]["page"] == 1
    assert answer.message_id > 0
    assert answer.usage["total_tokens"] > 0


def test_fallback_when_nothing_is_retrieved(services):
    answer = services.rag_service.ask("What is the parental leave policy?")

    assert answer.fallback is True
    assert answer.grounded is False
    assert answer.citations == []
    assert "could not find" in answer.answer.lower()


def test_document_filter_limits_retrieval(services, ingested):
    answer = services.rag_service.ask(
        "How many days per week can I work remotely?",
        document_ids=[ingested["IT_Security_Policy"]],
    )

    # Retrieval must never return chunks from the excluded document; with a
    # restricted, irrelevant corpus the service either answers from the allowed
    # document or refuses.
    assert all(
        source["document_id"] == ingested["IT_Security_Policy"] for source in answer.sources
    )
    assert all(
        citation["document_id"] == ingested["IT_Security_Policy"]
        for citation in answer.citations
    )
    if answer.fallback:
        assert answer.citations == []


def test_conversation_history_is_persisted(services, ingested):
    first = services.rag_service.ask("How many days per week can I work remotely?")
    services.rag_service.ask(
        "and can I work from outside India?", conversation_id=first.conversation_id
    )

    messages = services.repository.list_messages(first.conversation_id)
    conversation = services.repository.get_conversation(first.conversation_id)

    assert len(messages) == 4
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert conversation is not None
    assert conversation["title"].startswith("How many days")
    assert conversation["message_count"] == 4


def test_unknown_conversation_raises(services):
    with pytest.raises(ConversationNotFoundError):
        services.rag_service.ask("hello", conversation_id="missing-conversation")


def test_query_expansion_for_follow_ups():
    history = [{"role": "user", "content": "How many days can I work remotely?"}]

    assert expand_query("and outside India?", history).startswith("How many days")
    assert expand_query("What is the password rotation policy?", history) == (
        "What is the password rotation policy?"
    )
    assert expand_query("and outside India?", history, enabled=False) == "and outside India?"


def test_invalid_citation_markers_are_removed():
    text, citations, warnings = extract_citations(
        "Employees work two days remotely [1]. Invented fact [9].",
        [_chunk("Employees work two days remotely.")],
    )

    assert "[9]" not in text
    assert len(citations) == 1
    assert "invalid_citation_markers_removed" in warnings


def test_missing_markers_fall_back_to_best_source():
    text, citations, warnings = extract_citations(
        "Employees work remotely two days per week.",
        [_chunk("Employees work remotely two days per week.")],
    )

    assert text.startswith("Employees")
    assert citations[0]["filename"] == "policy.pdf"
    assert "answer_missing_citations" in warnings


def test_llm_failure_uses_extractive_fallback(services, ingested):
    class BrokenLLM(LLMClient):
        provider = "broken"
        model = "broken-v1"

        def answer(self, *, system_prompt, user_prompt, chunks, history=None) -> LLMResult:
            raise LLMError("provider is down")

    rag = RagService(
        repository=services.repository,
        retriever=Retriever(
            embedder=services.embedder,
            vector_store=services.vector_store,
            settings=services.settings,
        ),
        llm=BrokenLLM(),
        settings=services.settings,
        fallback_llm=ExtractiveLLM(),
    )

    answer = rag.ask("How many days per week can I work remotely?")

    assert "llm_fallback_used" in answer.warnings
    assert answer.provider == "extractive"
    assert answer.fallback is False


def test_empty_question_is_rejected(services):
    with pytest.raises(ValueError):
        services.rag_service.ask("   ")
