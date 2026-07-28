"""RAG orchestration (Features 5, 6 and 7).

One call to :meth:`RagService.ask` performs:

1. conversation lookup or creation,
2. history-aware query expansion,
3. embedding + Qdrant retrieval,
4. relevance gate (fallback when nothing is relevant),
5. grounded generation with a citation-first system prompt,
6. citation validation and persistence of messages + query log.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from sourceiq.config import Settings, get_settings
from sourceiq.db.repository import Repository
from sourceiq.llm.base import LLMClient, LLMError, LLMResult
from sourceiq.logging_config import get_logger
from sourceiq.rag.citations import extract_citations
from sourceiq.rag.generator import AnswerGenerator
from sourceiq.rag.retriever import Retriever
from sourceiq.vectorstore.base import RetrievedChunk

logger = get_logger(__name__)

FALLBACK_ANSWER = (
    "I could not find this information in the available documents. "
    "Try rephrasing the question, or upload the policy or manual that covers this topic."
)

_FOLLOW_UP = re.compile(
    r"^(it|its|they|them|that|this|those|these|what about|how about|and|also|the same)\b",
    re.IGNORECASE,
)


class ConversationNotFoundError(LookupError):
    """The requested conversation id does not exist."""


class RagGenerationError(RuntimeError):
    """The answer provider failed and no fallback was available."""


@dataclass
class RagAnswer:
    conversation_id: str
    message_id: int
    question: str
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    grounded: bool = False
    fallback: bool = False
    warnings: list[str] = field(default_factory=list)
    model: str = ""
    provider: str = ""
    retrieval_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)
    retrieved: list[RetrievedChunk] = field(default_factory=list, repr=False)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self, *, include_sources: bool = True) -> dict[str, Any]:
        data = asdict(self)
        # Raw chunks are used by the evaluator, not by API clients.
        data.pop("retrieved", None)
        if not include_sources:
            data.pop("sources", None)
        return data


def expand_query(
    question: str,
    history: Sequence[dict[str, Any]],
    *,
    enabled: bool = True,
) -> str:
    """Merge short follow-up questions with the previous user turn.

    Embeddings are computed for a single query string, so a follow-up like
    "and for two-year contracts?" would otherwise lose its subject. This
    lightweight rewrite keeps retrieval cheap and deterministic; a production
    system could replace it with an LLM query-rewrite step.
    """
    if not enabled or not history:
        return question

    previous = next(
        (message for message in reversed(history) if message.get("role") == "user"), None
    )
    if previous is None:
        return question

    words = question.split()
    if _FOLLOW_UP.match(question) or len(words) <= 4:
        combined = f"{previous['content']} {question}".strip()
        return combined[:300]
    return question


class RagService:
    def __init__(
        self,
        *,
        repository: Repository,
        retriever: Retriever,
        llm: LLMClient,
        settings: Settings | None = None,
        fallback_llm: LLMClient | None = None,
    ) -> None:
        self.repository = repository
        self.retriever = retriever
        self.llm = llm
        self.settings = settings or get_settings()
        self.generator = AnswerGenerator(llm, self.settings)
        self.fallback_llm = fallback_llm

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def ask(
        self,
        question: str,
        *,
        conversation_id: str | None = None,
        top_k: int | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> RagAnswer:
        query = (question or "").strip()
        if not query:
            raise ValueError("Question must not be empty.")

        conversation = self._resolve_conversation(conversation_id, query)
        conversation_id = conversation["id"]

        history = self.repository.recent_messages(
            conversation_id, limit=self.settings.max_history_messages
        )
        retrieval_query = expand_query(
            query, history, enabled=self.settings.query_expansion
        )

        retrieval = self.retriever.retrieve(
            retrieval_query, top_k=top_k, document_ids=document_ids
        )
        chunks = retrieval.chunks

        if not chunks or retrieval.top_score < self.settings.min_relevance_score:
            answer = RagAnswer(
                conversation_id=conversation_id,
                message_id=0,
                question=query,
                answer=FALLBACK_ANSWER,
                citations=[],
                sources=[],
                grounded=False,
                fallback=True,
                warnings=["below_relevance_threshold"] if chunks else ["no_retrieval_results"],
                model=self.llm.model,
                provider=self.llm.provider,
                retrieval_latency_ms=retrieval.latency_ms,
                retrieved=list(chunks),
            )
            return self._persist(answer, history)

        result, llm_latency, provider_warnings = self._generate(query, chunks, history)
        answer_text, citations, citation_warnings = extract_citations(result.text, chunks)

        if not answer_text.strip() or answer_text.strip().startswith("I could not find"):
            return self._persist(
                RagAnswer(
                    conversation_id=conversation_id,
                    message_id=0,
                    question=query,
                    answer=FALLBACK_ANSWER,
                    citations=[],
                    sources=[],
                    grounded=False,
                    fallback=True,
                    warnings=[*provider_warnings, "llm_refused"],
                    model=result.model,
                    provider=result.provider,
                    retrieval_latency_ms=retrieval.latency_ms,
                    llm_latency_ms=llm_latency,
                    usage=self._usage(result),
                    retrieved=list(chunks),
                ),
                history,
            )

        warnings = [*provider_warnings, *citation_warnings]
        grounded = "answer_missing_citations" not in citation_warnings

        answer = RagAnswer(
            conversation_id=conversation_id,
            message_id=0,
            question=query,
            answer=answer_text,
            citations=citations,
            sources=[chunk.to_dict(include_text=False) for chunk in chunks],
            grounded=grounded,
            fallback=False,
            warnings=warnings,
            model=result.model,
            provider=result.provider,
            retrieval_latency_ms=retrieval.latency_ms,
            llm_latency_ms=llm_latency,
            usage=self._usage(result),
            retrieved=list(chunks),
        )
        logger.info(
            "Answered '%s' with %d citations in %.0f ms",
            query[:60],
            len(citations),
            retrieval.latency_ms + llm_latency,
        )
        return self._persist(answer, history)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _resolve_conversation(
        self, conversation_id: str | None, question: str
    ) -> dict[str, Any]:
        if conversation_id:
            conversation = self.repository.get_conversation(conversation_id)
            if conversation is None:
                raise ConversationNotFoundError(
                    f"Conversation '{conversation_id}' does not exist."
                )
            return conversation
        return self.repository.create_conversation(title=question[:80])

    def _generate(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        history: Sequence[dict[str, Any]],
    ) -> tuple[LLMResult, float, list[str]]:
        warnings: list[str] = []
        started = time.perf_counter()
        try:
            result = self.generator.generate(question, chunks, history)
        except LLMError as exc:
            logger.error("LLM generation failed: %s", exc)
            if self.fallback_llm is None:
                raise RagGenerationError(str(exc)) from exc
            warnings.append("llm_fallback_used")
            fallback_generator = AnswerGenerator(self.fallback_llm, self.settings)
            result = fallback_generator.generate(question, chunks, history)
        return result, (time.perf_counter() - started) * 1000, warnings

    def _persist(self, answer: RagAnswer, history: Sequence[dict[str, Any]]) -> RagAnswer:
        self.repository.add_message(
            conversation_id=answer.conversation_id,
            role="user",
            content=answer.question,
        )
        assistant = self.repository.add_message(
            conversation_id=answer.conversation_id,
            role="assistant",
            content=answer.answer,
            citations=answer.citations,
            model=answer.model,
            latency_ms=answer.llm_latency_ms,
            usage=answer.usage,
        )
        self.repository.touch_conversation(answer.conversation_id)

        answer.message_id = int(assistant["id"])
        answer.total_latency_ms = (
            answer.retrieval_latency_ms + answer.llm_latency_ms
        )
        self.repository.log_query(
            question=answer.question,
            answer_preview=answer.answer,
            conversation_id=answer.conversation_id,
            retrieved_chunks=len(answer.sources),
            top_score=answer.sources[0]["score"] if answer.sources else None,
            citations_count=len(answer.citations),
            fallback=answer.fallback,
            retrieval_latency_ms=answer.retrieval_latency_ms,
            llm_latency_ms=answer.llm_latency_ms,
            total_latency_ms=answer.total_latency_ms,
            prompt_tokens=answer.usage.get("prompt_tokens", 0),
            completion_tokens=answer.usage.get("completion_tokens", 0),
            model=answer.model,
            status="fallback" if answer.fallback else "success",
        )
        return answer

    @staticmethod
    def _usage(result: LLMResult) -> dict[str, int]:
        return {
            "prompt_tokens": int(result.prompt_tokens),
            "completion_tokens": int(result.completion_tokens),
            "total_tokens": int(result.total_tokens),
        }


__all__ = [
    "ConversationNotFoundError",
    "FALLBACK_ANSWER",
    "RagAnswer",
    "RagGenerationError",
    "RagService",
    "expand_query",
]
