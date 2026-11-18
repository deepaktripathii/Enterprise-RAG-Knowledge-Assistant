"""Context assembly and grounded answer generation (Feature 5)."""

from __future__ import annotations

from collections.abc import Sequence

from rag_assistant.config import Settings, get_settings
from rag_assistant.llm.base import ChatMessage, LLMClient, LLMResult
from rag_assistant.llm.prompts import SYSTEM_PROMPT, build_context, build_user_prompt
from rag_assistant.vectorstore.base import RetrievedChunk


class AnswerGenerator:
    def __init__(self, llm: LLMClient, settings: Settings | None = None) -> None:
        self.llm = llm
        self.settings = settings or get_settings()

    def generate(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        history: Sequence[dict[str, object]] | None = None,
    ) -> LLMResult:
        context = build_context(chunks, max_chars=self.settings.max_context_chars)
        user_prompt = build_user_prompt(question, context)
        chat_history = [
            ChatMessage(role=str(message["role"]), content=str(message["content"]))
            for message in history or []
            if message.get("role") in {"user", "assistant"}
        ]
        return self.llm.answer(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            chunks=chunks,
            history=chat_history,
        )


__all__ = ["AnswerGenerator"]
