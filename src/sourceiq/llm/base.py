"""LLM client contract (Feature 5).

Two implementations ship with the project: an OpenAI-compatible HTTP client and
a deterministic extractive answerer. Both receive the *retrieved chunks*, not
just a rendered prompt, so the extractive fallback can build a grounded answer
without re-parsing text.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

from sourceiq.vectorstore.base import RetrievedChunk


class LLMError(RuntimeError):
    """The generation provider failed or returned an unusable response."""


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass
class LLMResult:
    text: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient(ABC):
    provider: str
    model: str

    @abstractmethod
    def answer(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        chunks: Sequence[RetrievedChunk],
        history: Sequence[ChatMessage] | None = None,
    ) -> LLMResult:
        """Produce a grounded answer for the question embedded in ``user_prompt``."""


__all__ = ["ChatMessage", "LLMClient", "LLMError", "LLMResult"]
