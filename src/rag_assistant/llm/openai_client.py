"""OpenAI-compatible chat completion client (Feature 5).

Works with OpenAI and any provider exposing the same ``/chat/completions``
contract (Azure OpenAI gateways, Together, Groq, vLLM, Ollama's compatibility
endpoint). Using httpx directly keeps the dependency footprint small.
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from rag_assistant.llm.base import ChatMessage, LLMClient, LLMError, LLMResult
from rag_assistant.logging_config import get_logger
from rag_assistant.vectorstore.base import RetrievedChunk

logger = get_logger(__name__)


class OpenAICompatibleLLM(LLMClient):
    provider = "openai-compatible"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
        temperature: float = 0.0,
        max_tokens: int = 600,
    ) -> None:
        if not api_key:
            raise ValueError("An API key is required for the OpenAI-compatible provider.")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _messages(
        self,
        system_prompt: str,
        user_prompt: str,
        history: Sequence[ChatMessage] | None,
    ) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": system_prompt}]
        for message in history or []:
            if message.role in {"user", "assistant"}:
                messages.append({"role": message.role, "content": message.content})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def answer(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        chunks: Sequence[RetrievedChunk],
        history: Sequence[ChatMessage] | None = None,
    ) -> LLMResult:
        payload = {
            "model": self.model,
            "messages": self._messages(system_prompt, user_prompt, history),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

        if response.status_code >= 400:
            detail = response.text[:300]
            raise LLMError(f"LLM provider returned HTTP {response.status_code}: {detail}")

        try:
            data = response.json()
            text = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError) as exc:
            raise LLMError("LLM provider returned an unexpected payload.") from exc

        usage = data.get("usage") or {}
        return LLMResult(
            text=str(text).strip(),
            model=str(data.get("model", self.model)),
            provider=self.provider,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )


__all__ = ["OpenAICompatibleLLM"]
