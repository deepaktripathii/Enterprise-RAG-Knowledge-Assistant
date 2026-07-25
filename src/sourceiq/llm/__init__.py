"""Answer generation providers and prompt construction."""

from sourceiq.llm.base import ChatMessage, LLMClient, LLMError, LLMResult
from sourceiq.llm.extractive import ExtractiveLLM
from sourceiq.llm.factory import build_llm
from sourceiq.llm.openai_client import OpenAICompatibleLLM
from sourceiq.llm.prompts import REFUSAL, SYSTEM_PROMPT, build_context, build_user_prompt

__all__ = [
    "REFUSAL",
    "SYSTEM_PROMPT",
    "ChatMessage",
    "ExtractiveLLM",
    "LLMClient",
    "LLMError",
    "LLMResult",
    "OpenAICompatibleLLM",
    "build_context",
    "build_llm",
    "build_user_prompt",
]
