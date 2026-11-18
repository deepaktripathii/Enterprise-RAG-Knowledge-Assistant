"""Answer generation providers and prompt construction."""

from rag_assistant.llm.base import ChatMessage, LLMClient, LLMError, LLMResult
from rag_assistant.llm.extractive import ExtractiveLLM
from rag_assistant.llm.factory import build_llm
from rag_assistant.llm.openai_client import OpenAICompatibleLLM
from rag_assistant.llm.prompts import REFUSAL, SYSTEM_PROMPT, build_context, build_user_prompt

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
