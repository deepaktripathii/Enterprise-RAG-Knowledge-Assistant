"""LLM factory: OpenAI when configured, extractive otherwise."""

from __future__ import annotations

from sourceiq.config import Settings, get_settings
from sourceiq.llm.base import LLMClient
from sourceiq.llm.extractive import ExtractiveLLM
from sourceiq.llm.openai_client import OpenAICompatibleLLM
from sourceiq.logging_config import get_logger

logger = get_logger(__name__)


def build_llm(settings: Settings | None = None, *, provider: str | None = None) -> LLMClient:
    cfg = settings or get_settings()
    selected = (provider or cfg.llm_provider).lower()

    if selected == "extractive":
        return ExtractiveLLM()

    if selected in {"openai", "openai-compatible", "openai_compatible"}:
        if not cfg.llm_api_key:
            raise ValueError(
                "SOURCEIQ_LLM_PROVIDER=openai requires SOURCEIQ_LLM_API_KEY to be set."
            )
        return OpenAICompatibleLLM(
            api_key=cfg.llm_api_key,
            model=cfg.llm_model,
            base_url=cfg.llm_base_url,
            timeout=cfg.llm_timeout,
            temperature=cfg.llm_temperature,
            max_tokens=cfg.llm_max_tokens,
        )

    if selected != "auto":
        raise ValueError(
            f"Unknown LLM provider '{selected}'. Use 'auto', 'openai' or 'extractive'."
        )

    if cfg.llm_api_key:
        logger.info("LLM provider: %s (%s)", cfg.llm_model, cfg.llm_base_url)
        return OpenAICompatibleLLM(
            api_key=cfg.llm_api_key,
            model=cfg.llm_model,
            base_url=cfg.llm_base_url,
            timeout=cfg.llm_timeout,
            temperature=cfg.llm_temperature,
            max_tokens=cfg.llm_max_tokens,
        )

    logger.info(
        "No SOURCEIQ_LLM_API_KEY configured; using the extractive grounded answerer. "
        "Set an API key in .env for fluent LLM answers."
    )
    return ExtractiveLLM()


__all__ = ["build_llm"]
