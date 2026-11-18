"""Runtime configuration for RAG Knowledge Assistant.

All tunables live here and can be overridden with ``RAG_*`` environment
variables (see ``.env.example``). A single cached ``Settings`` object is shared
by the API, the scripts and the evaluation runner so they always agree on
chunk sizes, retrieval depth and model choice.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser().resolve() if raw else default


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw.strip() if raw and raw.strip() else default


def _env_optional(name: str) -> str | None:
    raw = os.getenv(name)
    return raw.strip() if raw and raw.strip() else None


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Immutable, env-driven application settings."""

    project_root: Path
    data_dir: Path
    upload_dir: Path
    artifact_dir: Path
    sample_docs_dir: Path
    evaluation_dir: Path

    qdrant_path: Path
    qdrant_url: str | None
    qdrant_api_key: str | None
    qdrant_collection: str

    database_url: str

    embedding_backend: str
    embedding_model: str
    hashing_dimensions: int
    embedding_batch_size: int

    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int
    min_relevance_score: float

    max_history_messages: int
    max_context_chars: int
    query_expansion: bool

    llm_provider: str
    llm_model: str
    llm_base_url: str
    llm_api_key: str | None
    llm_temperature: float
    llm_timeout: float
    llm_max_tokens: int

    max_upload_mb: int
    app_name: str = "RAG Knowledge Assistant - Enterprise AI Knowledge Intelligence Platform"
    app_version: str = "1.0.0"
    cors_origins: tuple[str, ...] = ("*",)
    log_level: str = "INFO"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.upload_dir,
            self.artifact_dir,
            self.sample_docs_dir,
            self.evaluation_dir,
            self.artifact_dir / "evaluation",
            self.qdrant_path,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def with_storage(self, root: Path | str) -> Settings:
        """Point every mutable path at ``root`` (used by tests and scripts)."""
        from dataclasses import replace

        base = Path(root).expanduser().resolve()
        return replace(
            self,
            data_dir=base / "data",
            upload_dir=base / "data" / "uploads",
            artifact_dir=base / "artifacts",
            sample_docs_dir=base / "data" / "sample_documents",
            evaluation_dir=base / "data" / "evaluation",
            qdrant_path=base / ".qdrant",
            database_url=f"sqlite:///{(base / 'data' / 'rag_assistant.db').as_posix()}",
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    root = PROJECT_ROOT
    data_dir = _env_path("RAG_DATA_DIR", root / "data")
    artifact_dir = _env_path("RAG_ARTIFACT_DIR", root / "artifacts")

    return Settings(
        project_root=root,
        data_dir=data_dir,
        upload_dir=_env_path("RAG_UPLOAD_DIR", data_dir / "uploads"),
        artifact_dir=artifact_dir,
        sample_docs_dir=_env_path("RAG_SAMPLE_DOCS_DIR", data_dir / "sample_documents"),
        evaluation_dir=_env_path("RAG_EVALUATION_DIR", data_dir / "evaluation"),
        qdrant_path=_env_path("RAG_QDRANT_PATH", root / ".qdrant"),
        qdrant_url=_env_optional("RAG_QDRANT_URL"),
        qdrant_api_key=_env_optional("RAG_QDRANT_API_KEY"),
        qdrant_collection=_env_str("RAG_QDRANT_COLLECTION", "rag_assistant_chunks"),
        database_url=_env_str(
            "RAG_DATABASE_URL", f"sqlite:///{(data_dir / 'rag_assistant.db').as_posix()}"
        ),
        embedding_backend=_env_str("RAG_EMBEDDING_BACKEND", "auto"),
        embedding_model=_env_str(
            "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        hashing_dimensions=_env_int("RAG_HASHING_DIMENSIONS", 768),
        embedding_batch_size=_env_int("RAG_EMBEDDING_BATCH_SIZE", 32),
        chunk_size=_env_int("RAG_CHUNK_SIZE", 1000),
        chunk_overlap=_env_int("RAG_CHUNK_OVERLAP", 150),
        retrieval_top_k=_env_int("RAG_RETRIEVAL_TOP_K", 5),
        min_relevance_score=_env_float("RAG_MIN_RELEVANCE_SCORE", 0.2),
        max_history_messages=_env_int("RAG_MAX_HISTORY_MESSAGES", 8),
        max_context_chars=_env_int("RAG_MAX_CONTEXT_CHARS", 8000),
        query_expansion=_env_bool("RAG_QUERY_EXPANSION", True),
        llm_provider=_env_str("RAG_LLM_PROVIDER", "auto"),
        llm_model=_env_str("RAG_LLM_MODEL", "gpt-4o-mini"),
        llm_base_url=_env_str("RAG_LLM_BASE_URL", "https://api.openai.com/v1"),
        llm_api_key=_env_optional("RAG_LLM_API_KEY"),
        llm_temperature=_env_float("RAG_LLM_TEMPERATURE", 0.0),
        llm_timeout=_env_float("RAG_LLM_TIMEOUT", 60.0),
        llm_max_tokens=_env_int("RAG_LLM_MAX_TOKENS", 600),
        max_upload_mb=_env_int("RAG_MAX_UPLOAD_MB", 20),
        log_level=_env_str("RAG_LOG_LEVEL", "INFO"),
    )


def reset_settings_cache() -> None:
    """Clear the cached settings (tests, notebooks, long running processes)."""
    get_settings.cache_clear()


__all__ = ["PROJECT_ROOT", "Settings", "get_settings", "reset_settings_cache"]
