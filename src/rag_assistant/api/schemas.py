"""Pydantic schemas for the RAG Knowledge Assistant API (Feature 8)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

DocumentStatusLiteral = Literal["pending", "processing", "completed", "failed"]


class DocumentOut(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    filename: str
    original_filename: str
    content_type: str
    size_bytes: int
    sha256: str
    page_count: int
    chunk_count: int
    status: DocumentStatusLiteral
    error_message: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    processed_at: str | None = None


class ProcessResponse(BaseModel):
    document: DocumentOut
    message: str


class CitationOut(BaseModel):
    index: int
    document_id: str
    filename: str
    page: int
    chunk_index: int = 0
    score: float
    snippet: str


class SourceOut(BaseModel):
    document_id: str
    filename: str
    page: int
    chunk_index: int
    score: float


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "question": "How many days per week can I work remotely?",
                "conversation_id": None,
                "top_k": 5,
            }
        },
    )

    question: str = Field(..., min_length=2, max_length=2000)
    conversation_id: str | None = Field(
        default=None, description="Continue an existing conversation when provided."
    )
    top_k: int | None = Field(default=None, ge=1, le=20)
    document_ids: list[str] | None = Field(
        default=None, description="Restrict retrieval to specific documents."
    )

    @field_validator("question")
    @classmethod
    def _strip_question(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) < 2:
            raise ValueError("Question must contain at least two characters.")
        return cleaned


class ChatResponse(BaseModel):
    conversation_id: str
    message_id: int
    question: str
    answer: str
    citations: list[CitationOut] = Field(default_factory=list)
    sources: list[SourceOut] = Field(default_factory=list)
    grounded: bool
    fallback: bool
    warnings: list[str] = Field(default_factory=list)
    model: str
    provider: str
    retrieval_latency_ms: float
    llm_latency_ms: float
    total_latency_ms: float
    usage: dict[str, int] = Field(default_factory=dict)
    created_at: str


class MessageOut(BaseModel):
    id: int
    conversation_id: str
    role: str
    content: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    model: str | None = None
    latency_ms: float | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None


class ConversationOut(BaseModel):
    id: str
    title: str
    message_count: int
    created_at: str | None = None
    updated_at: str | None = None


class ConversationDetail(ConversationOut):
    messages: list[MessageOut] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    embedder: str
    embedding_dimensions: int
    llm_provider: str
    llm_model: str
    collection: str
    vectors: int
    documents: dict[str, int]
    components: dict[str, str]
    timestamp: str


class QueryLogOut(BaseModel):
    id: int
    conversation_id: str | None
    question: str
    answer_preview: str
    retrieved_chunks: int
    top_score: float | None
    citations_count: int
    fallback: bool
    retrieval_latency_ms: float
    llm_latency_ms: float
    total_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    status: str
    error: str | None = None
    created_at: str | None = None


class MonitoringSummary(BaseModel):
    documents: dict[str, int]
    vectors: int = 0
    conversations: int
    total_queries: int
    window_size: int
    avg_retrieval_latency_ms: float | None
    avg_llm_latency_ms: float | None
    avg_total_latency_ms: float | None
    p95_total_latency_ms: float | None
    avg_top_score: float | None
    fallback_rate: float | None
    avg_citations: float | None
    tokens: dict[str, int]
    models: dict[str, int]
    daily: list[dict[str, Any]]
    last_query_at: str | None
    queries_last_24h: int


__all__ = [
    "ChatRequest",
    "ChatResponse",
    "CitationOut",
    "ConversationDetail",
    "ConversationOut",
    "DocumentOut",
    "HealthResponse",
    "MessageOut",
    "MonitoringSummary",
    "ProcessResponse",
    "QueryLogOut",
    "SourceOut",
]
