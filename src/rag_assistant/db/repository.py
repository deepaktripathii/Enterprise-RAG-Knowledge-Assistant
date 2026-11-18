"""Repository layer around the SQLAlchemy session.

Every method returns plain dictionaries so the API layer never touches a
detached ORM object. Sessions are short lived; SQLite runs with
``check_same_thread=False`` because FastAPI executes sync endpoints in a
threadpool.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from rag_assistant.config import Settings, get_settings
from rag_assistant.db.models import (
    Base,
    Conversation,
    Document,
    DocumentStatus,
    Message,
    QueryLog,
    isoformat,
    utcnow,
)
from rag_assistant.logging_config import get_logger

logger = get_logger(__name__)


class Repository:
    def __init__(
        self,
        database_url: str | None = None,
        *,
        settings: Settings | None = None,
        echo: bool = False,
    ) -> None:
        cfg = settings or get_settings()
        self.settings = cfg
        self.database_url = database_url or cfg.database_url

        if self.database_url.startswith("sqlite:///"):
            path = Path(self.database_url[len("sqlite:///") :])
            if str(path) != ":memory:":
                path.parent.mkdir(parents=True, exist_ok=True)

        connect_args = (
            {"check_same_thread": False} if self.database_url.startswith("sqlite") else {}
        )
        self.engine = create_engine(
            self.database_url, echo=echo, future=True, connect_args=connect_args
        )
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)

    # ------------------------------------------------------------------ #
    # Documents
    # ------------------------------------------------------------------ #
    def create_document(
        self,
        *,
        document_id: str | None = None,
        filename: str,
        original_filename: str,
        content_type: str,
        size_bytes: int,
        sha256: str,
        status: DocumentStatus = DocumentStatus.PENDING,
    ) -> dict[str, Any]:
        record = Document(
            id=document_id or str(uuid4()),
            filename=filename,
            original_filename=original_filename,
            content_type=content_type,
            size_bytes=size_bytes,
            sha256=sha256,
            status=status.value,
        )
        with self._session_factory() as session:
            session.add(record)
            session.commit()
            session.refresh(record)
            return record.to_dict()

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self._session_factory() as session:
            record = session.get(Document, document_id)
            return record.to_dict() if record else None

    def find_document_by_hash(self, sha256: str) -> dict[str, Any] | None:
        with self._session_factory() as session:
            record = session.scalars(
                select(Document)
                .where(Document.sha256 == sha256, Document.status == DocumentStatus.COMPLETED.value)
                .limit(1)
            ).first()
            return record.to_dict() if record else None

    def list_documents(
        self, *, status: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        with self._session_factory() as session:
            query = select(Document).order_by(Document.created_at.desc()).limit(limit)
            if status:
                query = query.where(Document.status == status)
            return [record.to_dict() for record in session.scalars(query).all()]

    def update_document(
        self,
        document_id: str,
        *,
        status: DocumentStatus | None = None,
        page_count: int | None = None,
        chunk_count: int | None = None,
        error_message: str | None = None,
        processed: bool = False,
        clear_error: bool = False,
    ) -> dict[str, Any] | None:
        with self._session_factory() as session:
            record = session.get(Document, document_id)
            if record is None:
                return None
            if status is not None:
                record.status = status.value
            if page_count is not None:
                record.page_count = page_count
            if chunk_count is not None:
                record.chunk_count = chunk_count
            if clear_error:
                record.error_message = None
            if error_message is not None:
                record.error_message = error_message
            if processed:
                record.processed_at = utcnow()
            record.updated_at = utcnow()
            session.commit()
            session.refresh(record)
            return record.to_dict()

    def delete_document(self, document_id: str) -> bool:
        with self._session_factory() as session:
            record = session.get(Document, document_id)
            if record is None:
                return False
            session.delete(record)
            session.commit()
            return True

    def document_counts(self) -> dict[str, int]:
        with self._session_factory() as session:
            rows = session.execute(
                select(Document.status, func.count(Document.id)).group_by(Document.status)
            ).all()
        counts = {status.value: 0 for status in DocumentStatus}
        for status, count in rows:
            counts[str(status)] = int(count)
        counts["total"] = sum(counts[status.value] for status in DocumentStatus)
        return counts

    # ------------------------------------------------------------------ #
    # Conversations & messages
    # ------------------------------------------------------------------ #
    def create_conversation(self, title: str | None = None) -> dict[str, Any]:
        record = Conversation(
            id=str(uuid4()),
            title=(title or "New conversation")[:200],
        )
        with self._session_factory() as session:
            session.add(record)
            session.commit()
            session.refresh(record)
            return self._conversation_dict(record, message_count=0)

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self._session_factory() as session:
            record = session.get(Conversation, conversation_id)
            if record is None:
                return None
            count = session.scalar(
                select(func.count(Message.id)).where(Message.conversation_id == conversation_id)
            )
            return self._conversation_dict(record, message_count=int(count or 0))

    def list_conversations(self, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._session_factory() as session:
            records = session.scalars(
                select(Conversation).order_by(Conversation.updated_at.desc()).limit(limit)
            ).all()
            counts = dict(
                session.execute(
                    select(Message.conversation_id, func.count(Message.id))
                    .where(Message.conversation_id.in_([record.id for record in records]))
                    .group_by(Message.conversation_id)
                ).all()
            )
            return [
                self._conversation_dict(record, message_count=int(counts.get(record.id, 0)))
                for record in records
            ]

    def touch_conversation(self, conversation_id: str, title: str | None = None) -> None:
        with self._session_factory() as session:
            record = session.get(Conversation, conversation_id)
            if record is None:
                return
            record.updated_at = utcnow()
            if title:
                record.title = title[:200]
            session.commit()

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._session_factory() as session:
            record = session.get(Conversation, conversation_id)
            if record is None:
                return False
            session.delete(record)
            session.commit()
            return True

    def add_message(
        self,
        *,
        conversation_id: str,
        role: str,
        content: str,
        citations: list[dict[str, Any]] | None = None,
        model: str | None = None,
        latency_ms: float | None = None,
        usage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            citations=citations or [],
            model=model,
            latency_ms=latency_ms,
            usage=usage or {},
        )
        with self._session_factory() as session:
            session.add(record)
            session.commit()
            session.refresh(record)
            return record.to_dict()

    def list_messages(
        self, conversation_id: str, *, limit: int | None = None
    ) -> list[dict[str, Any]]:
        with self._session_factory() as session:
            query = (
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.id.asc())
            )
            if limit:
                query = query.limit(limit)
            return [record.to_dict() for record in session.scalars(query).all()]

    def recent_messages(self, conversation_id: str, *, limit: int) -> list[dict[str, Any]]:
        with self._session_factory() as session:
            records = session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.id.desc())
                .limit(limit)
            ).all()
        return [record.to_dict() for record in reversed(records)]

    @staticmethod
    def _conversation_dict(record: Conversation, *, message_count: int) -> dict[str, Any]:
        return {
            "id": record.id,
            "title": record.title,
            "message_count": message_count,
            "created_at": isoformat(record.created_at),
            "updated_at": isoformat(record.updated_at),
        }

    # ------------------------------------------------------------------ #
    # Query logs & monitoring
    # ------------------------------------------------------------------ #
    def log_query(
        self,
        *,
        question: str,
        answer_preview: str,
        conversation_id: str | None,
        retrieved_chunks: int,
        top_score: float | None,
        citations_count: int,
        fallback: bool,
        retrieval_latency_ms: float,
        llm_latency_ms: float,
        total_latency_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        model: str = "",
        status: str = "success",
        error: str | None = None,
    ) -> dict[str, Any]:
        record = QueryLog(
            conversation_id=conversation_id,
            question=question[:2000],
            answer_preview=answer_preview[:500],
            retrieved_chunks=retrieved_chunks,
            top_score=top_score,
            citations_count=citations_count,
            fallback=fallback,
            retrieval_latency_ms=retrieval_latency_ms,
            llm_latency_ms=llm_latency_ms,
            total_latency_ms=total_latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model=model,
            status=status,
            error=error,
        )
        with self._session_factory() as session:
            session.add(record)
            session.commit()
            session.refresh(record)
            return record.to_dict()

    def recent_queries(self, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._session_factory() as session:
            records = session.scalars(
                select(QueryLog).order_by(QueryLog.id.desc()).limit(limit)
            ).all()
        return [record.to_dict() for record in records]

    def monitoring_summary(self, *, recent_limit: int = 500) -> dict[str, Any]:
        with self._session_factory() as session:
            records = session.scalars(
                select(QueryLog).order_by(QueryLog.id.desc()).limit(recent_limit)
            ).all()
            total_queries = int(session.scalar(select(func.count(QueryLog.id))) or 0)
            total_conversations = int(session.scalar(select(func.count(Conversation.id))) or 0)

        document_counts = self.document_counts()
        if not records:
            return {
                "documents": document_counts,
                "conversations": total_conversations,
                "total_queries": total_queries,
                "window_size": 0,
                "avg_retrieval_latency_ms": None,
                "avg_llm_latency_ms": None,
                "avg_total_latency_ms": None,
                "p95_total_latency_ms": None,
                "avg_top_score": None,
                "fallback_rate": None,
                "avg_citations": None,
                "tokens": {"prompt": 0, "completion": 0, "total": 0},
                "models": {},
                "daily": [],
                "last_query_at": None,
                "queries_last_24h": 0,
            }

        retrieval = np.asarray([row.retrieval_latency_ms for row in records], dtype=float)
        llm = np.asarray([row.llm_latency_ms for row in records], dtype=float)
        total = np.asarray([row.total_latency_ms for row in records], dtype=float)
        scores = np.asarray([row.top_score or 0.0 for row in records], dtype=float)
        now = datetime.now(timezone.utc)

        def _utc(value: datetime) -> datetime:
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

        daily: dict[str, list[float]] = {}
        for row in records:
            key = _utc(row.created_at).date().isoformat()
            daily.setdefault(key, []).append(row.total_latency_ms)

        return {
            "documents": document_counts,
            "conversations": total_conversations,
            "total_queries": total_queries,
            "window_size": len(records),
            "avg_retrieval_latency_ms": round(float(retrieval.mean()), 2),
            "avg_llm_latency_ms": round(float(llm.mean()), 2),
            "avg_total_latency_ms": round(float(total.mean()), 2),
            "p95_total_latency_ms": round(float(np.percentile(total, 95)), 2),
            "avg_top_score": round(float(scores.mean()), 4),
            "fallback_rate": round(float(np.mean([row.fallback for row in records])), 4),
            "avg_citations": round(
                float(np.mean([row.citations_count for row in records])), 2
            ),
            "tokens": {
                "prompt": int(sum(row.prompt_tokens for row in records)),
                "completion": int(sum(row.completion_tokens for row in records)),
                "total": int(sum(row.total_tokens for row in records)),
            },
            "models": dict(Counter(row.model for row in records)),
            "daily": [
                {
                    "date": key,
                    "queries": len(values),
                    "avg_latency_ms": round(float(np.mean(values)), 2),
                }
                for key, values in sorted(daily.items())
            ],
            "last_query_at": isoformat(max(_utc(row.created_at) for row in records)),
            "queries_last_24h": int(
                sum(1 for row in records if _utc(row.created_at) >= now - timedelta(hours=24))
            ),
        }

    def close(self) -> None:
        self.engine.dispose()


__all__ = ["Repository"]
