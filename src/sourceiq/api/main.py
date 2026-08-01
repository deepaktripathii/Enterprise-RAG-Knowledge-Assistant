"""FastAPI application (Feature 8).

Endpoints
---------
``POST   /documents/upload``      upload a PDF (async indexing)
``GET    /documents``             list documents and processing status
``GET    /documents/{id}``        single document
``POST   /documents/{id}/process`` retry indexing synchronously
``DELETE /documents/{id}``        delete the file, vectors and metadata
``POST   /chat``                  grounded answer + citations + history
``GET    /conversations``         conversation list
``GET    /conversations/{id}``    transcript
``DELETE /conversations/{id}``    clear a conversation
``GET    /monitoring/summary``    volume, latency, token usage, fallbacks
``GET    /monitoring/queries``    recent query log entries
``GET    /evaluation/latest``     last RAG evaluation report
``POST   /evaluation/run``        run the fixed evaluation dataset
``GET    /health``                dependency health

The service degrades gracefully: if Qdrant or the LLM is unavailable the
affected endpoints return actionable errors while ``/health`` reports the exact
component that is down.
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware

from sourceiq import __version__
from sourceiq.api.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationDetail,
    ConversationOut,
    DocumentOut,
    HealthResponse,
    MonitoringSummary,
    ProcessResponse,
    QueryLogOut,
)
from sourceiq.config import Settings, get_settings
from sourceiq.db.models import DocumentStatus
from sourceiq.db.repository import Repository
from sourceiq.documents.service import (
    DocumentNotFoundError,
    DocumentService,
    DocumentValidationError,
)
from sourceiq.evaluation.runner import run_evaluation, save_report
from sourceiq.logging_config import get_logger
from sourceiq.rag.service import (
    ConversationNotFoundError,
    RagGenerationError,
    RagService,
)
from sourceiq.services import build_services

logger = get_logger(__name__)

DESCRIPTION = """
Turn an internal document library into a citation-backed knowledge assistant.

Upload PDFs, ask questions in natural language, and get answers grounded in the
retrieved passages with document + page citations. Conversations are persisted,
every query is logged, and the retrieval/answer quality can be evaluated against
a fixed dataset from `POST /evaluation/run`.
"""


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        services = build_services(cfg)

        app.state.settings = cfg
        app.state.repository = services.repository
        app.state.embedder = services.embedder
        app.state.vector_store = services.vector_store
        app.state.llm = services.llm
        app.state.document_service = services.document_service
        app.state.rag_service = services.rag_service
        app.state.started_at = datetime.now(timezone.utc).isoformat()

        logger.info(
            "SourceIQ ready | embedder=%s | llm=%s | collection=%s | vectors=%d",
            services.embedder.name,
            services.llm.model,
            services.vector_store.collection,
            services.vector_store.count(),
        )
        try:
            yield
        finally:
            services.close()

    app = FastAPI(
        title=cfg.app_name,
        version=cfg.app_version,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.cors_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_process_time(request: Request, call_next):  # type: ignore[no-untyped-def]
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - started) * 1000:.2f}"
        return response

    # ------------------------------------------------------------------ #
    # State helpers
    # ------------------------------------------------------------------ #
    def _documents(request: Request) -> DocumentService:
        service = getattr(request.app.state, "document_service", None)
        if service is None:  # pragma: no cover - lifespan always sets this
            raise HTTPException(status_code=503, detail="Document service is not ready.")
        return service

    def _rag(request: Request) -> RagService:
        service = getattr(request.app.state, "rag_service", None)
        if service is None:  # pragma: no cover
            raise HTTPException(status_code=503, detail="RAG service is not ready.")
        return service

    def _repository(request: Request) -> Repository:
        repository = getattr(request.app.state, "repository", None)
        if repository is None:  # pragma: no cover
            raise HTTPException(status_code=503, detail="Database is not ready.")
        return repository

    # ------------------------------------------------------------------ #
    # Meta
    # ------------------------------------------------------------------ #
    @app.get("/", tags=["meta"])
    def root() -> dict[str, str]:
        return {
            "name": "SourceIQ",
            "version": __version__,
            "docs": "/docs",
            "health": "/health",
        }

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health(request: Request) -> HealthResponse:
        state = request.app.state
        components: dict[str, str] = {}
        documents: dict[str, int] = {}
        vectors = 0

        try:
            documents = _repository(request).document_counts()
            components["database"] = "ok"
        except Exception:  # noqa: BLE001 - health must never raise
            components["database"] = "unavailable"

        store = getattr(state, "vector_store", None)
        if store is not None and store.health():
            components["vector_store"] = "ok"
            vectors = store.count()
        else:
            components["vector_store"] = "unavailable"

        embedder = getattr(state, "embedder", None)
        llm = getattr(state, "llm", None)
        components["embedding"] = "ok" if embedder is not None else "unavailable"
        components["llm"] = "ok" if llm is not None else "unavailable"

        healthy = all(value == "ok" for value in components.values())
        return HealthResponse(
            status="ok" if healthy else "degraded",
            version=__version__,
            embedder=getattr(embedder, "name", "unknown"),
            embedding_dimensions=int(getattr(embedder, "dimensions", 0)),
            llm_provider=getattr(llm, "provider", "unknown"),
            llm_model=getattr(llm, "model", "unknown"),
            collection=store.collection if store is not None else "unknown",
            vectors=vectors,
            documents=documents,
            components=components,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # ------------------------------------------------------------------ #
    # Documents
    # ------------------------------------------------------------------ #
    @app.post(
        "/documents/upload",
        response_model=DocumentOut,
        status_code=202,
        tags=["documents"],
        responses={200: {"description": "Identical document already indexed"}},
    )
    def upload_document(
        request: Request,
        response: Response,
        background_tasks: BackgroundTasks,
        file: UploadFile = File(..., description="PDF file to index"),
    ) -> DocumentOut:
        service = _documents(request)
        content = file.file.read()

        try:
            document, created = service.register_upload(
                filename=file.filename or "document.pdf",
                content=content,
                content_type=file.content_type or "",
            )
        except DocumentValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        response.headers["Location"] = f"/documents/{document['id']}"
        if created:
            background_tasks.add_task(service.process, document["id"])
            response.status_code = 202
            response.headers["X-Document-Status"] = "queued"
        else:
            response.status_code = 200
            response.headers["X-Duplicate"] = "true"
        return DocumentOut(**document)

    @app.get("/documents", response_model=list[DocumentOut], tags=["documents"])
    def list_documents(
        request: Request,
        status: str | None = Query(default=None, description="pending|processing|completed|failed"),
        limit: int = Query(default=200, ge=1, le=500),
    ) -> list[DocumentOut]:
        valid = {value.value for value in DocumentStatus}
        if status and status not in valid:
            raise HTTPException(status_code=400, detail=f"Invalid status '{status}'.")
        documents = _documents(request).list_documents(status=status)
        return [DocumentOut(**document) for document in documents[:limit]]

    @app.get("/documents/{document_id}", response_model=DocumentOut, tags=["documents"])
    def get_document(document_id: str, request: Request) -> DocumentOut:
        try:
            document = _documents(request).get_document(document_id)
        except DocumentNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return DocumentOut(**document)

    @app.post("/documents/{document_id}/process", response_model=ProcessResponse, tags=["documents"])
    def process_document(document_id: str, request: Request) -> ProcessResponse:
        service = _documents(request)
        try:
            updated = service.process(document_id)
        except DocumentNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        message = "Indexed" if updated["status"] == DocumentStatus.COMPLETED.value else "Processing failed"
        return ProcessResponse(document=DocumentOut(**updated), message=message)

    @app.delete("/documents/{document_id}", status_code=204, tags=["documents"])
    def delete_document(document_id: str, request: Request) -> Response:
        deleted = _documents(request).delete(document_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Document '{document_id}' does not exist.")
        return Response(status_code=204)

    # ------------------------------------------------------------------ #
    # Chat
    # ------------------------------------------------------------------ #
    @app.post("/chat", response_model=ChatResponse, tags=["chat"])
    def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        try:
            answer = _rag(request).ask(
                payload.question,
                conversation_id=payload.conversation_id,
                top_k=payload.top_k,
                document_ids=payload.document_ids,
            )
        except ConversationNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RagGenerationError as exc:
            raise HTTPException(status_code=502, detail=f"Answer generation failed: {exc}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return ChatResponse(**answer.to_dict())

    # ------------------------------------------------------------------ #
    # Conversations
    # ------------------------------------------------------------------ #
    @app.get("/conversations", response_model=list[ConversationOut], tags=["conversations"])
    def list_conversations(
        request: Request,
        limit: int = Query(default=50, ge=1, le=200),
    ) -> list[ConversationOut]:
        return [
            ConversationOut(**conversation)
            for conversation in _repository(request).list_conversations(limit=limit)
        ]

    @app.get(
        "/conversations/{conversation_id}",
        response_model=ConversationDetail,
        tags=["conversations"],
    )
    def get_conversation(conversation_id: str, request: Request) -> ConversationDetail:
        repository = _repository(request)
        conversation = repository.get_conversation(conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' does not exist.")
        messages = repository.list_messages(conversation_id)
        return ConversationDetail(**conversation, messages=messages)

    @app.delete("/conversations/{conversation_id}", status_code=204, tags=["conversations"])
    def delete_conversation(conversation_id: str, request: Request) -> Response:
        if not _repository(request).delete_conversation(conversation_id):
            raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' does not exist.")
        return Response(status_code=204)

    # ------------------------------------------------------------------ #
    # Monitoring
    # ------------------------------------------------------------------ #
    @app.get("/monitoring/summary", response_model=MonitoringSummary, tags=["monitoring"])
    def monitoring_summary(request: Request) -> MonitoringSummary:
        summary = _repository(request).monitoring_summary()
        store = getattr(request.app.state, "vector_store", None)
        summary["vectors"] = store.count() if store is not None else 0
        return MonitoringSummary(**summary)

    @app.get("/monitoring/queries", response_model=list[QueryLogOut], tags=["monitoring"])
    def monitoring_queries(
        request: Request,
        limit: int = Query(default=50, ge=1, le=500),
    ) -> list[QueryLogOut]:
        return [QueryLogOut(**row) for row in _repository(request).recent_queries(limit=limit)]

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    @app.get("/evaluation/latest", tags=["evaluation"])
    def evaluation_latest(request: Request) -> dict[str, Any]:
        report_path = Path(
            getattr(request.app.state.settings, "artifact_dir", cfg.artifact_dir)
        ) / "evaluation" / "latest.json"
        if not report_path.exists():
            raise HTTPException(
                status_code=404,
                detail="No evaluation report yet. Run `POST /evaluation/run` or scripts/evaluate.py.",
            )
        return json.loads(report_path.read_text(encoding="utf-8"))

    @app.post("/evaluation/run", tags=["evaluation"])
    def evaluation_run(
        request: Request,
        top_k: int = Query(default=5, ge=1, le=10),
        limit: int | None = Query(default=None, ge=1, le=100),
    ) -> dict[str, Any]:
        rag = _rag(request)
        embedder = request.app.state.embedder
        dataset = Path(cfg.evaluation_dir) / "questions.jsonl"
        if not dataset.exists():
            raise HTTPException(status_code=400, detail=f"Evaluation dataset not found at {dataset}.")

        report = run_evaluation(
            rag,
            dataset_path=dataset,
            embedder=embedder,
            top_k=top_k,
            limit=limit,
        )
        save_report(report, Path(cfg.artifact_dir) / "evaluation" / "latest.json")
        return report

    return app


app = create_app()


__all__ = ["app", "create_app"]
