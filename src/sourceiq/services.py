"""Service container shared by the API, scripts and tests.

Keeping the wiring in one place means the API, the ingestion scripts and the
evaluation runner all use the same embedder, vector collection and LLM. Tests can
build the same container against a temporary directory.
"""

from __future__ import annotations

from dataclasses import dataclass

from sourceiq.config import Settings, get_settings
from sourceiq.db.repository import Repository
from sourceiq.documents.service import DocumentService
from sourceiq.embeddings.base import Embedder
from sourceiq.embeddings.factory import build_embedder
from sourceiq.llm.base import LLMClient
from sourceiq.llm.extractive import ExtractiveLLM
from sourceiq.llm.factory import build_llm
from sourceiq.rag.retriever import Retriever
from sourceiq.rag.service import RagService
from sourceiq.vectorstore.base import VectorStore
from sourceiq.vectorstore.qdrant_store import QdrantVectorStore


@dataclass
class Services:
    settings: Settings
    repository: Repository
    embedder: Embedder
    vector_store: VectorStore
    llm: LLMClient
    document_service: DocumentService
    rag_service: RagService

    def close(self) -> None:
        self.vector_store.close()
        self.repository.close()


def build_services(
    settings: Settings | None = None,
    *,
    embedding_backend: str | None = None,
    llm_provider: str | None = None,
) -> Services:
    cfg = settings or get_settings()
    cfg.ensure_directories()

    repository = Repository(settings=cfg)
    embedder = build_embedder(cfg, backend=embedding_backend)
    vector_store = QdrantVectorStore(dimensions=embedder.dimensions, settings=cfg)
    llm = build_llm(cfg, provider=llm_provider)

    document_service = DocumentService(
        repository=repository,
        embedder=embedder,
        vector_store=vector_store,
        settings=cfg,
    )
    rag_service = RagService(
        repository=repository,
        retriever=Retriever(embedder=embedder, vector_store=vector_store, settings=cfg),
        llm=llm,
        settings=cfg,
        fallback_llm=ExtractiveLLM() if llm.provider != "extractive" else None,
    )

    return Services(
        settings=cfg,
        repository=repository,
        embedder=embedder,
        vector_store=vector_store,
        llm=llm,
        document_service=document_service,
        rag_service=rag_service,
    )


__all__ = ["Services", "build_services"]
