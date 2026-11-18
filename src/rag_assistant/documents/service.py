"""Document ingestion service (Features 1 and 2).

Lifecycle: ``pending -> processing -> completed`` (or ``failed`` with a stored
error message). Processing is separated from upload so the HTTP layer can accept
a file immediately and do the expensive parse/embed work in a background task.

The service is deliberately synchronous and framework free; FastAPI wraps
:meth:`DocumentService.process` in a BackgroundTask, scripts call it directly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from uuid import uuid4

from rag_assistant.config import Settings, get_settings
from rag_assistant.db.models import DocumentStatus
from rag_assistant.db.repository import Repository
from rag_assistant.documents.chunking import chunk_pages
from rag_assistant.documents.extract import PdfExtractionError, extract_pdf
from rag_assistant.documents.storage import DocumentStorage, sanitise_filename
from rag_assistant.embeddings.base import Embedder
from rag_assistant.logging_config import get_logger
from rag_assistant.vectorstore.base import ChunkRecord, VectorStore

logger = get_logger(__name__)

ALLOWED_CONTENT_TYPES = {"application/pdf", "application/octet-stream", "binary/octet-stream", ""}


class DocumentValidationError(ValueError):
    """The upload is not a valid, safe PDF."""


class DocumentNotFoundError(LookupError):
    """No document exists with the requested id."""


class DocumentProcessingError(RuntimeError):
    """Extraction or embedding failed for a structurally valid PDF."""


class DocumentService:
    def __init__(
        self,
        *,
        repository: Repository,
        embedder: Embedder,
        vector_store: VectorStore,
        settings: Settings | None = None,
        storage: DocumentStorage | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.repository = repository
        self.embedder = embedder
        self.vector_store = vector_store
        self.storage = storage or DocumentStorage(self.settings.upload_dir)

    # ------------------------------------------------------------------ #
    # Upload
    # ------------------------------------------------------------------ #
    def register_upload(
        self,
        *,
        filename: str,
        content: bytes,
        content_type: str = "application/pdf",
    ) -> tuple[dict[str, Any], bool]:
        """Validate and persist an upload.

        Returns ``(document, created)``. ``created`` is ``False`` when an
        identical, already completed document exists (dedup by SHA-256).
        """
        if not content:
            raise DocumentValidationError("The uploaded file is empty.")

        safe_name = sanitise_filename(filename)
        if not safe_name.lower().endswith(".pdf"):
            raise DocumentValidationError("Only PDF documents are supported right now.")

        if content_type not in ALLOWED_CONTENT_TYPES:
            raise DocumentValidationError(
                f"Unsupported content type '{content_type}'. Upload an application/pdf file."
            )

        if len(content) > self.settings.max_upload_bytes:
            raise DocumentValidationError(
                f"File is {len(content) / 1024 / 1024:.1f} MB; the limit is "
                f"{self.settings.max_upload_mb} MB."
            )

        if b"%PDF-" not in content[:1024]:
            raise DocumentValidationError("The file does not look like a PDF (missing %PDF header).")

        digest = hashlib.sha256(content).hexdigest()
        existing = self.repository.find_document_by_hash(digest)
        if existing is not None:
            logger.info("Duplicate upload detected (sha256=%s)", digest[:12])
            return existing, False

        document_id = str(uuid4())
        try:
            self.storage.save(document_id, content)
        except OSError as exc:
            raise DocumentValidationError(f"Could not store the upload: {exc}") from exc

        document = self.repository.create_document(
            document_id=document_id,
            filename=safe_name,
            original_filename=filename,
            content_type=content_type or "application/pdf",
            size_bytes=len(content),
            sha256=digest,
        )
        return document, True

    # ------------------------------------------------------------------ #
    # Processing
    # ------------------------------------------------------------------ #
    def process(self, document_id: str) -> dict[str, Any]:
        """Extract, chunk, embed and index a document.

        Errors are captured on the document row (``status=failed``) instead of
        being raised, so a background worker cannot crash the API process.
        """
        document = self.repository.get_document(document_id)
        if document is None:
            raise DocumentNotFoundError(f"Document '{document_id}' does not exist.")

        path = self.storage.path(document_id)
        if path is None:
            return self.repository.update_document(
                document_id,
                status=DocumentStatus.FAILED,
                error_message="Stored file is missing; upload the document again.",
            ) or document

        self.repository.update_document(
            document_id, status=DocumentStatus.PROCESSING, clear_error=True
        )

        try:
            extraction = extract_pdf(path)
            chunks: list[ChunkRecord] = chunk_pages(
                extraction.pages,
                document_id=document_id,
                filename=document["filename"],
                chunk_size=self.settings.chunk_size,
                overlap=self.settings.chunk_overlap,
            )
            if not chunks:
                raise DocumentProcessingError("No chunks were produced from this PDF.")

            vectors = self.embedder.encode([chunk.text for chunk in chunks])
            # Reprocessing must not duplicate vectors.
            self.vector_store.delete_document(document_id)
            stored = self.vector_store.upsert_chunks(chunks, vectors)

            updated = self.repository.update_document(
                document_id,
                status=DocumentStatus.COMPLETED,
                page_count=extraction.page_count,
                chunk_count=stored,
                processed=True,
                clear_error=True,
            )
            logger.info(
                "Indexed '%s': %d pages -> %d chunks",
                document["filename"],
                extraction.page_count,
                stored,
            )
            return updated or document

        except (PdfExtractionError, DocumentProcessingError, ValueError) as exc:
            return self._mark_failed(document_id, exc)
        except Exception as exc:  # noqa: BLE001 - embeddings/vector store failures
            logger.exception("Unexpected failure while processing %s", document_id)
            return self._mark_failed(document_id, exc)

    def _mark_failed(self, document_id: str, exc: Exception) -> dict[str, Any]:
        logger.error("Processing failed for %s: %s", document_id, exc)
        try:
            self.vector_store.delete_document(document_id)
        except Exception:  # noqa: BLE001 - best effort cleanup
            logger.warning("Could not clean partial vectors for %s", document_id)
        updated = self.repository.update_document(
            document_id,
            status=DocumentStatus.FAILED,
            error_message=f"{type(exc).__name__}: {exc}"[:1000],
            clear_error=False,
        )
        if updated is None:  # pragma: no cover - deleted mid flight
            raise DocumentNotFoundError(document_id)
        return updated

    # ------------------------------------------------------------------ #
    # Reads / deletes
    # ------------------------------------------------------------------ #
    def get_document(self, document_id: str) -> dict[str, Any]:
        document = self.repository.get_document(document_id)
        if document is None:
            raise DocumentNotFoundError(f"Document '{document_id}' does not exist.")
        return document

    def list_documents(self, *, status: str | None = None) -> list[dict[str, Any]]:
        return self.repository.list_documents(status=status)

    def delete(self, document_id: str) -> bool:
        document = self.repository.get_document(document_id)
        if document is None:
            return False

        try:
            self.vector_store.delete_document(document_id)
        except Exception:  # noqa: BLE001
            logger.warning("Vector cleanup failed for %s", document_id, exc_info=True)

        self.storage.delete(document_id)
        return self.repository.delete_document(document_id)

    def ingest_file(self, path: str | Path) -> dict[str, Any]:
        """Convenience helper for scripts: upload + process in one call."""
        file_path = Path(path)
        document, _created = self.register_upload(
            filename=file_path.name,
            content=file_path.read_bytes(),
        )
        if document["status"] == DocumentStatus.COMPLETED.value:
            return document
        return self.process(document["id"])


__all__ = [
    "DocumentNotFoundError",
    "DocumentProcessingError",
    "DocumentService",
    "DocumentValidationError",
]
