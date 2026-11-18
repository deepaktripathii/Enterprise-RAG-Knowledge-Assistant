"""PDF ingestion: extraction, chunking, storage and orchestration."""

from rag_assistant.documents.chunking import chunk_pages
from rag_assistant.documents.extract import (
    ExtractionResult,
    PageText,
    PdfExtractionError,
    clean_text,
    extract_pdf,
)
from rag_assistant.documents.service import (
    DocumentNotFoundError,
    DocumentProcessingError,
    DocumentService,
    DocumentValidationError,
)
from rag_assistant.documents.storage import DocumentStorage, sanitise_filename

__all__ = [
    "DocumentNotFoundError",
    "DocumentProcessingError",
    "DocumentService",
    "DocumentStorage",
    "DocumentValidationError",
    "ExtractionResult",
    "PageText",
    "PdfExtractionError",
    "chunk_pages",
    "clean_text",
    "extract_pdf",
    "sanitise_filename",
]
