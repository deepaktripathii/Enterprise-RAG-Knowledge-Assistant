"""Document lifecycle: validation, dedup, indexing and deletion."""

from __future__ import annotations

import pymupdf
import pytest

from rag_assistant.documents.service import DocumentValidationError


def _upload(services, path, *, content_type="application/pdf"):
    return services.document_service.register_upload(
        filename=path.name,
        content=path.read_bytes(),
        content_type=content_type,
    )


def test_upload_creates_pending_document(services, pdf_library):
    document, created = _upload(services, pdf_library["remote"])

    assert created is True
    assert document["status"] == "pending"
    assert document["size_bytes"] > 0
    assert services.document_service.storage.path(document["id"]).exists()


def test_processing_indexes_chunks(services, pdf_library):
    document, _ = _upload(services, pdf_library["remote"])

    processed = services.document_service.process(document["id"])

    assert processed["status"] == "completed"
    assert processed["page_count"] == 2
    assert processed["chunk_count"] >= 2
    assert processed["error_message"] is None
    assert services.vector_store.count() == processed["chunk_count"]


def test_duplicate_upload_returns_existing_document(services, pdf_library):
    first, _ = _upload(services, pdf_library["remote"])
    services.document_service.process(first["id"])

    second, created = _upload(services, pdf_library["remote"])

    assert created is False
    assert second["id"] == first["id"]
    assert services.repository.document_counts()["total"] == 1


def test_reprocessing_does_not_duplicate_vectors(services, pdf_library):
    document, _ = _upload(services, pdf_library["remote"])
    processed = services.document_service.process(document["id"])
    count = services.vector_store.count()

    reprocessed = services.document_service.process(document["id"])

    assert reprocessed["chunk_count"] == processed["chunk_count"]
    assert services.vector_store.count() == count


def test_delete_removes_file_metadata_and_vectors(services, pdf_library):
    document, _ = _upload(services, pdf_library["remote"])
    services.document_service.process(document["id"])
    assert services.vector_store.count() > 0

    assert services.document_service.delete(document["id"]) is True

    assert services.repository.get_document(document["id"]) is None
    assert services.document_service.storage.path(document["id"]) is None
    assert services.vector_store.count() == 0


@pytest.mark.parametrize(
    ("filename", "content", "content_type"),
    [
        ("notes.txt", b"%PDF-1.4 fake", "text/plain"),
        ("empty.pdf", b"", "application/pdf"),
        ("fake.pdf", b"this is not a pdf", "application/pdf"),
    ],
)
def test_invalid_uploads_are_rejected(services, filename, content, content_type):
    with pytest.raises(DocumentValidationError):
        services.document_service.register_upload(
            filename=filename, content=content, content_type=content_type
        )


def test_upload_size_limit(services, tmp_settings):
    oversized = b"%PDF-1.4" + b"0" * (tmp_settings.max_upload_bytes + 10)

    with pytest.raises(DocumentValidationError, match="limit"):
        services.document_service.register_upload(
            filename="big.pdf", content=oversized, content_type="application/pdf"
        )


def test_blank_pdf_is_marked_failed(services, tmp_path):
    blank = tmp_path / "blank.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(blank)
    document.close()

    uploaded, _ = _upload(services, blank)
    result = services.document_service.process(uploaded["id"])

    assert result["status"] == "failed"
    assert "no extractable text" in (result["error_message"] or "")
    assert services.vector_store.count() == 0


def test_processing_unknown_document_raises(services):
    from rag_assistant.documents.service import DocumentNotFoundError

    with pytest.raises(DocumentNotFoundError):
        services.document_service.process("does-not-exist")
