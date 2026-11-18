"""API contract: uploads, chat, history, monitoring and evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_assistant.api.main import create_app


@pytest.fixture()
def client(tmp_settings):
    app = create_app(tmp_settings)
    with TestClient(app) as test_client:
        yield test_client


def _upload(client: TestClient, path: Path):
    with path.open("rb") as handle:
        return client.post(
            "/documents/upload",
            files={"file": (path.name, handle, "application/pdf")},
        )


def test_health_reports_components(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["components"]["database"] == "ok"
    assert body["components"]["vector_store"] == "ok"
    assert body["embedder"].startswith("hashing")
    assert body["vectors"] == 0


def test_upload_indexes_document(client, pdf_library):
    response = _upload(client, pdf_library["remote"])

    assert response.status_code == 202
    created = response.json()
    assert created["status"] in {"pending", "completed"}
    assert "Location" in response.headers

    document = client.get(f"/documents/{created['id']}").json()
    assert document["status"] == "completed"
    assert document["page_count"] == 2
    assert document["chunk_count"] >= 2

    listed = client.get("/documents").json()
    assert len(listed) == 1


def test_duplicate_upload_is_idempotent(client, pdf_library):
    first = _upload(client, pdf_library["remote"])
    assert first.status_code == 202

    second = _upload(client, pdf_library["remote"])

    assert second.status_code == 200
    assert second.headers["X-Duplicate"] == "true"
    assert second.json()["id"] == first.json()["id"]


def test_upload_rejects_invalid_file(client):
    response = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", b"not a pdf", "text/plain")},
    )

    assert response.status_code == 400


def test_chat_returns_citations_and_persists_history(client, pdf_library):
    _upload(client, pdf_library["remote"])

    response = client.post(
        "/chat", json={"question": "How many days per week can I work remotely?"}
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["fallback"] is False
    assert body["grounded"] is True
    assert "two days" in body["answer"].lower()
    assert body["citations"][0]["filename"] == "Remote_Work_Policy.pdf"
    assert body["citations"][0]["page"] == 1
    assert body["usage"]["total_tokens"] > 0

    conversation = client.get(f"/conversations/{body['conversation_id']}").json()
    assert len(conversation["messages"]) == 2
    assert conversation["messages"][0]["role"] == "user"

    listed = client.get("/conversations").json()
    assert len(listed) == 1


def test_chat_rejects_unknown_conversation(client):
    response = client.post(
        "/chat", json={"question": "hello there", "conversation_id": "missing"}
    )

    assert response.status_code == 404


def test_chat_validates_payload(client):
    assert client.post("/chat", json={"question": "a"}).status_code == 422
    assert (
        client.post("/chat", json={"question": "valid question", "extra": 1}).status_code
        == 422
    )


def test_monitoring_summary_and_queries(client, pdf_library):
    _upload(client, pdf_library["remote"])
    client.post("/chat", json={"question": "How many days per week can I work remotely?"})

    summary = client.get("/monitoring/summary").json()
    assert summary["total_queries"] == 1
    assert summary["documents"]["total"] == 1
    assert summary["vectors"] > 0
    assert summary["tokens"]["total"] > 0

    queries = client.get("/monitoring/queries").json()
    assert len(queries) == 1
    assert queries[0]["fallback"] is False


def test_evaluation_endpoints(client, pdf_library, tmp_settings):
    _upload(client, pdf_library["remote"])

    assert client.get("/evaluation/latest").status_code == 404

    dataset = tmp_settings.evaluation_dir / "questions.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "How many days per week can I work remotely?",
                "gold_document": "Remote_Work_Policy.pdf",
                "gold_pages": [1],
                "expected_keywords": ["two days"],
            }
        ),
        encoding="utf-8",
    )

    report = client.post("/evaluation/run", params={"limit": 1, "top_k": 3})

    assert report.status_code == 200, report.text
    body = report.json()
    assert body["summary"]["n_cases"] == 1
    assert body["cases"][0]["hit_at_k"] == 1.0

    latest = client.get("/evaluation/latest")
    assert latest.status_code == 200
    assert latest.json()["summary"]["n_cases"] == 1


def test_delete_document_removes_vectors(client, pdf_library):
    created = _upload(client, pdf_library["remote"]).json()

    response = client.delete(f"/documents/{created['id']}")

    assert response.status_code == 204
    assert client.get(f"/documents/{created['id']}").status_code == 404
    assert client.get("/health").json()["vectors"] == 0


def test_delete_unknown_document(client):
    assert client.delete("/documents/does-not-exist").status_code == 404
