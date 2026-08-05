"""Evaluation metrics and the end-to-end runner."""

from __future__ import annotations

import json

import pytest

from sourceiq.evaluation.metrics import (
    citation_accuracy,
    document_matches,
    faithfulness_score,
    keyword_coverage,
    retrieval_metrics,
)
from sourceiq.evaluation.runner import run_evaluation
from sourceiq.vectorstore.base import RetrievedChunk


def _chunk(filename: str, page: int, text: str = "text", score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(
        document_id=filename,
        filename=filename,
        chunk_index=0,
        page=page,
        text=text,
        score=score,
    )


def test_document_matches_ignores_case_and_extensions():
    assert document_matches("remote work policy.pdf", "Remote_Work_Policy.pdf")
    assert not document_matches("IT_Security_Policy.pdf", "Remote_Work_Policy.pdf")


def test_retrieval_metrics_hit_and_mrr():
    chunks = [_chunk("other.pdf", 1), _chunk("Remote_Work_Policy.pdf", 2)]

    metrics = retrieval_metrics(
        chunks, gold_document="Remote_Work_Policy.pdf", gold_pages=[2], k=2
    )

    assert metrics["hit_at_k"] == 1.0
    assert metrics["mrr"] == pytest.approx(0.5)
    assert metrics["context_precision"] == pytest.approx(0.5)


def test_retrieval_metrics_miss():
    metrics = retrieval_metrics(
        [_chunk("other.pdf", 1)], gold_document="Remote_Work_Policy.pdf", gold_pages=[2]
    )

    assert metrics["hit_at_k"] == 0.0
    assert metrics["mrr"] == 0.0


def test_keyword_coverage():
    assert keyword_coverage("Employees get 24 days of leave.", ["24", "leave"]) == 1.0
    assert keyword_coverage("Employees get leave.", ["24"]) == 0.0
    assert keyword_coverage("anything", []) == 1.0


def test_citation_accuracy_counts_matching_documents():
    citations = [
        {"filename": "Employee_Handbook.pdf", "page": 3},
        {"filename": "IT_Security_Policy.pdf", "page": 2},
    ]

    assert citation_accuracy(
        citations, gold_document="Employee_Handbook.pdf", gold_pages=[3]
    ) == pytest.approx(0.5)
    assert citation_accuracy([], gold_document="x.pdf") == 0.0


def test_faithfulness_rewards_supported_sentences():
    snippet = "Employees may work remotely for two days per week."
    citations = [{"filename": "remote.pdf", "page": 2, "snippet": snippet}]

    supported = faithfulness_score("- Employees may work remotely for two days per week. [1]", citations)
    unsupported = faithfulness_score(
        "- Employees receive a company car and free lunch every day. [1]", citations
    )

    assert supported == 1.0
    assert unsupported < 0.5


def test_run_evaluation_end_to_end(services, ingested, tmp_settings):
    dataset = tmp_settings.evaluation_dir / "questions.jsonl"
    records = [
        {
            "id": "q1",
            "question": "How many days per week can I work remotely?",
            "gold_document": "Remote_Work_Policy.pdf",
            "gold_pages": [1],
            "expected_keywords": ["two days"],
        },
        {
            "id": "q2",
            "question": "What is the minimum password length?",
            "gold_document": "IT_Security_Policy.pdf",
            "gold_pages": [1],
            "expected_keywords": ["14"],
        },
    ]
    dataset.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    report = run_evaluation(
        services.rag_service,
        dataset_path=dataset,
        embedder=services.embedder,
        top_k=4,
    )

    assert report["summary"]["n_cases"] == 2
    assert report["summary"]["retrieval_hit_rate"] >= 0.5
    assert report["summary"]["fallback_rate"] <= 0.5
    assert len(report["cases"]) == 2
    assert {"faithfulness", "citation_accuracy", "answer_relevance"} <= set(report["cases"][0])
