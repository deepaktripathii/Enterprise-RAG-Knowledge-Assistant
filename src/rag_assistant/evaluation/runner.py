"""Evaluation dataset loading and end-to-end runner (Feature 8).

The dataset is a JSONL file with one question per line:

```json
{"id": "q01", "question": "...", "gold_document": "Remote_Work_Policy.pdf",
 "gold_pages": [2], "expected_keywords": ["two days"], "reference_answer": "..."}
```

Each question is asked through the real ``RagService`` (retrieval + generation)
so the report measures the shipped pipeline, not an isolated retriever.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from rag_assistant.embeddings.base import Embedder
from rag_assistant.evaluation.metrics import (
    answer_relevance,
    citation_accuracy,
    faithfulness_score,
    keyword_coverage,
    retrieval_metrics,
)
from rag_assistant.logging_config import get_logger
from rag_assistant.rag.service import RagService

logger = get_logger(__name__)

DEFAULT_DATASET = Path(__file__).resolve().parents[3] / "data" / "evaluation" / "questions.jsonl"


@dataclass
class EvalCase:
    id: str
    question: str
    gold_document: str
    gold_pages: list[int] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    reference_answer: str = ""


def load_dataset(path: str | Path) -> list[EvalCase]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

    cases: list[EvalCase] = []
    for line_number, raw in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number} of {dataset_path}") from exc
        cases.append(
            EvalCase(
                id=str(payload.get("id", f"q{line_number}")),
                question=str(payload["question"]),
                gold_document=str(payload.get("gold_document", "")),
                gold_pages=[int(page) for page in payload.get("gold_pages", [])],
                expected_keywords=[str(word) for word in payload.get("expected_keywords", [])],
                reference_answer=str(payload.get("reference_answer", "")),
            )
        )

    if not cases:
        raise ValueError(f"Evaluation dataset is empty: {dataset_path}")
    return cases


def _mean(values: Sequence[float]) -> float:
    return round(float(np.mean(values)), 4) if values else 0.0


def run_evaluation(
    rag: RagService,
    *,
    dataset_path: str | Path = DEFAULT_DATASET,
    embedder: Embedder,
    top_k: int = 5,
    limit: int | None = None,
    progress: Any | None = None,
) -> dict[str, Any]:
    cases = load_dataset(dataset_path)
    if limit:
        cases = cases[:limit]

    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        if progress:
            progress(f"[{index}/{len(cases)}] {case.question[:70]}")
        answer = rag.ask(case.question, top_k=top_k)

        retrieval = retrieval_metrics(
            answer.retrieved,
            gold_document=case.gold_document,
            gold_pages=case.gold_pages,
            k=top_k,
        )
        sources = citation_accuracy(
            answer.citations,
            gold_document=case.gold_document,
            gold_pages=case.gold_pages,
        )
        row = {
            "id": case.id,
            "question": case.question,
            "gold_document": case.gold_document,
            "gold_pages": case.gold_pages,
            "answer": answer.answer,
            "citations": answer.citations,
            "grounded": answer.grounded,
            "fallback": answer.fallback,
            "warnings": answer.warnings,
            **retrieval,
            "answer_keyword_coverage": keyword_coverage(
                answer.answer, case.expected_keywords
            ),
            "citation_accuracy": sources,
            "faithfulness": faithfulness_score(
                answer.answer, answer.citations, answer.retrieved
            ),
            "answer_relevance": answer_relevance(case.question, answer.answer, embedder),
            "retrieval_latency_ms": round(answer.retrieval_latency_ms, 2),
            "llm_latency_ms": round(answer.llm_latency_ms, 2),
            "total_latency_ms": round(answer.total_latency_ms, 2),
            "model": answer.model,
        }
        rows.append(row)

    summary = {
        "retrieval_hit_rate": _mean([row["hit_at_k"] for row in rows]),
        "mrr": _mean([row["mrr"] for row in rows]),
        "context_precision": _mean([row["context_precision"] for row in rows]),
        "answer_keyword_coverage": _mean([row["answer_keyword_coverage"] for row in rows]),
        "citation_accuracy": _mean([row["citation_accuracy"] for row in rows]),
        "faithfulness": _mean([row["faithfulness"] for row in rows]),
        "answer_relevance": _mean([row["answer_relevance"] for row in rows]),
        "grounded_rate": _mean([1.0 if row["grounded"] else 0.0 for row in rows]),
        "fallback_rate": _mean([1.0 if row["fallback"] else 0.0 for row in rows]),
        "avg_retrieval_latency_ms": _mean([row["retrieval_latency_ms"] for row in rows]),
        "avg_llm_latency_ms": _mean([row["llm_latency_ms"] for row in rows]),
        "avg_total_latency_ms": _mean([row["total_latency_ms"] for row in rows]),
        "n_cases": len(rows),
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "top_k": top_k,
        "embedder": embedder.name,
        "summary": summary,
        "cases": rows,
    }


def save_report(report: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Evaluation report written to %s", destination)
    return destination


__all__ = ["DEFAULT_DATASET", "EvalCase", "load_dataset", "run_evaluation", "save_report"]
