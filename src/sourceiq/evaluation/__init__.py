"""Retrieval and answer evaluation."""

from sourceiq.evaluation.metrics import (
    answer_relevance,
    citation_accuracy,
    document_matches,
    faithfulness_score,
    keyword_coverage,
    retrieval_metrics,
)
from sourceiq.evaluation.runner import (
    DEFAULT_DATASET,
    EvalCase,
    load_dataset,
    run_evaluation,
    save_report,
)

__all__ = [
    "DEFAULT_DATASET",
    "EvalCase",
    "answer_relevance",
    "citation_accuracy",
    "document_matches",
    "faithfulness_score",
    "keyword_coverage",
    "load_dataset",
    "retrieval_metrics",
    "run_evaluation",
    "save_report",
]
