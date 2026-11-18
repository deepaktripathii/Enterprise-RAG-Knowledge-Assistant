"""RAG evaluation metrics (Feature 8).

Four questions are measured separately so a regression can be attributed:

* **Retrieval relevance** - did we fetch chunks from the right document/page?
* **Answer faithfulness** - is every answer sentence supported by the cited text?
* **Answer relevance** - does the answer actually address the question?
* **Source correctness** - do the citations point at the gold document/page?

Retrieval metrics use the gold document/page labels from the evaluation set.
Faithfulness and relevance are deterministic heuristics, so the suite runs in CI
without an LLM judge; an optional LLM grader can be layered on top later.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rag_assistant.embeddings.base import Embedder
from rag_assistant.llm.extractive import content_tokens, split_sentences
from rag_assistant.vectorstore.base import RetrievedChunk

_MARKER = re.compile(r"\[\d{1,2}\]")
_BOILERPLATE = re.compile(r"^(based on|sources?[:\s]|note[:\s])", re.IGNORECASE)


def _normalise_document(name: str) -> str:
    stem = Path(name).stem.lower()
    return re.sub(r"[^a-z0-9]+", " ", stem).strip()


def document_matches(candidate: str, gold: str) -> bool:
    candidate_norm = _normalise_document(candidate)
    gold_norm = _normalise_document(gold)
    if not candidate_norm or not gold_norm:
        return False
    return candidate_norm == gold_norm or gold_norm in candidate_norm or candidate_norm in gold_norm


def retrieval_metrics(
    chunks: Sequence[RetrievedChunk],
    *,
    gold_document: str,
    gold_pages: Sequence[int] | None = None,
    k: int = 5,
) -> dict[str, Any]:
    """Hit rate, MRR and context precision for one question."""
    pages = {int(page) for page in (gold_pages or [])}
    relevant: list[bool] = []
    for chunk in chunks:
        same_doc = document_matches(chunk.filename, gold_document)
        relevant.append(same_doc and (not pages or chunk.page in pages))

    hit = any(relevant)
    mrr = 0.0
    for rank, is_relevant in enumerate(relevant, start=1):
        if is_relevant:
            mrr = 1.0 / rank
            break

    return {
        "hit_at_k": 1.0 if hit else 0.0,
        "mrr": round(mrr, 4),
        "context_precision": round(sum(relevant) / len(chunks), 4) if chunks else 0.0,
        "retrieved_chunks": len(chunks),
        "relevant_chunks": int(sum(relevant)),
    }


def keyword_coverage(answer: str, keywords: Sequence[str]) -> float:
    if not keywords:
        return 1.0
    lowered = answer.lower()
    matched = sum(1 for keyword in keywords if keyword.lower() in lowered)
    return round(matched / len(keywords), 4)


def citation_accuracy(
    citations: Sequence[dict[str, Any]],
    *,
    gold_document: str,
    gold_pages: Sequence[int] | None = None,
) -> float:
    if not citations:
        return 0.0
    pages = {int(page) for page in (gold_pages or [])}
    correct = 0
    for citation in citations:
        if not document_matches(str(citation.get("filename", "")), gold_document):
            continue
        if pages and int(citation.get("page", -1)) not in pages:
            continue
        correct += 1
    return round(correct / len(citations), 4)


def faithfulness_score(
    answer: str,
    citations: Sequence[dict[str, Any]],
    chunks: Sequence[RetrievedChunk] | None = None,
) -> float:
    """Share of answer sentences whose content words appear in the cited text.

    ``chunks`` (the full retrieved passages) is preferred when available: the
    citation snippet shown in the UI is truncated, so scoring against snippets
    alone would under-count long but fully supported sentences.
    """
    if chunks:
        by_source = {
            (chunk.document_id, chunk.chunk_index): chunk.text for chunk in chunks
        }
        evidence = " ".join(
            by_source.get(
                (str(citation.get("document_id", "")), int(citation.get("chunk_index", -1))),
                str(citation.get("snippet", "")),
            )
            for citation in citations
        )
    else:
        evidence = " ".join(str(citation.get("snippet", "")) for citation in citations)

    evidence_tokens = content_tokens(evidence)
    if not evidence_tokens:
        return 0.0

    sentences = [
        sentence
        for sentence in split_sentences(_MARKER.sub("", answer))
        if not _BOILERPLATE.match(sentence.strip())
    ]
    if not sentences:
        return 1.0 if not answer.strip() else 0.0

    supported = 0
    for sentence in sentences:
        tokens = content_tokens(sentence)
        if not tokens:
            continue
        overlap = len(tokens & evidence_tokens) / len(tokens)
        if overlap >= 0.45:
            supported += 1
    return round(supported / len(sentences), 4)


def answer_relevance(question: str, answer: str, embedder: Embedder) -> float:
    if not answer.strip():
        return 0.0
    vectors = embedder.encode([question, answer])
    similarity = float(np.dot(vectors[0], vectors[1]))
    return round(max(0.0, min(1.0, similarity)), 4)


__all__ = [
    "answer_relevance",
    "citation_accuracy",
    "document_matches",
    "faithfulness_score",
    "keyword_coverage",
    "retrieval_metrics",
]
