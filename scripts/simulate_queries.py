"""Generate demo chat traffic so monitoring has data to show.

Usage::

    python scripts/simulate_queries.py --n 25
"""

from __future__ import annotations

import argparse
import random

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from rag_assistant.config import get_settings
from rag_assistant.evaluation.runner import load_dataset
from rag_assistant.services import build_services


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask sample questions through the RAG service.")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--backend", default=None, choices=["auto", "sentence-transformers", "hashing"])
    args = parser.parse_args(argv)

    settings = get_settings()
    dataset = settings.evaluation_dir / "questions.jsonl"
    cases = load_dataset(dataset)

    services = build_services(settings, embedding_backend=args.backend)
    random.seed(args.seed)
    latencies: list[float] = []
    try:
        if services.vector_store.count() == 0:
            print("Vector store is empty. Run `python scripts/seed_documents.py` first.")
            return 1

        for index in range(args.n):
            case = cases[index % len(cases)]
            answer = services.rag_service.ask(case.question, top_k=settings.retrieval_top_k)
            latencies.append(answer.total_latency_ms)
            flag = "fallback" if answer.fallback else f"{len(answer.citations)} cites"
            print(
                f"[{index + 1:>3}/{args.n}] {answer.total_latency_ms:7.0f} ms | "
                f"{flag:<10} | {case.question[:60]}"
            )
    finally:
        services.close()

    if latencies:
        average = sum(latencies) / len(latencies)
        print(f"\nAvg latency: {average:.0f} ms over {len(latencies)} queries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
