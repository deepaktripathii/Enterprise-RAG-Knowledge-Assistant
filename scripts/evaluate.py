"""Run the fixed RAG evaluation dataset.

Usage::

    python scripts/evaluate.py
    python scripts/evaluate.py --limit 5 --top-k 4 --backend hashing
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from sourceiq.config import get_settings
from sourceiq.evaluation.runner import run_evaluation, save_report
from sourceiq.services import build_services


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answer quality.")
    parser.add_argument("--dataset", default=None, help="JSONL evaluation set.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--backend", default=None, choices=["auto", "sentence-transformers", "hashing"])
    parser.add_argument("--json", action="store_true", help="Print the full report as JSON.")
    args = parser.parse_args(argv)

    settings = get_settings()
    dataset = Path(args.dataset) if args.dataset else settings.evaluation_dir / "questions.jsonl"

    services = build_services(settings, embedding_backend=args.backend)
    try:
        if services.vector_store.count() == 0:
            print("Vector store is empty. Run `python scripts/seed_documents.py` first.")
            return 1

        print(f"Embedder : {services.embedder.name}")
        print(f"Dataset  : {dataset}")
        report = run_evaluation(
            services.rag_service,
            dataset_path=dataset,
            embedder=services.embedder,
            top_k=args.top_k,
            limit=args.limit,
            progress=print,
        )
        destination = save_report(
            report, settings.artifact_dir / "evaluation" / "latest.json"
        )
    finally:
        services.close()

    summary = report["summary"]
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("\nSummary")
        print(f"  cases                : {summary['n_cases']}")
        print(f"  retrieval hit rate   : {summary['retrieval_hit_rate']:.3f}")
        print(f"  MRR                  : {summary['mrr']:.3f}")
        print(f"  context precision    : {summary['context_precision']:.3f}")
        print(f"  answer keywords      : {summary['answer_keyword_coverage']:.3f}")
        print(f"  citation accuracy    : {summary['citation_accuracy']:.3f}")
        print(f"  faithfulness         : {summary['faithfulness']:.3f}")
        print(f"  answer relevance     : {summary['answer_relevance']:.3f}")
        print(f"  grounded rate        : {summary['grounded_rate']:.3f}")
        print(f"  fallback rate        : {summary['fallback_rate']:.3f}")
        print(f"  avg total latency    : {summary['avg_total_latency_ms']:.0f} ms")
        print(f"\nReport written to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
