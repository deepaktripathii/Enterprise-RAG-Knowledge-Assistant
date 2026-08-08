"""Ingest the sample PDF library into Qdrant.

Usage::

    python scripts/seed_documents.py
    python scripts/seed_documents.py --path data/sample_documents --backend hashing
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from sourceiq.config import get_settings
from sourceiq.services import build_services


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Index sample PDFs into the vector store.")
    parser.add_argument("--path", default=None, help="Directory containing PDF files.")
    parser.add_argument(
        "--backend",
        default=None,
        choices=["auto", "sentence-transformers", "hashing"],
        help="Override the embedding backend for this run.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop the Qdrant collection before indexing (needed when switching embedders).",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    directory = Path(args.path) if args.path else settings.sample_docs_dir
    pdfs = sorted(directory.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {directory}. Run `python scripts/make_sample_documents.py` first.")
        return 1

    services = build_services(settings, embedding_backend=args.backend)
    print(f"Embedder : {services.embedder.name} ({services.embedder.dimensions} dims)")
    print(f"Qdrant   : {services.vector_store.collection}")

    try:
        if args.reset:
            services.vector_store.reset()
            print("Collection reset.")
        for pdf in pdfs:
            document = services.document_service.ingest_file(pdf)
            status = document["status"]
            detail = (
                f"{document['page_count']} pages -> {document['chunk_count']} chunks"
                if status == "completed"
                else document.get("error_message") or ""
            )
            print(f"  {pdf.name:<36} {status:<10} {detail}")
    finally:
        print(f"Vectors in collection: {services.vector_store.count()}")
        services.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
