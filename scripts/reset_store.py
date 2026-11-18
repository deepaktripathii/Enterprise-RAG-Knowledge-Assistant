"""Delete the local Qdrant collection and uploads (dev helper).

Switching embedding models changes the vector dimensions, and Qdrant cannot mix
dimensions in one collection. Use this script before reseeding with a different
model:

    python scripts/reset_store.py --yes
"""

from __future__ import annotations

import argparse
import shutil

from qdrant_client import QdrantClient

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from rag_assistant.config import get_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reset the RAG Knowledge Assistant vector store.")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    parser.add_argument("--keep-uploads", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    if not args.yes:
        answer = input(
            f"This deletes collection '{settings.qdrant_collection}'"
            f"{'' if args.keep_uploads else ' and every uploaded PDF'}. Continue? [y/N] "
        )
        if answer.strip().lower() not in {"y", "yes"}:
            print("Aborted.")
            return 1

    if settings.qdrant_url:
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=10)
        try:
            if client.collection_exists(settings.qdrant_collection):
                client.delete_collection(settings.qdrant_collection)
                print(f"Deleted remote collection '{settings.qdrant_collection}'.")
        finally:
            client.close()
    elif settings.qdrant_path.exists():
        shutil.rmtree(settings.qdrant_path)
        print(f"Removed local Qdrant storage at {settings.qdrant_path}.")

    if not args.keep_uploads and settings.upload_dir.exists():
        shutil.rmtree(settings.upload_dir)
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        print(f"Cleared uploads in {settings.upload_dir}.")

    print("Vector store reset. Re-run `python scripts/seed_documents.py`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
