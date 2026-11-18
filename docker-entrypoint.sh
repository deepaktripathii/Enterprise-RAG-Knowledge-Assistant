#!/bin/sh
# Prepare the container: directories, demo corpus and an initial index.
set -eu

: "${RAG_DATA_DIR:=/app/data}"
: "${RAG_ARTIFACT_DIR:=/app/artifacts}"
: "${RAG_SEED_DEMO:=1}"
export RAG_DATA_DIR RAG_ARTIFACT_DIR

mkdir -p "$RAG_DATA_DIR/uploads" "$RAG_ARTIFACT_DIR" "$RAG_ARTIFACT_DIR/evaluation"

if [ "$RAG_SEED_DEMO" = "1" ] && [ -f /app/scripts/make_sample_documents.py ]; then
    echo "[entrypoint] ensuring sample policy documents exist"
    python scripts/make_sample_documents.py || true
    echo "[entrypoint] indexing sample documents (idempotent by content hash)"
    python scripts/seed_documents.py || echo "[entrypoint] indexing skipped; upload documents via the UI"
fi

exec "$@"
