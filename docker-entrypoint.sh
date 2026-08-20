#!/bin/sh
# Prepare the container: directories, demo corpus and an initial index.
set -eu

: "${SOURCEIQ_DATA_DIR:=/app/data}"
: "${SOURCEIQ_ARTIFACT_DIR:=/app/artifacts}"
: "${SOURCEIQ_SEED_DEMO:=1}"
export SOURCEIQ_DATA_DIR SOURCEIQ_ARTIFACT_DIR

mkdir -p "$SOURCEIQ_DATA_DIR/uploads" "$SOURCEIQ_ARTIFACT_DIR" "$SOURCEIQ_ARTIFACT_DIR/evaluation"

if [ "$SOURCEIQ_SEED_DEMO" = "1" ] && [ -f /app/scripts/make_sample_documents.py ]; then
    echo "[entrypoint] ensuring sample policy documents exist"
    python scripts/make_sample_documents.py || true
    echo "[entrypoint] indexing sample documents (idempotent by content hash)"
    python scripts/seed_documents.py || echo "[entrypoint] indexing skipped; upload documents via the UI"
fi

exec "$@"
