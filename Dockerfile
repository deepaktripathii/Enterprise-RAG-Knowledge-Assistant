# syntax=docker/dockerfile:1

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    SOURCEIQ_DATA_DIR=/app/data \
    SOURCEIQ_ARTIFACT_DIR=/app/artifacts \
    SOURCEIQ_QDRANT_PATH=/app/.qdrant \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# libgomp is required by torch on slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-embeddings.txt requirements-postgres.txt ./
RUN pip install --upgrade pip && pip install -r requirements-postgres.txt -r requirements-embeddings.txt

# Bake the embedding model into the image so containers start without network.
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY data/sample_documents ./data/sample_documents
COPY data/evaluation ./data/evaluation
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

RUN pip install --no-deps -e . && chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "sourceiq.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

EXPOSE 8000
