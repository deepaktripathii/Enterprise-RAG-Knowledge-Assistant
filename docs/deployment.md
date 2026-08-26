# Deployment notes

## Configuration

All settings are environment variables prefixed with `SOURCEIQ_` (see
`.env.example`). The ones that change behaviour the most:

| Variable | Default | Notes |
| --- | --- | --- |
| `SOURCEIQ_EMBEDDING_BACKEND` | `auto` | `sentence-transformers`, `hashing`, or `auto` |
| `SOURCEIQ_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Any Sentence Transformers model |
| `SOURCEIQ_EMBEDDING_BATCH_SIZE` | `32` | Lower it if ingestion runs out of memory |
| `SOURCEIQ_CHUNK_SIZE` / `_OVERLAP` | `1000` / `150` | Characters; re-index after changing |
| `SOURCEIQ_RETRIEVAL_TOP_K` | `5` | Prompt size vs recall |
| `SOURCEIQ_MIN_RELEVANCE_SCORE` | `0.2` | Raise it for stricter refusals |
| `SOURCEIQ_LLM_PROVIDER` | `auto` | `openai`, `extractive`, or `auto` |
| `SOURCEIQ_LLM_API_KEY` | unset | Required for `openai` |
| `SOURCEIQ_QDRANT_URL` | unset | Unset = embedded Qdrant on disk |
| `SOURCEIQ_DATABASE_URL` | SQLite in `data/` | Use `postgresql+psycopg://...` for Postgres |
| `SOURCEIQ_MAX_UPLOAD_MB` | `20` | Enforced before touching disk |

## Modes

### Laptop / demo

SQLite + embedded Qdrant + hashing or MiniLM embeddings. No services required:

```bash
pip install -r requirements.txt && pip install -e .
python scripts/make_sample_documents.py
python scripts/seed_documents.py --backend hashing
uvicorn sourceiq.api.main:app --reload
```

### Docker Compose

`docker compose up --build` starts Qdrant, PostgreSQL, the API and the Next.js
frontend. The API image bakes the embedding model so containers do not need
network access at start, and the entrypoint seeds the sample corpus on first
boot (`SOURCEIQ_SEED_DEMO=0` disables that).

### Production checklist

- Set `SOURCEIQ_DATABASE_URL` to PostgreSQL and run the API behind HTTPS.
- Point `SOURCEIQ_QDRANT_URL` at a managed Qdrant cluster; set
  `SOURCEIQ_QDRANT_API_KEY` when required.
- Put uploads on object storage if you run more than one API replica; the local
  filesystem is shared only inside one container.
- Add authentication/authorisation in front of the API and per-document ACLs in
  retrieval filters before exposing internal documents.
- Run `python scripts/evaluate.py` as a release gate and compare against the
  numbers in `docs/rag-evaluation.md`.
- Monitor `/monitoring/summary`: a rising fallback rate usually means a content
  gap or a drifting query mix; a rising `avg_top_score` with stable traffic is
  often a duplicate ingestion.

## Switching embedding models

Qdrant collections have a fixed vector size. Changing `SOURCEIQ_EMBEDDING_MODEL`
(or switching between hashing and Sentence Transformers) requires a fresh
collection:

```bash
python scripts/reset_store.py --yes
python scripts/seed_documents.py --backend sentence-transformers
```

The API fails fast with a dimension-mismatch message instead of silently
comparing incompatible vectors.

## Backups

- PostgreSQL: standard `pg_dump` of the `sourceiq` database (documents,
  conversations, query logs).
- Qdrant: snapshot the collection or the `qdrant_data` volume.
- Uploads: back up `data/uploads/` together with the database. A database row
  without its PDF cannot be re-indexed, though existing vectors keep working.

## Scaling notes

- Ingestion is CPU bound (PDF parsing + embeddings). Move it to a worker queue
  (Celery/RQ/Arq) for more than a few uploads at a time.
- Retrieval is stateless: add API replicas behind a load balancer.
- Reuse one embedder per process; loading MiniLM twice doubles memory.
- For latency, cache `(normalised question, top_k, document_ids)` → retrieval
  results, and stream LLM tokens over SSE.
