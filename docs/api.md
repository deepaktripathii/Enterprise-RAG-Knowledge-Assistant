# API reference

Base URL `http://localhost:8000` · interactive docs at `/docs`.

## `POST /documents/upload`

Multipart upload. Only `application/pdf`, maximum 20 MB (configurable).
Returns `202` with the document row while indexing runs in the background.
Identical content (same SHA-256) that already finished indexing returns `200`
with header `X-Duplicate: true`.

```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@data/sample_documents/Remote_Work_Policy.pdf"
```

```json
{
  "id": "0f0b6e9e-8a3f-4a3f-9d1e-6f0c1f4c1d2a",
  "filename": "Remote_Work_Policy.pdf",
  "original_filename": "Remote_Work_Policy.pdf",
  "content_type": "application/pdf",
  "size_bytes": 8493,
  "sha256": "8b1d...",
  "page_count": 0,
  "chunk_count": 0,
  "status": "pending",
  "error_message": null,
  "created_at": "2026-08-04T09:12:31.412Z",
  "updated_at": "2026-08-04T09:12:31.412Z",
  "processed_at": null
}
```

## `GET /documents?status=completed&limit=100`

List documents, newest first. Processing states: `pending`, `processing`,
`completed`, `failed`. Failed rows include `error_message`.

## `GET /documents/{id}`

One document. Use it to poll indexing progress.

## `POST /documents/{id}/process`

Re-run extraction, chunking and embedding synchronously. Useful for retrying a
failed document or re-indexing after a config change.

## `DELETE /documents/{id}`

`204 No Content`. Removes the uploaded file, every vector for that document and
the database row.

## `POST /chat`

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How many days per week can I work remotely?",
    "conversation_id": null,
    "top_k": 5
  }'
```

```json
{
  "conversation_id": "4ec5d3c0-2a24-4a1d-9ac0-1a2a5b1c5e11",
  "message_id": 2,
  "question": "How many days per week can I work remotely?",
  "answer": "Based on the retrieved documents:\n- Remote-eligible employees may work remotely for two days per week. [1]\n- The remaining three days are spent in the office; Tuesday and Thursday are mandatory office days. [1]",
  "citations": [
    {
      "index": 1,
      "document_id": "0f0b6e9e-...",
      "filename": "Remote_Work_Policy.pdf",
      "page": 2,
      "chunk_index": 1,
      "score": 0.6712,
      "snippet": "2. Hybrid schedule and approval. Remote-eligible employees may work remotely for two days per week…"
    }
  ],
  "sources": [
    {
      "document_id": "0f0b6e9e-...",
      "filename": "Remote_Work_Policy.pdf",
      "page": 2,
      "chunk_index": 1,
      "score": 0.6712
    }
  ],
  "grounded": true,
  "fallback": false,
  "warnings": [],
  "model": "extractive-grounded-v1",
  "provider": "extractive",
  "retrieval_latency_ms": 12.4,
  "llm_latency_ms": 0.6,
  "total_latency_ms": 13.0,
  "usage": { "prompt_tokens": 184, "completion_tokens": 48, "total_tokens": 232 },
  "created_at": "2026-08-04T09:15:02.118Z"
}
```

Fields worth knowing:

| Field | Meaning |
| --- | --- |
| `grounded` | The answer cites at least one retrieved source |
| `fallback` | Retrieval was empty/irrelevant, or the model refused |
| `warnings` | `below_relevance_threshold`, `no_retrieval_results`, `answer_missing_citations`, `invalid_citation_markers_removed`, `llm_fallback_used`, `llm_refused` |
| `sources` | All retrieved chunks (without text) for debugging ranking |
| `usage` | Provider token counts, or an estimate for the extractive answerer |

Optional request fields: `conversation_id` to continue a thread, `top_k`
(1–20), and `document_ids` to restrict retrieval.

Errors: `404` unknown conversation, `422` invalid payload, `502` generation
failure with no fallback available.

## `GET /conversations` · `GET /conversations/{id}`

Conversation list and full transcript (user/assistant messages with the
citations stored at answer time).

## `DELETE /conversations/{id}`

`204 No Content`. Deletes messages and the conversation.

## `GET /monitoring/summary`

```json
{
  "documents": { "pending": 0, "processing": 0, "completed": 5, "failed": 0, "total": 5 },
  "vectors": 23,
  "conversations": 12,
  "total_queries": 18,
  "window_size": 18,
  "avg_retrieval_latency_ms": 8.1,
  "avg_llm_latency_ms": 0.7,
  "avg_total_latency_ms": 8.8,
  "p95_total_latency_ms": 12.0,
  "avg_top_score": 0.62,
  "fallback_rate": 0.06,
  "avg_citations": 1.6,
  "tokens": { "prompt": 4210, "completion": 940, "total": 5150 },
  "models": { "extractive-grounded-v1": 18 },
  "daily": [{ "date": "2026-08-04", "queries": 18, "avg_latency_ms": 8.8 }],
  "last_query_at": "2026-08-04T09:20:11+00:00",
  "queries_last_24h": 18
}
```

## `GET /monitoring/queries?limit=50`

Recent question log: question, answer preview, retrieved chunk count, top score,
citation count, fallback flag, per-stage latency, token usage and model.

## `GET /evaluation/latest`

Returns the most recent retrieval evaluation report (`404` before the first
run). See [rag-evaluation.md](rag-evaluation.md) for metric definitions.

## `POST /evaluation/run?top_k=5&limit=12`

Runs the fixed question set through the live pipeline and persists the report to
`artifacts/evaluation/latest.json`. Creates conversations in the database, so run
it on a demo or staging instance.

## `GET /health`

```json
{
  "status": "ok",
  "version": "1.0.0",
  "embedder": "sentence-transformers:sentence-transformers/all-MiniLM-L6-v2",
  "embedding_dimensions": 384,
  "llm_provider": "extractive",
  "llm_model": "extractive-grounded-v1",
  "collection": "sourceiq_chunks",
  "vectors": 23,
  "documents": { "completed": 5, "total": 5 },
  "components": {
    "database": "ok",
    "vector_store": "ok",
    "embedding": "ok",
    "llm": "ok"
  },
  "timestamp": "2026-08-04T09:21:00+00:00"
}
```
