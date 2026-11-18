# Enterprise RAG Knowledge Assistant

Ask questions across internal PDFs and get answers grounded in retrieved
passages, with document and page citations, conversation history and monitoring.

```
PDF upload -> extraction -> page-aware chunking -> embeddings -> Qdrant
  -> question embedding -> top-k retrieval -> relevance gate
  -> grounded LLM answer (or extractive fallback) -> citation validation
  -> answer + sources -> PostgreSQL/SQLite history -> monitoring/evaluation
```

## Evaluation

12 questions over 5 sample policy PDFs (23 page-level chunks), top-k 5,
`all-MiniLM-L6-v2` embeddings, extractive answerer (no API key):

| Metric | Score |
| --- | --- |
| Retrieval hit rate | 1.000 |
| MRR | 0.958 |
| Context precision | 0.200 |
| Answer keyword coverage | 0.917 |
| Citation accuracy | 0.917 |
| Faithfulness | 1.000 |
| Answer relevance | 0.735 |
| Fallback rate | 0.000 |
| Average latency | ~9 ms |

Context precision of 0.20 is expected: one gold page among five retrieved.
Methodology and limits: [docs/rag-evaluation.md](docs/rag-evaluation.md).

## Features

1. Document management: PDF upload, multi-file, list, metadata, delete and a
   `pending -> processing -> completed/failed` lifecycle with content-hash
   de-duplication.
2. Processing and chunking: PyMuPDF page-wise extraction, cleaning,
   sentence-aware overlapping chunks with page numbers preserved.
3. Embeddings: Sentence Transformers with a deterministic hashing fallback for
   offline and CI use.
4. Vector store: Qdrant collection with upsert, top-k cosine search, document
   filters and bulk delete.
5. RAG answering: query embedding, retrieval, numbered context and grounded
   generation.
6. Citations and hallucination control: relevance gate, citation-first prompt,
   server-side marker validation and a refusal path.
7. Conversation history: persisted messages, follow-up query expansion,
   transcript and delete endpoints.
8. FastAPI, evaluation and monitoring: REST API, query log, latency and token
   usage, fixed evaluation dataset and runner.

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .

# optional but recommended (Sentence Transformers + torch CPU)
pip install -r requirements-embeddings.txt

python scripts/make_sample_documents.py
python scripts/seed_documents.py
uvicorn rag_assistant.api.main:app --reload
```

Dashboard:

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

Without an API key the extractive answerer is used. For fluent answers set
`RAG_LLM_API_KEY` (any OpenAI-compatible endpoint) in `.env`.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/documents/upload` | Upload and index a PDF |
| GET | `/documents` | List with processing status |
| DELETE | `/documents/{id}` | Remove file, vectors and metadata |
| POST | `/chat` | Grounded answer with citations and history |
| GET | `/conversations/{id}` | Conversation transcript |
| GET | `/monitoring/summary` | Volume, latency, tokens, fallbacks |
| GET | `/evaluation/latest` | Last retrieval evaluation |
| GET | `/health` | Component health |

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days per week can I work remotely?"}'
```

```json
{
  "answer": "Remote-eligible employees may work remotely for two days per week. [1]",
  "citations": [
    {"filename": "Remote_Work_Policy.pdf", "page": 2, "score": 0.67}
  ],
  "grounded": true,
  "fallback": false
}
```

## Project structure

```
src/rag_assistant/
  config.py, services.py
  db/            documents, conversations, message and query log
  documents/     extraction, chunking, storage, lifecycle
  embeddings/    sentence-transformers and hashing backends
  vectorstore/   Qdrant wrapper
  llm/           prompts, OpenAI-compatible client, extractive answerer
  rag/           retriever, generator, citations, orchestration
  evaluation/    metrics and dataset runner
  api/           FastAPI app and schemas
frontend/        Next.js 14 UI
scripts/         sample documents, seed, evaluate, simulate
tests/           65 offline tests
docs/            architecture, API, evaluation, deployment, interview notes
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check src tests scripts
```

## Docker

```bash
docker compose up --build
# Frontend  -> http://localhost:3000
# API       -> http://localhost:8000
# Qdrant    -> http://localhost:6333/dashboard
```
