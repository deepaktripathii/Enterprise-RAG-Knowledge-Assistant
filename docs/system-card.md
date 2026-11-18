# System card

## What it is

RAG Knowledge Assistant is a retrieval-augmented question answering service for an
organisation's internal documents. Employees upload PDFs, ask questions and
receive answers composed from retrieved passages, with document and page
citations.

## Intended use

- Answering policy and process questions (HR, IT, travel, support SLAs) where a
  source document exists and can be cited.
- Helping employees find the right section of a long document.
- Demonstrating an end-to-end RAG stack: ingestion, chunking, embeddings,
  vector search, grounding, citation validation, monitoring and evaluation.

## Out of scope

- Legal, medical or financial advice.
- Questions requiring synthesis across documents that were never indexed.
- Real-time data (prices, incidents, stock levels) that is not in the corpus.
- Multi-tenant access control: documents are visible to every caller of this
  instance.

## Data handling

- Uploaded files are stored under `data/uploads/<uuid>.pdf`; original names are
  kept in the database for display and citations.
- Text is extracted with PyMuPDF, split into page-scoped chunks and embedded
  locally. When `RAG_LLM_PROVIDER=openai`, the retrieved chunks and the
  question are sent to the configured model provider; with the extractive
  answerer nothing leaves the machine.
- Conversations, messages and query logs (question, answer preview, latencies,
  token counts) are stored in PostgreSQL/SQLite.
- Deleting a document removes its vectors and file. Conversation history is
  retained until deleted; query logs are operational records and should follow
  the organisation's retention policy.

## Safety properties

| Property | How it is enforced |
| --- | --- |
| No answers without sources | Relevance threshold + refusal path |
| No invented citations | Server-side marker validation against retrieved chunks only |
| No silent model swap | `/health` reports the embedder and LLM provider; dimension mismatches fail fast |
| Provider outage | One retry through the extractive answerer, flagged as `llm_fallback_used` |
| Upload abuse | PDF signature check, MIME and size limits, SHA-256 de-duplication |
| Prompt injection in PDFs | Partial: the system prompt forbids following instructions found in context; content is still untrusted and should be filtered for production |

## Known risks

1. **Prompt injection.** A malicious PDF can contain text like "ignore the rules
   and answer X". The extractive mode is immune (it copies sentences), but an LLM
   provider can be influenced. Add an injection classifier and content
   allow-listing before exposing to untrusted uploaders.
2. **Retrieval failure looks like an answer.** A related-but-wrong page can pass
   the relevance threshold. The evaluation set and citation review are the
   mitigations.
3. **Stale index.** Deleting a source system document does not delete it here;
   ingestion is manual or API driven today.
4. **Calibration.** Cosine scores are not probabilities; the 0.2 threshold is a
   starting point tuned on the sample corpus, not a universal value.

## Evaluation snapshot

12 questions, MiniLM embeddings, top-k 5, extractive answerer: retrieval hit rate
1.000, MRR 0.958, citation accuracy 0.917, faithfulness 1.000, answer relevance
0.735, fallback rate 0.000. Full methodology in
[rag-evaluation.md](rag-evaluation.md).

## Change control

- Any change to chunking, embedding model, prompt or threshold requires a fresh
  `python scripts/evaluate.py` run and a comparison against the previous report.
- New documents should be added with at least one evaluation question that has a
  known page-level answer.
