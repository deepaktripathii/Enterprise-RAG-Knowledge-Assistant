# Interview notes

## Document processing & chunking

**What is chunking?**
Splitting a long document into smaller passages before embedding. The embedding
model has a limited input window, the vector store compares one vector per
passage, and the LLM can only read a bounded amount of context. Chunking is the
unit of retrieval: too large and the match is diluted, too small and the answer
is cut in half.

**Why overlap?**
Sentences that define a policy often straddle the cut ("…requires prior approval
from HR and the" / "security team"). Overlapping windows keep the complete idea
retrievable from at least one chunk, at the cost of storing some text twice.
Here the default is 1,000 characters with 150 characters of overlap.

**How do you choose chunk size?**
Start from the embedding model's effective window (MiniLM handles ~256 tokens
well), then tune on the evaluation set. Bigger chunks raise recall of context but
lower context precision and increase prompt cost; smaller chunks do the
opposite. Page-scoped chunking was chosen so the citation always maps to one
page, and the sample PDFs are short enough that one page ≈ one chunk.

**Why preserve metadata?**
A vector alone cannot produce a citation. Each Qdrant point carries
`document_id`, `filename`, `page`, `chunk_index` and the original text, so the
API can say "Remote_Work_Policy.pdf, page 2" and show the exact snippet. Metadata
also powers filters (search only one document) and deletion (remove all chunks
for a document id).

## Embeddings & vector search

**What are embeddings?**
Dense numeric representations of text where geometric closeness encodes semantic
similarity. `all-MiniLM-L6-v2` maps a passage to 384 floats. The pipeline
L2-normalises them so cosine similarity is a simple dot product.

**How do embeddings work?**
A transformer encoder is trained with contrastive objectives so that paraphrases
land close together and unrelated text lands far apart. At inference the model
returns one vector per input; no labels are needed, which is why a pre-trained
model can be used on company documents immediately.

**What is semantic similarity?**
Two passages are similar if they mean the same thing, even with different words
("time off" vs "annual leave"). Cosine similarity measures the angle between
their vectors: 1.0 identical, 0 orthogonal, negative opposite.

**Keyword search vs vector search**
Keyword search (BM25) matches exact tokens, which is excellent for IDs,
acronyms and error codes but fails on paraphrases. Vector search matches meaning
but can miss a rare identifier. Production RAG often runs both and merges the
results. This project ships dense retrieval and exposes the retrieved scores so a
hybrid layer can be added without changing the API.

## Qdrant

**What is a vector database?**
A store built for high-dimensional nearest-neighbour search plus payload
metadata. It maintains an index (HNSW by default) so a query does not scan every
vector, and it supports filters such as `document_id in [...]`.

**Why Qdrant and not PostgreSQL?**
PostgreSQL can store vectors (pgvector) and is the right home for documents,
conversations and audit logs. Qdrant adds purpose-built ANN indexing, payload
filtering, collection snapshots and horizontal scaling with far less tuning.
Both are used here: PostgreSQL/SQLite for relational state, Qdrant for vectors.

**What is cosine similarity?**
`dot(a, b) / (||a|| ||b||)` - the cosine of the angle between two vectors. With
normalised vectors it is just the dot product, bounded to [-1, 1]. Qdrant's
`COSINE` distance ranks higher as closer.

**What is top-K retrieval?**
Return the K nearest chunks. K trades recall against prompt size and latency.
The default is 5, configurable per request (`top_k`) and globally
(`RAG_RETRIEVAL_TOP_K`).

## RAG

**What is RAG?**
Retrieval-Augmented Generation: retrieve relevant passages from an external
knowledge base, put them in the prompt, and ask the LLM to answer only from that
context. It grounds answers in private documents without retraining the model.

**Why is RAG needed?**
An LLM's parameters contain public training data, not this company's policies,
and they go stale. RAG keeps knowledge in a store you control, updates instantly
when a PDF changes, and provides citations. It is also cheaper and safer than
fine-tuning for factual, frequently changing content.

**What happens during retrieval?**
The question is embedded with the same model used for ingestion, Qdrant compares
it with every stored vector and returns the top-K chunks with scores. Optional
filters restrict the search to specific documents. In this project a follow-up
question can be merged with the previous turn before embedding, so "and outside
India?" still retrieves the remote-work policy.

**RAG vs fine-tuning**
Fine-tuning changes behaviour, style and format; RAG changes knowledge. Policies
change monthly - re-indexing a PDF takes seconds, retraining does not. Fine-tune
when you need a consistent tone or a task format, use RAG when the answer must be
traceable to a document.

**How does the LLM use the retrieved context?**
Context is numbered (`[1] Source: file, page n`) and the system prompt forbids
outside knowledge and requires `[n]` markers. The model writes the answer, the
server validates every marker against the retrieved set, and the UI renders each
number as an expandable source card.

## Citations & hallucination control

**How do you reduce hallucinations?**
Five layers: (1) a relevance threshold that triggers refusal, (2) a system
prompt that forbids outside knowledge, (3) numbered context with page metadata,
(4) server-side validation that strips invented citation markers, and (5) an
evaluation set that measures faithfulness. The extractive answerer is a sixth,
ultra-conservative mode that copies sentences verbatim.

**Why are citations important?**
They make the answer verifiable. An employee can open page 2 of the policy and
check the statement before acting on it. Citations also make failures visible: a
wrong page is a debuggable retrieval error, not an invisible model mistake.

**What if no relevant document is found?**
The service returns a fixed refusal ("I could not find this information in the
available documents") with `fallback=true` and no citations. The query is still
logged so the team can see which topics are missing from the knowledge base -
that log doubles as a content-gap report.

**How do you verify answer relevance?**
Retrieval metrics (hit rate, MRR), keyword coverage of the expected fact,
citation correctness against gold pages, and embedding similarity between
question and answer. An LLM judge can be added on top, but only after calibrating
it against a human-labelled sample.

## Conversation & production concerns

**How is conversation history handled?**
Messages are stored with the citation snapshot. Only the last
`RAG_MAX_HISTORY_MESSAGES` turns are passed to the LLM, and short follow-ups
are merged with the previous question before retrieval. Clearing a conversation
deletes its messages; documents and vectors are untouched.

**How would you scale this?**
Qdrant and PostgreSQL scale independently, ingestion runs in a background worker
queue, embeddings are batched, and retrieval is stateless. Next steps: hybrid
search, a cross-encoder reranker, response streaming, per-document ACLs and a
cache keyed on normalised question + retrieved chunk ids.

**What about security?**
Uploaded files are stored with opaque UUID names, only PDFs with a valid header
are accepted, size is capped, and vectors are deleted with the document. A
production deployment would add authentication, per-tenant collections, malware
scanning and prompt-injection filtering on extracted text.
