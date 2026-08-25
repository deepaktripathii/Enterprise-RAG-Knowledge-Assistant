# RAG evaluation

Evaluation answers four separate questions, because a RAG failure can happen at
any stage:

| Stage | Metric | What it catches |
| --- | --- | --- |
| Retrieval | `retrieval_hit_rate` | Did any retrieved chunk come from the gold document/page? |
| Retrieval | `mrr` | How highly ranked was the first correct chunk? |
| Retrieval | `context_precision` | How much of the context sent to the LLM was relevant? |
| Answer | `answer_keyword_coverage` | Does the answer contain the expected fact (e.g. "24", "6,000")? |
| Answer | `citation_accuracy` | Do citations point at the gold document/page? |
| Answer | `faithfulness` | Is every answer sentence supported by the cited snippets? |
| Answer | `answer_relevance` | Cosine similarity between question and answer embeddings |
| Behaviour | `fallback_rate` | How often the assistant refused instead of guessing |

All metrics are deterministic and computed locally, so the suite runs in CI
without an LLM judge or API key.

## Dataset

`data/evaluation/questions.jsonl` contains 12 questions over the 5 sample policy
PDFs (23 page-level chunks). Each row has:

```json
{
  "id": "q05",
  "question": "What is the home office allowance for new employees?",
  "gold_document": "Remote_Work_Policy.pdf",
  "gold_pages": [3],
  "expected_keywords": ["25,000"],
  "reference_answer": "New employees receive a one-time home office allowance of INR 25,000."
}
```

Why these questions:

- single-fact lookups with a numeric constraint (limits, deadlines, password length),
- questions whose answer lives on a specific page, so page citations are testable,
- questions phrased differently from the source text, so lexical overlap alone
  is not enough,
- one cross-document topic (security reporting appears in both the security and
  remote-work policies) to check that the right document wins.

## Reproduction

```bash
python scripts/make_sample_documents.py
python scripts/seed_documents.py --backend sentence-transformers
python scripts/evaluate.py --backend sentence-transformers
```

Or through the API: `POST /evaluation/run`, then `GET /evaluation/latest`.

## Results (12 questions, top-k = 5)

Measured with `sentence-transformers/all-MiniLM-L6-v2` embeddings and the
extractive answerer (no LLM API key), on the 5 sample PDFs:

| Metric | Score |
| --- | --- |
| Retrieval hit rate | **1.000** |
| MRR | **0.958** |
| Context precision | 0.200 |
| Answer keyword coverage | **0.917** |
| Citation accuracy | **0.917** |
| Faithfulness | **1.000** |
| Answer relevance | 0.735 |
| Grounded rate | **1.000** |
| Fallback rate | 0.000 |
| Average total latency | **~9 ms** |

For comparison, the same pipeline with the offline hashing embedder reaches
0.917 hit rate, 0.854 MRR and 0.667 grounded rate - still usable, but the
semantic encoder clearly ranks the right page higher.

Interpretation:

- **Context precision of 0.20 is expected**, not a bug: only one of five pages
  is the gold page. Sending five chunks gives the LLM surrounding context at the
  cost of prompt tokens. Lower `top_k` raises this number and reduces recall.
- **Citation accuracy of 0.917** means 11 of 12 answers cited only gold pages.
  The remaining case cited a second page from the same document that also shares
  vocabulary with the question.
- **Faithfulness of 1.0** is a property of the extractive answerer: sentences are
  copied from the retrieved text. With an LLM provider the score drops and
  becomes a meaningful signal - that is exactly why the metric exists.
- **Answer relevance below 1.0** reflects short, factual answers: "P1 first
  response within 15 minutes" is a correct but lexically narrow answer to a
  broader question.

## Known limitations

1. Heuristic faithfulness uses content-word overlap, not entailment. A fluent
   paraphrase can score low; an LLM judge with human calibration is the next
   step.
2. Twelve questions is enough to catch regressions, not to certify quality.
   Grow the set with real employee questions and stratify by document type.
3. The extractive answerer cannot combine facts across chunks as fluently as an
   LLM; evaluate again with the configured provider before deployment.
4. Latency numbers exclude network time to an external LLM provider. Retrieval
   and answer latency are tracked separately so both can be monitored.
5. No adversarial set yet (prompt injection inside PDFs, contradictory policies).
   The citation validator blocks invented sources, but content-level trust needs
   curated tests.
