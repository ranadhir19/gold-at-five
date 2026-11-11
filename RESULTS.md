# gold-at-five — bench results

Corpus: 16 documents, 52 chunks. Labelled questions: 44. LLM reranker served by: not run in this snapshot.

## Strategy comparison

| strategy | recall@5 | precision@5 | lift | harm | ctx tokens/correct | total cost/correct |
|---|---|---|---|---|---|---|
| bm25 | 0.795 | 0.173 | 0 | 0 | 195.2 | 195.2 |
| tfidf | 0.818 | 0.177 | 0 | 0 | 188.8 | 188.8 |
| hybrid | 0.818 | 0.177 | 0 | 0 | 189.5 | 189.5 |
| hybrid+lex | 0.727 | 0.155 | 0 | 4 | 215.1 | 215.1 |

## Stage-1 recall curve (fraction of golds at or above K)

| strategy | @1 | @3 | @5 | @10 | @20 | @50 |
|---|---|---|---|---|---|---|
| bm25 | 0.568 | 0.750 | 0.795 | 0.886 | 0.909 | 1.000 |
| tfidf | 0.568 | 0.750 | 0.818 | 0.841 | 0.909 | 1.000 |
| hybrid | 0.591 | 0.750 | 0.818 | 0.841 | 0.909 | 1.000 |
| hybrid+lex | 0.591 | 0.750 | 0.818 | 0.841 | 0.909 | 1.000 |

## Failure taxonomy (misses only)

- **bm25**: below_cutoff ×1, chunk_boundary_straddle ×1, paraphrase_gap ×1, vocabulary_mismatch ×6
- **tfidf**: below_cutoff ×1, paraphrase_gap ×1, vocabulary_mismatch ×6
- **hybrid**: below_cutoff ×1, paraphrase_gap ×1, vocabulary_mismatch ×6
- **hybrid+lex**: below_cutoff ×1, paraphrase_gap ×1, reranker_harm ×4, vocabulary_mismatch ×6

## Taxonomy audit

Questions authored to fail in a named way: 26. Missed under hybrid: 7. Classifier agreed with the authored kind: 3.

## Verdict

Run **tfidf**: recall@5 0.818, precision@5 0.177, 188.8 tokens per correct answer (context + rerank cost). That is level with plain hybrid retrieval on recall@5 — the win is on precision and cost, not coverage. Its commonest remaining failure is vocabulary_mismatch (6 of 44 questions) — fix that before buying a bigger model.
