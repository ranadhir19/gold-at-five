# gold-at-five

**A retrieval diagnostics bench for RAG pipelines. It does not return
a reranked list. It returns a verdict.**

Every reranking tool hands you the same two things: a reordered list
and a pair of aggregate metrics. Neither tells you the three things
you actually need before you ship:

1. **Which stage lost the answer?** A gold chunk can be missed by the
   retriever, or *found* by the retriever and then thrown away by the
   reranker. Recall@50 and precision@5 reported side by side never
   reconcile the two.
2. **Is the reranker hurting?** Rerankers are sold on *lift* — gold
   chunks promoted into the top-5. They can also *demote* chunks the
   retriever had right. That number is almost never measured. Here it
   is a headline metric.
3. **What is the cheapest fix?** A miss caused by a chunk boundary, a
   miss caused by near-duplicate flooding, and a miss caused by the
   question using words the corpus never uses need three different
   fixes at three different prices. A single recall number cannot
   tell them apart. A failure taxonomy can.

gold-at-five runs five strategies head-to-head on *your* labelled
questions — BM25, TF-IDF, hybrid (reciprocal rank fusion), hybrid +
lexical rerank, hybrid + LLM rerank — and reports, per strategy:
stage-1 recall@K curves, **lift and harm**, precision@5, context
tokens per correct answer, a named failure cause with its cheapest
fix for every miss, and a recommendation generated from the numbers.

Pure Python standard library. No dependencies, no API keys required
(the LLM arm is optional and uses free tiers).

## Run it

```bash
python3 validate.py     # harness: tests + reproduction + scans
python3 run_bench.py    # writes results.json and RESULTS.md
```

Bring your own data: `--corpus` accepts a JSON file
(`{"documents": [{"id", "title", "paragraphs": [...]}]}`) or a
directory of `.txt`/`.md` files (one document per file, paragraphs
split on blank lines). `--questions` takes
`{"questions": [{"id", "question", "gold", "relevant", "kind"}]}`
where `gold` is the chunk id that answers the question and `relevant`
lists every chunk that carries the answer. Chunk ids are
`<doc_id>#<paragraph number>`.

The LLM arm is cached: `python3 run_bench.py --llm refresh` calls the
model once per question and stores the raw rankings in
`llm_cache.json`; every later run recomputes the arm's metrics from
that artefact, offline, deterministically. Providers: Groq
(`--model openai/gpt-oss-20b`, key via `GROQ_API_KEY`) or Google AI
Studio (`--model gemini-3.5-flash`, key via `GEMINI_API_KEY`). The
cache records the model that actually *served* the answers.

## What the bundled bench found

The bundled corpus is an original operations handbook for a fictional
two-property hotel group — 16 documents, 52 paragraph chunks — with
44 labelled questions: 18 direct, 10 paraphrase, 6 written against
facts split across paragraphs, 6 aimed at the wrong property's
near-identical policy document, 4 using words the corpus never uses.

| strategy | recall@5 | precision@5 | lift | harm | tokens/correct |
|---|---|---|---|---|---|
| bm25 | 0.795 | 0.173 | 0 | 0 | 195.2 |
| tfidf | 0.818 | 0.177 | 0 | 0 | 188.8 |
| hybrid | 0.818 | 0.177 | 0 | 0 | 189.5 |
| hybrid+lex | 0.727 | 0.155 | **0** | **4** | 215.1 |

Three findings, none of them the expected one:

- **The lexical reranker was a net negative.** It promoted *nothing*
  into the top-5 (lift 0) and pushed four gold chunks *out* of it —
  including two that stage 1 had ranked **first** (a chunk answering
  "how many EV chargers, on what level" fell from rank 1 to 13).
  Aggregate recall hid this completely in every tool that doesn't
  count harm: 0.818 → 0.727 looks like noise; "your second stage
  deleted two correct answers" does not.
- **The retrievers barely differ; the questions differ.** Stage-1
  recall@50 is 1.000 for every strategy — every gold chunk is *in*
  the index and *in* the ranking. The residual misses are vocabulary
  failures (6 of 8 under hybrid): "valet", "sauna", "campervan"
  questions whose words appear in no chunk at all. No ranker fixes
  that; query expansion does. The bench's verdict engine therefore
  recommends plain TF-IDF (level with hybrid on recall, cheapest per
  correct answer) and names vocabulary mismatch, not ranking depth,
  as the thing to fix.
- **The taxonomy disagreed with the test author, usefully.** Of 26
  questions authored to fail in a named way, only 7 missed under
  hybrid — the near-duplicate property traps and the cross-paragraph
  facts were all handled. Four questions authored as "paraphrase"
  were classified `vocabulary_mismatch`, and on inspection the
  classifier was right: more than half their content words ("flight",
  "holiday", "straight away") exist nowhere in the corpus. A
  paraphrase whose replacement words are absent from the corpus *is*
  a vocabulary failure, whatever the test intended. The bench reports
  the disagreement (agreement 3 of 7 misses) instead of hiding it.

**Honest limitation — the LLM arm.** The `hybrid+llm` strategy is
implemented and probe-verified (single-question probes on both free
tiers each lifted the gold chunk from rank 7 to rank 1; Gemini
returned a complete 20-id ranking, Groq's reasoning model returned a
1-id partial ranking, which the merge handles). The full 44-question arm
could not be run for this snapshot: both free tiers returned HTTP 429
rate limits during the build, and the arm is all-or-nothing by design
(a partial-question arm would not be comparable). Run
`python3 run_bench.py --llm refresh` when a free tier has quota and
the arm joins the table with its lift, harm, and token cost measured
the same way. The verdict above is computed from the four arms that
did run.

## The taxonomy

Every miss is assigned the first matching cause, in a fixed
precedence (structural causes before ranking-depth causes), each with
its cheapest fix:

| category | signal | cheapest fix |
|---|---|---|
| `reranker_harm` | stage 1 top-5, final below 5 | constrain or drop the reranker |
| `chunk_boundary_straddle` | gold + sibling jointly cover the question, gold alone doesn't | overlap / parent-child chunking |
| `near_duplicate_flooding` | ≥3 near-identical chunks crowd the top-5 | dedupe / diversity (MMR) |
| `vocabulary_mismatch` | ≥half the question's terms appear in no chunk | query expansion / synonyms |
| `below_cutoff` | gold at stage-1 rank 6–20 | widen K or add a reranker |
| `paraphrase_gap` | terms exist, gold barely shares them, a rival wins on surface form | hybrid + rewriting, or embeddings |
| `deep_retrieval_miss` | none of the above | revisit indexing |

## Lineage

Ground-Up AI Labs Lab 03 ("Retrieval Without Magic") built a TF-IDF
retriever from arithmetic and measured chunking losing to
whole-document retrieval at rank 1 on a 26-question set. gold-at-five
is the instrument that lab's result calls for: reranking, harm,
taxonomy, and strategy verdicts, on a different and larger corpus.
The failure taxonomy's categories are operational definitions — each
is a checkable signal over the rankings, not a vibe — and the
classifier is itself audited against the corpus's authored kinds.

## Layout

| path | what |
|---|---|
| `goldatfive/retrievers.py` | BM25 + TF-IDF from scratch, RRF hybrid |
| `goldatfive/rerankers.py` | lexical interaction scorer; LLM order merge/parse |
| `goldatfive/metrics.py` | recall curves, precision@5, lift/harm attribution, context cost |
| `goldatfive/taxonomy.py` | the failure classifier + cheapest-fix table |
| `goldatfive/bench.py` | strategy runner; LLM rankings cached as an artefact |
| `goldatfive/report.py` | the verdict generator + Markdown report |
| `data/` | the bundled corpus + labelled questions |
| `tests/` | unit tests, including synthetic cases for every taxonomy branch |
| `validate.py` | the harness |
