"""The bench: run every strategy over the labelled set, attribute
every outcome to a stage, classify every miss, and keep the raw
numbers that the report (and the README) are written from.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import taxonomy
from .corpus import Corpus, load_corpus, load_questions
from .metrics import (CURVE_KS, POOL_K, context_tokens,
                      precision_at_5, rank_of, recall_at_k,
                      stage_attribution)
from .report import make_verdict
from .retrievers import RETRIEVERS, Index, hybrid_rank
from .rerankers import lexical_rerank, merge_llm_order, parse_llm_order

# strategy name -> (first stage, reranker or None)
STRATEGIES = {
    "bm25": ("bm25", None),
    "tfidf": ("tfidf", None),
    "hybrid": ("hybrid", None),
    "hybrid+lex": ("hybrid", "lex"),
    "hybrid+llm": ("hybrid", "llm"),
}

_KIND_TO_CATEGORY = {
    "paraphrase": "paraphrase_gap",
    "straddle": "chunk_boundary_straddle",
    "duplicate": "near_duplicate_flooding",
    "vocab": "vocabulary_mismatch",
}


def _stage1(index: Index, stage: str, query: str) -> list[str]:
    if stage == "hybrid":
        return hybrid_rank(index, query)
    return RETRIEVERS[stage](index, query)


def _llm_prompt(question: str, pool: list[str],
                by_id: dict) -> str:
    lines = [f"[{cid}] {by_id[cid].text[:500]}" for cid in pool]
    return (
        "You are ranking passages for a search system.\n\n"
        f"Question: {question}\n\nPassages:\n" + "\n".join(lines) +
        "\n\nReturn ONLY a JSON array of the passage ids, ordered "
        "most relevant first, each id exactly once.")


def _llm_rankings(questions: list[dict], stage1_by_q: dict,
                  by_id: dict, cache_path: Path | None,
                  mode: str, model: str) -> dict | None:
    """{qid: {"order": [...pool order...], "prompt_tokens": n,
    "completion_tokens": n}} or None if the LLM arm cannot run."""
    cache: dict = {}
    if cache_path and cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    rankings = cache.get("rankings", {})
    if mode == "cache":
        return rankings if rankings else None
    if mode == "off":
        return None
    missing = [q for q in questions if q["id"] not in rankings]
    if missing:
        from . import llm_client
        served = cache.get("model_served")
        for q in missing:
            pool = stage1_by_q[q["id"]][:POOL_K]
            reply = llm_client.chat(
                model, _llm_prompt(q["question"], pool, by_id))
            order = parse_llm_order(reply["text"], pool)
            rankings[q["id"]] = {
                "order": order,
                "prompt_tokens": reply["prompt_tokens"],
                "completion_tokens": reply["completion_tokens"],
            }
            served = reply["model"] or served
        cache = {"model_requested": model, "model_served": served,
                 "rankings": rankings}
        if cache_path:
            cache_path.write_text(
                json.dumps(cache, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
    return rankings


def run(corpus_path, questions_path, llm_cache_path=None,
        llm_mode: str = "cache",
        llm_model: str = "gemini-3.5-flash") -> dict:
    corpus: Corpus = load_corpus(corpus_path)
    questions = load_questions(questions_path)
    index = Index(corpus)
    by_id = corpus.by_id()
    texts = {c.id: c.text for c in corpus.chunks}

    stage1_by_strategy: dict[str, dict[str, list[str]]] = {}
    for name, (stage, _) in STRATEGIES.items():
        stage1_by_strategy[name] = {
            q["id"]: _stage1(index, stage, q["question"])
            for q in questions}

    llm = None
    if llm_mode != "off":
        llm = _llm_rankings(questions, stage1_by_strategy["hybrid+llm"],
                            by_id,
                            Path(llm_cache_path) if llm_cache_path
                            else None,
                            llm_mode, llm_model)

    active = [n for n in STRATEGIES
              if n != "hybrid+llm" or llm is not None]
    results: dict = {
        "corpus": {"documents": len({c.doc_id for c in corpus.chunks}),
                   "chunks": len(corpus.chunks)},
        "questions": len(questions),
        "strategies": {},
        "llm_model": (json.loads(Path(llm_cache_path).read_text())
                      .get("model_served") if llm is not None
                      and llm_cache_path
                      and Path(llm_cache_path).exists() else None),
    }

    for name in active:
        _, reranker = STRATEGIES[name]
        per_q = []
        curve = {str(k): 0 for k in CURVE_KS}
        attrib = {"lift": 0, "harm": 0, "held": 0, "still_out": 0}
        tax_counts: dict[str, int] = {}
        tot_prec = 0.0
        tot_ctx = 0
        correct = 0
        rerank_tokens = 0
        r1_ranks, rf_ranks = [], []
        for q in questions:
            s1 = stage1_by_strategy[name][q["id"]]
            pool = s1[:POOL_K]
            if reranker == "lex":
                final = lexical_rerank(q["question"], pool, by_id) \
                    + s1[POOL_K:]
            elif reranker == "llm":
                entry = llm[q["id"]]
                merged = merge_llm_order(pool, entry["order"])
                final = merged + s1[POOL_K:]
                rerank_tokens += (entry.get("prompt_tokens") or 0) \
                    + (entry.get("completion_tokens") or 0)
            else:
                final = s1
            gold = q["gold"]
            relevant = set(q["relevant"])
            for k in CURVE_KS:
                if recall_at_k(s1, gold, k):
                    curve[str(k)] += 1
            att = stage_attribution(s1, final, gold)
            attrib[att] += 1
            hit = recall_at_k(final, gold, 5)
            correct += 1 if hit else 0
            tot_prec += precision_at_5(final, relevant)
            tot_ctx += context_tokens(final, texts)
            r1 = rank_of(s1, gold)
            rf = rank_of(final, gold)
            if r1:
                r1_ranks.append(r1)
            if rf:
                rf_ranks.append(rf)
            cat = taxonomy.classify(q["question"], gold, relevant, s1,
                                    final, reranker is not None,
                                    index, corpus)
            if cat:
                tax_counts[cat[0]] = tax_counts.get(cat[0], 0) + 1
            per_q.append({
                "id": q["id"], "kind": q["kind"], "gold": gold,
                "stage1_rank": r1, "final_rank": rf,
                "attribution": att,
                "taxonomy": cat[0] if cat else None,
            })
        n = len(questions)
        strat = {
            "recall_curve_stage1": {
                k: round(v / n, 6) for k, v in curve.items()},
            "recall_at_5": round(correct / n, 6),
            "precision_at_5": round(tot_prec / n, 6),
            "attribution": attrib,
            "mean_gold_rank_stage1": round(sum(r1_ranks)
                                           / len(r1_ranks), 4),
            "mean_gold_rank_final": round(sum(rf_ranks)
                                          / len(rf_ranks), 4),
            "context_tokens_total": tot_ctx,
            "context_tokens_per_correct": (
                round(tot_ctx / correct, 1) if correct else None),
            "rerank_tokens_total": rerank_tokens,
            "cost_tokens_per_correct": (
                round((tot_ctx + rerank_tokens) / correct, 1)
                if correct else None),
            "taxonomy": dict(sorted(tax_counts.items())),
            "per_question": per_q,
        }
        results["strategies"][name] = strat

    # Taxonomy audit: does the classifier name the failures the corpus
    # was authored to contain? Judged on the hybrid baseline's misses.
    hyb = results["strategies"]["hybrid"]["per_question"]
    by_qid = {p["id"]: p for p in hyb}
    audit = {"labelled": 0, "missed": 0, "agreed": 0, "details": {}}
    for q in questions:
        expected = _KIND_TO_CATEGORY.get(q["kind"])
        if not expected:
            continue
        audit["labelled"] += 1
        got = by_qid[q["id"]]["taxonomy"]
        if got is not None:
            audit["missed"] += 1
            if got == expected:
                audit["agreed"] += 1
            audit["details"][q["id"]] = {"expected": expected,
                                         "got": got}
        else:
            audit["details"][q["id"]] = {"expected": expected,
                                         "got": "no_miss"}
    results["taxonomy_audit"] = audit
    results["verdict"] = make_verdict(results)
    return results
