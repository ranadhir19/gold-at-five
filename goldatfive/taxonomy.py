"""Failure taxonomy: every miss gets a named cause and a cheapest fix.

Categories are assigned by the FIRST matching signal in a fixed
precedence — structural causes (a split answer, a flooded top-5)
before ranking-depth causes (the gold simply sat below the line):

  reranker_harm          stage 1 had it in the top-5; the reranker
                         pushed it out     → constrain or drop reranker
  chunk_boundary_straddle the answer is split across the gold chunk
                         and a sibling     → fix chunking (overlap /
                         parent-child), not the ranker
  near_duplicate_flooding >=3 near-identical chunks crowd the top-5
                                           → dedupe / diversity (MMR)
  vocabulary_mismatch    half the question's terms appear in NO chunk
                                           → query expansion / synonyms
  below_cutoff           gold sits at rank 6-20 of stage 1
                                           → widen K or add a reranker
  paraphrase_gap         the terms exist, the gold barely shares them,
                         a surface-matching rival wins
                                           → hybrid + expansion, or
                                             embeddings
  deep_retrieval_miss    none of the above; stage 1 never surfaced it
                                           → revisit indexing entirely
"""

from __future__ import annotations

import math

from .corpus import Corpus
from .metrics import POOL_K, KEEP, rank_of
from .retrievers import Index
from .text import content_terms

FIXES = {
    "reranker_harm": "constrain or drop the reranker (it is losing "
                     "more than the pool gives it)",
    "chunk_boundary_straddle": "fix chunking: overlap windows or "
                               "parent-child retrieval",
    "near_duplicate_flooding": "dedupe near-identical chunks or add "
                               "diversity (MMR) before the cut",
    "below_cutoff": "widen retrieval K or add a reranker stage",
    "vocabulary_mismatch": "query expansion / synonym table; the "
                           "corpus never uses the question's words",
    "paraphrase_gap": "hybrid retrieval + query rewriting, or "
                      "embeddings; surface terms point elsewhere",
    "deep_retrieval_miss": "revisit indexing (analyser, chunking, "
                           "embeddings); no cheap stage fix",
}

_STRADDLE_GOLD_MAX = 0.5
_STRADDLE_JOINT_MIN = 0.8
_FLOOD_MIN = 3
_FLOOD_COS = 0.55
_VOCAB_FRAC = 0.5
_PARA_GOLD_MAX = 0.4


def _coverage(terms: list[str], text_terms: set[str]) -> float:
    if not terms:
        return 0.0
    uniq = list(dict.fromkeys(terms))
    return sum(1 for t in uniq if t in text_terms) / len(uniq)


def _cosine(index: Index, a: str, b: str) -> float:
    ta, tb = index.tf[a], index.tf[b]
    dot = 0.0
    na = 0.0
    for t, f in ta.items():
        w = (1 + math.log(f)) * index.idf(t)
        na += w * w
        if t in tb:
            dot += w * (1 + math.log(tb[t])) * index.idf(t)
    nb = math.sqrt(sum(((1 + math.log(f)) * index.idf(t)) ** 2
                       for t, f in tb.items()))
    return dot / ((math.sqrt(na) or 1.0) * (nb or 1.0))


def classify(question: str, gold: str, relevant: set[str],
             stage1: list[str], final: list[str], has_reranker: bool,
             index: Index, corpus: Corpus) -> tuple[str, str] | None:
    """(category, cheapest_fix) for a missed question, else None."""
    rf = rank_of(final, gold)
    if rf is not None and rf <= KEEP:
        return None
    by_id = corpus.by_id()
    gold_chunk = by_id[gold]
    q_terms = content_terms(question)
    q_set = set(q_terms)
    r1 = rank_of(stage1, gold)

    if has_reranker and r1 is not None and r1 <= KEEP:
        return "reranker_harm", FIXES["reranker_harm"]

    gold_terms = set(index.tokens[gold])
    gold_cover = _coverage(q_terms, gold_terms)
    sibling_terms: set[str] = set()
    for sib in corpus.siblings(gold_chunk):
        sibling_terms |= set(index.tokens[sib.id])
    if gold_cover < _STRADDLE_GOLD_MAX and \
            _coverage(q_terms, gold_terms | sibling_terms) \
            >= _STRADDLE_JOINT_MIN:
        return "chunk_boundary_straddle", FIXES["chunk_boundary_straddle"]

    top5 = [cid for cid in final[:KEEP] if cid not in relevant]
    for cid in top5:
        cluster = 1 + sum(1 for other in top5
                          if other != cid
                          and _cosine(index, cid, other) > _FLOOD_COS)
        if cluster >= _FLOOD_MIN:
            return ("near_duplicate_flooding",
                    FIXES["near_duplicate_flooding"])

    absent = sum(1 for t in dict.fromkeys(q_terms)
                 if index.df.get(t, 0) == 0)
    if q_terms and absent / len(dict.fromkeys(q_terms)) >= _VOCAB_FRAC:
        return "vocabulary_mismatch", FIXES["vocabulary_mismatch"]

    if r1 is not None and r1 <= POOL_K:
        return "below_cutoff", FIXES["below_cutoff"]

    if gold_cover < _PARA_GOLD_MAX and final:
        top1_cover = _coverage(q_terms, set(index.tokens[final[0]]))
        if top1_cover > gold_cover:
            return "paraphrase_gap", FIXES["paraphrase_gap"]

    return "deep_retrieval_miss", FIXES["deep_retrieval_miss"]
