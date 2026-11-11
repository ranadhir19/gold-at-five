"""First-stage retrievers, implemented from scratch on the stdlib.

BM25 (Okapi) and TF-IDF cosine share one inverted-index pass. The
hybrid is reciprocal rank fusion of the two rankings. Every retriever
returns the FULL ranking of chunk ids — the bench needs gold ranks far
below the cutoff to attribute failures, not just a top-K.
"""

from __future__ import annotations

import math
from collections import Counter

from .corpus import Corpus
from .text import tokenize


class Index:
    def __init__(self, corpus: Corpus):
        self.chunks = corpus.chunks
        self.tokens: dict[str, list[str]] = {}
        self.tf: dict[str, Counter] = {}
        self.df: Counter = Counter()
        self.doc_len: dict[str, int] = {}
        for c in self.chunks:
            toks = tokenize(c.text)
            self.tokens[c.id] = toks
            self.tf[c.id] = Counter(toks)
            self.doc_len[c.id] = max(len(toks), 1)
            for term in set(toks):
                self.df[term] += 1
        self.n = len(self.chunks)
        self.avg_len = sum(self.doc_len.values()) / max(self.n, 1)

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        # Robertson/Spärck Jones idf, floored at 0.
        return max(math.log(1 + (self.n - df + 0.5) / (df + 0.5)), 0.0)


def _rank(scores: dict[str, float]) -> list[str]:
    # Score desc; chunk id asc as a deterministic tie-break.
    return sorted(scores, key=lambda cid: (-scores[cid], cid))


def bm25_rank(index: Index, query: str, k1: float = 1.5,
              b: float = 0.75) -> list[str]:
    terms = tokenize(query)
    scores: dict[str, float] = {}
    for c in index.chunks:
        tf = index.tf[c.id]
        dl = index.doc_len[c.id]
        norm = k1 * (1 - b + b * dl / index.avg_len)
        s = 0.0
        for t in terms:
            f = tf.get(t, 0)
            if f:
                s += index.idf(t) * f * (k1 + 1) / (f + norm)
        scores[c.id] = s
    return _rank(scores)


def tfidf_rank(index: Index, query: str) -> list[str]:
    qtf = Counter(tokenize(query))
    qvec = {t: (1 + math.log(c)) * index.idf(t) for t, c in qtf.items()}
    qnorm = math.sqrt(sum(v * v for v in qvec.values())) or 1.0
    scores: dict[str, float] = {}
    for c in index.chunks:
        tf = index.tf[c.id]
        dot = 0.0
        cnorm_sq = 0.0
        for t, f in tf.items():
            w = (1 + math.log(f)) * index.idf(t)
            cnorm_sq += w * w
            if t in qvec:
                dot += w * qvec[t]
        scores[c.id] = dot / ((math.sqrt(cnorm_sq) or 1.0) * qnorm)
    return _rank(scores)


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal rank fusion over full rankings."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for i, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + i + 1)
    return _rank(scores)


RETRIEVERS = {
    "bm25": bm25_rank,
    "tfidf": tfidf_rank,
}


def hybrid_rank(index: Index, query: str) -> list[str]:
    return rrf_fuse([bm25_rank(index, query), tfidf_rank(index, query)])
