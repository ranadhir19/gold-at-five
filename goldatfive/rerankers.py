"""Second-stage rerankers.

Two implementations, deliberately different in kind:

- ``lexical_rerank`` — a deterministic interaction scorer. A
  cross-encoder's advantage over a bag-of-words retriever is that it
  reads query and chunk *together*; these features approximate that
  reading with arithmetic: how much of the query the chunk covers,
  whether query phrases survive intact, how tightly the matched terms
  cluster, and whether the match sits in the title.
- ``llm_rerank`` — listwise ranking by a hosted model over the
  candidate pool. Raw rankings are cached to a JSON artefact by the
  caller so reported numbers reproduce offline; this function only
  merges a returned ordering with the pool.
"""

from __future__ import annotations

import re

from .corpus import Chunk
from .text import content_terms, tokenize

W_COVERAGE = 3.0
W_PHRASE = 0.6
W_PROXIMITY = 1.0
W_TITLE = 0.4


def lexical_score(query: str, chunk: Chunk) -> float:
    q_terms = list(dict.fromkeys(content_terms(query)))
    if not q_terms:
        return 0.0
    toks = tokenize(chunk.text)
    tok_set = set(toks)
    matched = [t for t in q_terms if t in tok_set]
    coverage = len(matched) / len(q_terms)

    # Phrase survival: adjacent content-word pairs of the query, in order.
    raw = tokenize(query)
    pairs = [(raw[i], raw[i + 1]) for i in range(len(raw) - 1)
             if raw[i] in q_terms or raw[i + 1] in q_terms]
    phrase_hits = 0
    for a, b in pairs:
        for i in range(len(toks) - 1):
            if toks[i] == a and toks[i + 1] == b:
                phrase_hits += 1
                break

    # Proximity: smallest window of chunk tokens covering all matched
    # terms, as a fraction of chunk length (tighter scores higher).
    proximity = 0.0
    if len(matched) >= 2:
        positions = {t: [i for i, x in enumerate(toks) if x == t]
                     for t in matched}
        best = None
        for start in positions[matched[0]]:
            end = start
            ok = True
            for t in matched[1:]:
                nxt = [p for p in positions[t] if p >= start]
                if not nxt:
                    ok = False
                    break
                end = max(end, nxt[0])
            if ok:
                span = end - start + 1
                best = span if best is None else min(best, span)
        if best is not None:
            proximity = 1.0 - (best - len(matched)) / max(len(toks), 1)

    title_terms = set(tokenize(chunk.title))
    title_hits = sum(1 for t in matched if t in title_terms)

    return (W_COVERAGE * coverage + W_PHRASE * phrase_hits
            + W_PROXIMITY * proximity + W_TITLE * title_hits)


def lexical_rerank(query: str, pool: list[str],
                   by_id: dict[str, Chunk]) -> list[str]:
    scored = {cid: lexical_score(query, by_id[cid]) for cid in pool}
    return sorted(pool, key=lambda cid: (-scored[cid], pool.index(cid)))


_ID_RE = re.compile(r"[A-Za-z0-9_-]+#\d+")


def merge_llm_order(pool: list[str], returned: list[str]) -> list[str]:
    """Model's ordering first (deduped, pool members only), then the
    unreturned pool members in their original order."""
    seen: list[str] = []
    for cid in returned:
        if cid in pool and cid not in seen:
            seen.append(cid)
    return seen + [cid for cid in pool if cid not in seen]


def parse_llm_order(reply: str, pool: list[str]) -> list[str]:
    """Extract chunk ids from a model reply: a JSON array if present,
    else ids in order of appearance."""
    found = _ID_RE.findall(reply or "")
    return merge_llm_order(pool, found)
