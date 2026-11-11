"""Stage attribution: who found the gold chunk, who lost it.

Every metric is computed from two full rankings per question — the
first-stage ranking and the strategy's final ranking — plus the
labelled gold chunk and relevant set.
"""

from __future__ import annotations

from .text import estimate_tokens

CURVE_KS = (1, 3, 5, 10, 20, 50)
POOL_K = 20   # reranker candidate pool depth
KEEP = 5      # what the context window actually receives


def rank_of(ranking: list[str], chunk_id: str) -> int | None:
    """1-based rank, or None if absent."""
    try:
        return ranking.index(chunk_id) + 1
    except ValueError:
        return None


def recall_at_k(ranking: list[str], gold: str, k: int) -> bool:
    r = rank_of(ranking, gold)
    return r is not None and r <= k


def precision_at_5(final: list[str], relevant: set[str]) -> float:
    top = final[:KEEP]
    return sum(1 for cid in top if cid in relevant) / KEEP


def context_tokens(final: list[str], texts: dict[str, str]) -> int:
    return sum(estimate_tokens(texts[cid]) for cid in final[:KEEP])


def stage_attribution(stage1: list[str], final: list[str],
                      gold: str) -> str:
    """lift / harm / held / still_out, relative to the top-5 line.

    lift:  outside stage-1 top-5, inside final top-5 (reranker saved it)
    harm:  inside stage-1 top-5, outside final top-5 (reranker lost it)
    held:  inside both
    still_out: outside both
    """
    r1 = rank_of(stage1, gold)
    rf = rank_of(final, gold)
    in1 = r1 is not None and r1 <= KEEP
    inf = rf is not None and rf <= KEEP
    if inf and not in1:
        return "lift"
    if in1 and not inf:
        return "harm"
    if in1 and inf:
        return "held"
    return "still_out"
