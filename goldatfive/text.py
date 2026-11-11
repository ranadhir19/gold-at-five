"""Tokenisation and token estimation.

One tokeniser is used everywhere in the bench — indexing, scoring,
taxonomy signals, cost accounting — so every comparison inside a run
is exact even where an absolute count is approximate.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_EST_RE = re.compile(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]")

STOPWORDS = frozenset(
    "a an the and or of to in on for with at by from is are was were be "
    "been it its this that these those i you he she we they me my your "
    "our their do does did can could will would should may might "
    "if then than so as not no yes what when where which who whom how "
    "much many any some all per".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, punctuation dropped."""
    return _TOKEN_RE.findall(text.lower())


def content_terms(text: str) -> list[str]:
    """Tokens minus stopwords — the terms a match is judged on."""
    return [t for t in tokenize(text) if t not in STOPWORDS]


def estimate_tokens(text: str) -> int:
    """Approximate token count: word runs + individual punctuation.

    Tracks real tokenisers closely enough for cost accounting and is
    identical on every run, which is what comparisons need.
    """
    return len(_EST_RE.findall(text))
