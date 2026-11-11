"""Corpus loading and chunking.

Input formats:

- a JSON file: ``{"documents": [{"id", "title", "paragraphs": [...]}]}``
  (the bundled bench corpus uses this), or
- a directory of ``.txt`` / ``.md`` files, one document per file,
  paragraphs split on blank lines, first line treated as the title if
  it starts with ``#``.

Chunking is one chunk per paragraph — the configuration whose failure
modes (notably the boundary straddle) this bench exists to diagnose.
A chunk id is ``<doc_id>#<n>`` with ``n`` 1-based.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    id: str
    doc_id: str
    index: int  # 0-based position within the document
    title: str
    text: str


@dataclass
class Corpus:
    chunks: list[Chunk] = field(default_factory=list)

    def by_id(self) -> dict[str, Chunk]:
        return {c.id: c for c in self.chunks}

    def siblings(self, chunk: Chunk) -> list[Chunk]:
        """Chunks adjacent to ``chunk`` in the same document."""
        return [c for c in self.chunks
                if c.doc_id == chunk.doc_id
                and abs(c.index - chunk.index) == 1]


def _chunks_from_docs(docs: list[dict]) -> Corpus:
    corpus = Corpus()
    for doc in docs:
        for i, para in enumerate(doc["paragraphs"]):
            corpus.chunks.append(Chunk(
                id=f"{doc['id']}#{i + 1}",
                doc_id=doc["id"],
                index=i,
                title=doc.get("title", doc["id"]),
                text=para.strip(),
            ))
    return corpus


def load_corpus(path: str | Path) -> Corpus:
    path = Path(path)
    if path.is_file() and path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return _chunks_from_docs(data["documents"])
    if path.is_dir():
        docs = []
        for f in sorted(path.iterdir()):
            if f.suffix not in (".txt", ".md"):
                continue
            raw = f.read_text(encoding="utf-8").strip()
            lines = raw.splitlines()
            title = f.stem
            if lines and lines[0].startswith("#"):
                title = lines[0].lstrip("# ").strip()
                raw = "\n".join(lines[1:]).strip()
            paras = [p.strip() for p in raw.split("\n\n") if p.strip()]
            docs.append({"id": f.stem, "title": title, "paragraphs": paras})
        return _chunks_from_docs(docs)
    raise ValueError(f"cannot load corpus from {path}")


def load_questions(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["questions"]
