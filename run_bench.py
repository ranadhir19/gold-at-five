#!/usr/bin/env python3
"""Run the gold-at-five bench and write results.json + RESULTS.md.

Usage:
  python3 run_bench.py [--corpus data/corpus.json]
                       [--questions data/questions.json]
                       [--llm cache|auto|refresh|off]
                       [--model gemini-3.5-flash]

--llm cache (default) recomputes the LLM arm from the committed
llm_cache.json so results reproduce offline; refresh re-calls the
model and rewrites the cache; off skips the LLM arm entirely.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from goldatfive import bench
from goldatfive.report import render_markdown

ROOT = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "data" / "corpus.json"))
    ap.add_argument("--questions",
                    default=str(ROOT / "data" / "questions.json"))
    ap.add_argument("--llm", default="cache",
                    choices=["cache", "auto", "refresh", "off"])
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--out", default=str(ROOT / "results.json"))
    args = ap.parse_args()

    mode = {"cache": "cache", "auto": "auto",
            "refresh": "auto", "off": "off"}[args.llm]
    cache_path = ROOT / "llm_cache.json"
    if args.llm == "refresh" and cache_path.exists():
        cache_path.unlink()

    results = bench.run(args.corpus, args.questions,
                        llm_cache_path=cache_path, llm_mode=mode,
                        llm_model=args.model)
    Path(args.out).write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    (ROOT / "RESULTS.md").write_text(render_markdown(results),
                                      encoding="utf-8")
    v = results["verdict"]
    print(f"verdict: {v['strategy']} — {v['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
