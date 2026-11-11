#!/usr/bin/env python3
"""Validation harness for gold-at-five.

Stages (each reported PASS/FAIL; the run fails if any stage fails):
  1. compile check of every Python file in the repo;
  2. unit tests (toy corpora with known right answers, including a
     synthetic case for every taxonomy branch);
  3. reproduction: run_bench.py is re-run and results.json +
     RESULTS.md must come back byte-identical — the numbers in the
     README are the numbers the code produces (the LLM arm, when its
     cache is present, is recomputed from the cached rankings);
  4. data integrity: gold/relevant ids exist, >=40 questions, every
     authored failure kind present;
  5. reference-integrity scan of the exact repo content (the
     portfolio pre-push gate; the scanner lives outside this repo);
  6. banned-name check: the forbidden portfolio name appears nowhere.

Run: python3 validate.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
# The scanner path is assembled at runtime (portfolio convention)
# so this file never carries a self-matching literal for stage 5.
SCANNER = (Path.home() / "workspace" / "github-fis-watch"
           / ("fis" + "_scan.py"))


def stage(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}"
          + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    ok = True

    # 1. compile
    r = subprocess.run(
        [sys.executable, "-m", "compileall", "-q",
         str(ROOT / "goldatfive"), str(ROOT / "run_bench.py"),
         str(ROOT / "tests")],
        capture_output=True, text=True)
    ok &= stage("compile", r.returncode == 0, r.stderr.strip()[:200])

    # 2. unit tests
    r = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s",
         str(ROOT / "tests")],
        capture_output=True, text=True, cwd=ROOT)
    ok &= stage("unit tests", r.returncode == 0,
                r.stderr.strip().splitlines()[-1][:120]
                if r.stderr.strip() else "")

    # 3. reproduction
    before_j = (ROOT / "results.json").read_bytes()
    before_m = (ROOT / "RESULTS.md").read_bytes()
    r = subprocess.run([sys.executable, "run_bench.py", "--llm",
                        "cache"],
                       capture_output=True, text=True, cwd=ROOT)
    same = r.returncode == 0 \
        and (ROOT / "results.json").read_bytes() == before_j \
        and (ROOT / "RESULTS.md").read_bytes() == before_m
    ok &= stage("reproduction", same,
                "results.json + RESULTS.md identical after re-run"
                if same else (r.stderr.strip()[:200]
                              or "DRIFT detected"))

    # 4. data integrity
    try:
        from goldatfive.corpus import load_corpus, load_questions
        corpus = load_corpus(ROOT / "data" / "corpus.json")
        questions = load_questions(ROOT / "data" / "questions.json")
        ids = {c.id for c in corpus.chunks}
        kinds: dict[str, int] = {}
        problems = []
        for q in questions:
            kinds[q["kind"]] = kinds.get(q["kind"], 0) + 1
            if q["gold"] not in ids:
                problems.append(f"{q['id']}: gold missing")
            if not set(q["relevant"]) <= ids:
                problems.append(f"{q['id']}: relevant id missing")
            if q["gold"] not in q["relevant"]:
                problems.append(f"{q['id']}: gold not in relevant")
        for kind in ("paraphrase", "straddle", "duplicate", "vocab"):
            if kinds.get(kind, 0) < 4:
                problems.append(f"kind {kind}: only "
                                f"{kinds.get(kind, 0)} questions")
        ok &= stage("data integrity",
                    len(questions) >= 40 and not problems,
                    f"{len(questions)} questions, kinds {kinds}"
                    + (f"; {problems}" if problems else ""))
    except Exception as exc:  # noqa: BLE001
        ok &= stage("data integrity", False, str(exc)[:200])

    # 5. reference-integrity scan
    if SCANNER.exists():
        scan = subprocess.run(
            [sys.executable, str(SCANNER), str(ROOT)],
            capture_output=True, text=True)
        hits = [ln for ln in scan.stdout.splitlines()
                if ln.startswith("{")]
        summary = [ln for ln in scan.stdout.splitlines()
                   if ln.startswith("#")]
        ok &= stage("reference-integrity scan", not hits,
                    summary[0] if summary else f"{len(hits)} candidate(s)")
    else:
        ok &= stage("reference-integrity scan", False,
                    "scanner not found on this machine")

    # 6. banned-name check
    banned = "Dev" + "Pulse"
    offenders = []
    for p in ROOT.rglob("*"):
        if p.is_file() and ".git" not in p.parts \
                and "__pycache__" not in p.parts:
            try:
                if banned in p.read_text(encoding="utf-8"):
                    offenders.append(str(p.relative_to(ROOT)))
            except (UnicodeDecodeError, OSError):
                pass
    ok &= stage("banned-name check", not offenders,
                ", ".join(offenders) if offenders else "clean")

    print("OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
