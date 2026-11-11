"""The verdict and the Markdown report — generated from the measured
numbers, never written around them.
"""

from __future__ import annotations


def _rank_key(item):
    name, s = item
    cost = s["cost_tokens_per_correct"]
    return (-s["recall_at_5"], -s["precision_at_5"],
            cost if cost is not None else float("inf"))


def make_verdict(results: dict) -> dict:
    strategies = results["strategies"]
    ranked = sorted(strategies.items(), key=_rank_key)
    winner, w = ranked[0]
    hyb = strategies["hybrid"]
    lines = [
        f"Run **{winner}**: recall@5 {w['recall_at_5']:.3f}, "
        f"precision@5 {w['precision_at_5']:.3f}, "
        f"{w['cost_tokens_per_correct']} tokens per correct answer "
        f"(context + rerank cost).",
    ]
    gain = w["recall_at_5"] - hyb["recall_at_5"]
    if winner != "hybrid":
        if abs(gain) < 0.001:
            lines.append(
                "That is level with plain hybrid retrieval on "
                "recall@5 — the win is on precision and cost, not "
                "coverage.")
        else:
            lines.append(
                f"Against plain hybrid retrieval that is "
                f"{gain:+.3f} recall@5.")
    if "hybrid+llm" in strategies:
        llm = strategies["hybrid+llm"]
        lex = strategies["hybrid+lex"]
        if llm["recall_at_5"] <= lex["recall_at_5"]:
            lines.append(
                f"The LLM reranker bought nothing here: recall@5 "
                f"{llm['recall_at_5']:.3f} vs the lexical reranker's "
                f"{lex['recall_at_5']:.3f}, at "
                f"{llm['rerank_tokens_total']} extra tokens. On this "
                f"labelled set the free local stage wins.")
        lines.append(
            f"LLM reranker attribution: lift "
            f"{llm['attribution']['lift']}, harm "
            f"{llm['attribution']['harm']}.")
    if w["attribution"]["harm"]:
        lines.append(
            f"Warning: the winning strategy still carries "
            f"{w['attribution']['harm']} reranker-harm case(s) — "
            f"questions stage 1 had right that the second stage lost.")
    if w["taxonomy"]:
        top = max(w["taxonomy"].items(), key=lambda kv: kv[1])
        lines.append(
            f"Its commonest remaining failure is {top[0]} "
            f"({top[1]} of {results['questions']} questions) — fix "
            f"that before buying a bigger model.")
    return {"strategy": winner, "text": " ".join(lines)}


def render_markdown(results: dict) -> str:
    s = results["strategies"]
    out = ["# gold-at-five — bench results", ""]
    out.append(f"Corpus: {results['corpus']['documents']} documents, "
               f"{results['corpus']['chunks']} chunks. "
               f"Labelled questions: {results['questions']}. "
               f"LLM reranker served by: "
               f"{results['llm_model'] or 'not run in this snapshot'}.")
    out += ["", "## Strategy comparison", "",
            "| strategy | recall@5 | precision@5 | lift | harm | "
            "ctx tokens/correct | total cost/correct |",
            "|---|---|---|---|---|---|---|"]
    for name, st in s.items():
        out.append(
            f"| {name} | {st['recall_at_5']:.3f} | "
            f"{st['precision_at_5']:.3f} | {st['attribution']['lift']} "
            f"| {st['attribution']['harm']} | "
            f"{st['context_tokens_per_correct']} | "
            f"{st['cost_tokens_per_correct']} |")
    out += ["", "## Stage-1 recall curve (fraction of golds at or "
            "above K)", "",
            "| strategy | @1 | @3 | @5 | @10 | @20 | @50 |",
            "|---|---|---|---|---|---|---|"]
    for name, st in s.items():
        c = st["recall_curve_stage1"]
        out.append(f"| {name} | " + " | ".join(
            f"{c[k]:.3f}" for k in ("1", "3", "5", "10", "20", "50"))
            + " |")
    out += ["", "## Failure taxonomy (misses only)", ""]
    for name, st in s.items():
        if st["taxonomy"]:
            items = ", ".join(f"{k} ×{v}"
                              for k, v in st["taxonomy"].items())
            out.append(f"- **{name}**: {items}")
        else:
            out.append(f"- **{name}**: no misses")
    a = results["taxonomy_audit"]
    out += ["", "## Taxonomy audit", "",
            f"Questions authored to fail in a named way: "
            f"{a['labelled']}. Missed under hybrid: {a['missed']}. "
            f"Classifier agreed with the authored kind: {a['agreed']}.",
            "", "## Verdict", "", results["verdict"]["text"], ""]
    return "\n".join(out)
