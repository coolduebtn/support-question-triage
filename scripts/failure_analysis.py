"""Failure analysis from saved runs (no API calls): confusion matrices, per-class results, B's error table,
and A's risk-coverage curve. Usage: uv run python scripts/failure_analysis.py [split]"""
import json
import sys
from collections import Counter
from pathlib import Path

from triage.evals.analysis import error_table, risk_coverage
from triage.schemas import GoldItem, RunResult

split = sys.argv[1] if len(sys.argv) > 1 else "dev"
APPROACHES = {("A", "vector", False): "A", ("A+", "vector", False): "A+", ("B", "vector", False): "B"}
LABELS = ["answer", "clarify", "escalate"]
gold = {g.qid: g for g in (GoldItem.model_validate_json(l) for l in Path("data/gold/gold.jsonl").read_text().splitlines())}


def latest_runs() -> dict[str, RunResult]:
    out = {}
    for d in sorted(Path("results/runs").glob(f"*-{split}")):             # newest last
        cfg = json.loads((d / "config.json").read_text())["config"]
        key = APPROACHES.get((cfg["policy"], cfg["retriever"], bool(cfg.get("clean_query"))))
        if key is None or cfg.get("synthetic", "both") == "only":
            continue
        preds = [json.loads(l) for l in (d / "predictions.jsonl").read_text().splitlines() if l.strip()]
        preds = [p for p in preds if p.get("repeat", 0) == 0]
        m = json.loads((d / "metrics.json").read_text())
        out[key] = RunResult(run_id=d.name, config_hash=d.name.split("-")[2], split=split, predictions=preds, metrics=m,
                             n_errors=sum(p.get("error") is not None for p in preds),
                             cost_per_q=sum(p["cost_usd"] for p in preds) / len(preds), latency_p50=None)
    return out


def confusion(r: RunResult) -> list[str]:
    c = Counter((p["gold"], p["pred"]) for p in r.predictions)
    lines = ["| correct \\ predicted | answer | clarify | escalate | error |", "|---|---|---|---|---|"]
    for g in LABELS:
        lines.append(f"| {g} | " + " | ".join(str(c[(g, p)]) for p in [*LABELS, "error"]) + " |")
    return lines


def recall_of(r: RunResult, label: str, synthetic: bool | None = None) -> str:
    rows = [p for p in r.predictions if p["gold"] == label and (synthetic is None or gold[p["qid"]].synthetic == synthetic)]
    if not rows:
        return "n/a"
    return f"{sum(p['pred'] == label for p in rows)} of {len(rows)}"


R = latest_runs()
out = [f"# Failure analysis ({split} split, saved runs, no API calls)\n",
       "Runs: " + ", ".join(f"{k} `{v.run_id}`" for k, v in R.items()) + "\n"]
for name, r in R.items():
    out += [f"\n## {name}: what it predicted for each correct action\n", *confusion(r)]

out += ["\n## Per-class recall (share of items of that class the approach got right)\n",
        "| approach | answer | escalate | clarify (all) | clarify, real | clarify, constructed |", "|---|---|---|---|---|---|"]
for name, r in R.items():
    out.append(f"| {name} | {recall_of(r, 'answer')} | {recall_of(r, 'escalate')} | {recall_of(r, 'clarify')} | "
               f"{recall_of(r, 'clarify', False)} | {recall_of(r, 'clarify', True)} |")

if "B" in R:
    tab = error_table(R["B"], list(gold.values()))
    causes = Counter(t["cause_hint"] for t in tab)
    out += [f"\n## B: {len(tab)} wrong of {len(R['B'].predictions)}, by automatic cause hint\n",
            "| cause hint | count |", "|---|---|", *[f"| {c} | {n} |" for c, n in causes.most_common()],
            "\nHints are automatic (retrieval_miss = no gold page in the retrieved pages; missed_ambiguity = answered a CLARIFY item; "
            "unsafe_answer = answered an ESCALATE item; class_boundary = any other mismatch). Confirm or correct them by reading the rows.\n",
            "| qid | correct | predicted | hint | question | why B chose that |", "|---|---|---|---|---|---|"]
    for t in tab:
        q = t["text"].replace("\n", " ")[:110].replace("|", "/")
        why = (t["reason"] or "").replace("\n", " ")[:140].replace("|", "/")
        out.append(f"| {t['qid']} | {t['gold']} | {t['pred']} | {t['cause_hint']} | {q} | {why} |")
    judged = [p for p in R["B"].predictions if p["pred"] == "clarify" and p["gold"] == "clarify" and p.get("judge")]
    if judged:
        graded = [p for p in judged if p["judge"].get("followup") is not None]
        ok = sum(bool(p["judge"]["followup"]) for p in graded)
        out.append(f"\nFollow-up targeting (B, correct CLARIFY items judged): {ok} of {len(graded)} follow-ups name the missing detail.")

if "A" in R:
    curve = risk_coverage(R["A"])
    out += ["\n## Risk-coverage for A (sweep the top vector cosine; this is what A+'s threshold does)\n",
            "Coverage = share of all questions answered. Risk = share of answered questions that should not have been answered.\n",
            "| coverage | risk |", "|---|---|"]
    step = max(1, len(curve) // 12)
    out += [f"| {c:.2f} | {rk:.2f} |" for c, rk in curve[::step]]
    out.append("\nIf risk stays high at every coverage level, similarity alone cannot separate answerable from unanswerable questions.")
Path("results/failure_analysis.md").write_text("\n".join(out) + "\n")
print("wrote results/failure_analysis.md")
