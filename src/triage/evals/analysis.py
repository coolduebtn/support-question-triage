from triage.evals.metrics import macro_f1
from triage.evals.stats import mcnemar_exact, min_detectable_effect, paired_bootstrap_diff
from triage.schemas import GoldItem, RunResult


def select_final(dev_results: dict[str, RunResult], limits: dict, complexity: dict[str, int]) -> str:
    """Pre-registered rule: best dev macro-F1 among configs within limits; configs within the minimum
    detectable effect of the best are treated as tied and the simplest of them wins."""
    eligible: dict[str, RunResult] = {}
    for name, r in dev_results.items():
        m = r.metrics
        if (m.get("over_ask_rate") or 0.0) > limits.get("over_ask_max", 1.0):
            continue
        if (m.get("over_escalation_rate") or 0.0) > limits.get("over_escalation_max", 1.0):
            continue
        if (m.get("unsafe_answer_rate") or 0.0) > limits.get("unsafe_answer_max", 1.0):
            continue
        if r.cost_per_q > limits.get("cost_per_q_max", float("inf")):
            continue
        eligible[name] = r
    if not eligible:
        raise ValueError("no configuration is within the pre-registered limits")
    best = max(r.metrics["macro_f1"] for r in eligible.values())
    mde = min_detectable_effect(max(r.metrics["n_items"] for r in eligible.values()))
    tied = [n for n, r in eligible.items() if best - r.metrics["macro_f1"] <= mde]
    return min(tied, key=lambda n: (complexity[n], -eligible[n].metrics["macro_f1"]))


def compare_pair(a: RunResult, b: RunResult, seed: int = 13) -> dict:
    """Paired comparison of b against a on the same items. `claim` is True only when the difference is both
    larger than the minimum detectable effect and its bootstrap interval excludes zero."""
    ra = {r["qid"]: r for r in a.predictions}
    rb = {r["qid"]: r for r in b.predictions}
    if ra.keys() != rb.keys():
        raise ValueError("results cover different items")
    qids = sorted(ra)
    y_true = [ra[q]["gold"] for q in qids]
    pa, pb = [ra[q]["pred"] for q in qids], [rb[q]["pred"] for q in qids]
    diff = paired_bootstrap_diff(y_true, pa, pb, macro_f1, n=1000, seed=seed)
    p = mcnemar_exact([t == x for t, x in zip(y_true, pa)], [t == x for t, x in zip(y_true, pb)])
    mde = min_detectable_effect(len(qids))
    lo, hi = diff["ci"]
    return {"delta": diff["delta"], "ci": diff["ci"], "mcnemar_p": p, "mde": mde,
            "claim": abs(diff["delta"]) > mde and (lo > 0 or hi < 0)}


def error_table(result: RunResult, gold: list[GoldItem]) -> list[dict]:
    """Every misclassified item with an automatic cause hint; the owner confirms or corrects the hints."""
    by_qid = {g.qid: g for g in gold}
    table = []
    for r in sorted(result.predictions, key=lambda r: r["qid"]):
        if r["gold"] == r["pred"]:
            continue
        g = by_qid[r["qid"]]
        gold_pages = {e.page_id for e in g.evidence}
        if r["pred"] == "error":
            cause = "error"
        elif g.label.value == "answer" and gold_pages and not (gold_pages & set(r["retrieved_pages"])):
            cause = "retrieval_miss"
        elif g.label.value == "clarify" and r["pred"] == "answer":
            cause = "missed_ambiguity"
        elif g.label.value == "escalate" and r["pred"] == "answer":
            cause = "unsafe_answer"
        else:
            cause = "class_boundary"
        table.append({"qid": r["qid"], "gold": r["gold"], "pred": r["pred"], "text": g.text[:200],
                      "retrieved_pages": r["retrieved_pages"], "gold_pages": sorted(gold_pages),
                      "reason": r.get("reason", ""), "cause_hint": cause})
    return table


def risk_coverage(result: RunResult) -> list[tuple[float, float]]:
    """Sweep a threshold over the top vector cosine: coverage = share of questions answered, risk = share of
    answered questions whose gold label was not ANSWER."""
    rows = result.predictions
    n = len(rows)
    tops = sorted({(r.get("top_cosine") if r.get("top_cosine") is not None else -1.0) for r in rows}, reverse=True)
    curve = []
    for t in tops:
        answered = [r for r in rows if r["pred"] == "answer" and (r.get("top_cosine") if r.get("top_cosine") is not None else -1.0) >= t]
        if answered:
            curve.append((len(answered) / n, sum(r["gold"] != "answer" for r in answered) / len(answered)))
    return sorted(set(curve))


def _paired_mean_diff(diffs: list[float], n: int = 2000, seed: int = 13) -> dict:
    import numpy as np

    arr = np.array(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(arr, size=len(arr), replace=True).mean()) for _ in range(n)]
    lo, hi = float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
    return {"delta": float(arr.mean()), "ci": (lo, hi), "improves": lo > 0, "n": len(arr)}


def compare_retrieval(rows_a: list[dict], rows_b: list[dict], gold: dict[str, GoldItem], k: int = 5,
                      seed: int = 13) -> dict:
    """Paired comparison (b minus a) of page-level recall on gold ANSWER items, and of judged correctness on
    items where both runs have a judge verdict. `improves` means the bootstrap interval of the mean difference excludes zero."""
    ra, rb = {r["qid"]: r for r in rows_a}, {r["qid"]: r for r in rows_b}
    qids = sorted(q for q in ra if q in rb and gold[q].label.value == "answer" and gold[q].evidence)
    if not qids:
        raise ValueError("no gold ANSWER items with evidence to compare")

    def recall(row: dict, q: str) -> float:
        pages = {e.page_id for e in gold[q].evidence}
        return len(pages & set(row["retrieved_pages"][:k])) / len(pages)

    recall_out = _paired_mean_diff([recall(rb[q], q) - recall(ra[q], q) for q in qids], seed=seed)
    both = [q for q in qids if (ra[q].get("judge") or {}).get("correct") is not None
            and (rb[q].get("judge") or {}).get("correct") is not None]
    judged = (_paired_mean_diff([float(rb[q]["judge"]["correct"]) - float(ra[q]["judge"]["correct"]) for q in both], seed=seed)
              if both else None)
    return {"n": len(qids), "recall": recall_out, "judged": judged}
