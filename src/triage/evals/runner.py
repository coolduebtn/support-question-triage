import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

from triage.evals import metrics as M
from triage.evals.stats import bootstrap_ci
from triage.schemas import Action, Decision, GoldItem, RunResult


class ConfigNotFrozenError(RuntimeError):
    pass


def config_hash(config: dict) -> str:
    """Covers everything that defines a configuration (prompts, snapshot/index hashes, models, thresholds, seeds).
    The git commit is recorded next to it but deliberately not hashed."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _frozen_path(results_dir: Path) -> Path:
    return Path(results_dir) / "frozen.json"


def _read_frozen(results_dir: Path) -> dict:
    p = _frozen_path(results_dir)
    return json.loads(p.read_text()) if p.exists() else {}


def _pred_label(d: Decision) -> str:
    return d.action.value if isinstance(d.action, Action) else str(d.action)


def _judge_columns(first: list[dict], items: list[GoldItem]) -> dict:
    """Judge scores only count on the subsets defined in the design: correctness where gold and prediction are
    both ANSWER, citation support on every drafted answer, follow-up targeting where both are CLARIFY."""
    out: dict = {}
    for name, key, applies in (
        ("judged_correctness", "correct", lambda g, r: g.label == Action.ANSWER and r["pred"] == "answer"),
        ("citation_support", "citation_support", lambda g, r: r["pred"] == "answer" and bool(r["draft"])),
        ("followup_targeting", "followup", lambda g, r: g.label == Action.CLARIFY and r["pred"] == "clarify"),
    ):
        vals = [r["judge"][key] for g, r in zip(items, first)
                if r["judge"] and applies(g, r) and r["judge"].get(key) is not None]
        out[name] = statistics.fmean(float(v) for v in vals) if vals else None
        out[f"{name}_n"] = len(vals)
    return out


def run_eval(policy, retriever, gold: list[GoldItem], split: str, config: dict, judge=None, repeats: int = 1,
             results_dir: Path = Path("results")) -> RunResult:
    results_dir = Path(results_dir)
    h = config_hash(config)
    if split == "test" and h not in _read_frozen(results_dir):
        raise ConfigNotFrozenError(f"config {h[:8]} has no recorded dev run; run it on dev first")
    items = [g for g in gold if g.split == split]
    if not items:
        raise ValueError(f"no gold items in split {split!r}")
    k = int(config.get("k", 5))
    clean = bool(config.get("clean_query", False))
    rows: list[dict] = []
    for repeat in range(repeats):
        for g in items:
            query = g.text if clean else (g.text_raw or g.text)
            retrieved: list = []
            error = None
            if not query.strip():
                d = Decision(action=Action.ESCALATE, reason="empty question")
            else:
                try:
                    retrieved = retriever.retrieve(query, k)
                    d = policy.decide(query, retrieved, repeat=repeat)
                except Exception as e:  # noqa: BLE001 - any failure becomes a counted error prediction
                    d = Decision(action="error", reason="policy or retrieval failed", error_detail=repr(e))
                    error = repr(e)
            pages: list[str] = []
            for r in retrieved:
                if r.chunk.page_id not in pages:
                    pages.append(r.chunk.page_id)
            verdict = None
            if judge is not None and repeat == 0 and error is None:
                verdict = judge.evaluate(g, d, retrieved)
            rows.append({"qid": g.qid, "repeat": repeat, "gold": g.label.value, "pred": _pred_label(d),
                         "reason": d.reason, "draft": d.draft, "citations": d.citations, "missing_detail": d.missing_detail,
                         "cost_usd": d.cost_usd, "latency_s": d.latency_s, "cached": d.cached, "error": error,
                         "retrieved_pages": pages, "judge": verdict})
    first = [r for r in rows if r["repeat"] == 0]
    y_true, y_pred = [r["gold"] for r in first], [r["pred"] for r in first]
    prf = M.per_class_prf(y_true, y_pred)
    answer_rows = [r for r, g in zip(first, items) if g.label == Action.ANSWER and g.evidence]
    metrics: dict = {
        "n_items": len(items),
        "macro_f1": M.macro_f1(y_true, y_pred),
        "macro_f1_ci": list(bootstrap_ci(y_true, y_pred, M.macro_f1, n=1000, seed=int(config.get("seed", 13)))),
        "per_class": prf,
        "unsafe_answer_rate": M.unsafe_answer_rate(y_true, y_pred),
        "over_ask_rate": M.over_ask_rate(y_true, y_pred),
        "over_escalation_rate": M.over_escalation_rate(y_true, y_pred),
        "n_retrieval_items": len(answer_rows),
    }
    if answer_rows:
        gold_pages = [{e.page_id for e in g.evidence} for r, g in zip(first, items) if g.label == Action.ANSWER and g.evidence]
        rm = M.retrieval_metrics(gold_pages, [r["retrieved_pages"] for r in answer_rows], k)
        metrics.update({f"hit@{k}": rm["hit"], f"recall@{k}": rm["recall"], "mrr": rm["mrr"]})
    if judge is not None:
        metrics.update(_judge_columns(first, items))
    if repeats > 1:
        metrics["macro_f1_repeats"] = [
            M.macro_f1([r["gold"] for r in rows if r["repeat"] == i], [r["pred"] for r in rows if r["repeat"] == i])
            for i in range(repeats)]
    uncached = [r["latency_s"] for r in first if not r["cached"] and not r["error"]]
    result = RunResult(
        run_id=f"{time.strftime('%Y%m%d-%H%M%S')}-{h[:8]}-{split}", config_hash=h, split=split,
        predictions=first, metrics=metrics, n_errors=sum(r["error"] is not None for r in first),
        cost_per_q=statistics.fmean(r["cost_usd"] for r in first),
        latency_p50=statistics.median(uncached) if uncached else None)
    out = results_dir / "runs" / result.run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (out / "config.json").write_text(json.dumps({"config": config, "config_hash": h, "git_commit": _git_commit(),
                                                 "split": split, "repeats": repeats}, indent=1, default=str))
    if split == "dev":
        frozen = _read_frozen(results_dir)
        frozen[h] = result.run_id
        _frozen_path(results_dir).write_text(json.dumps(frozen, indent=1))
    return result
