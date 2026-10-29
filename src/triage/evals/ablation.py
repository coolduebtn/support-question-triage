from pathlib import Path

from triage.schemas import RunResult

HEADER = ("| step | split | config | macro-F1 [95% CI] | answer F1 (n) | clarify F1 (n) | escalate F1 (n) | unsafe-answer | "
          "over-ask | recall@k | judged correct | citation support | follow-up targeting | cost/q (USD) | "
          "latency p50 (s) | errors | decision | notes | run |")
SEP = "|" + "---|" * 19


def _f(v, fmt="{:.2f}") -> str:
    return "n/a" if v is None else fmt.format(v)


def append_ablation_row(result: RunResult, step: str, decision: str, notes: str = "",
                        path: Path = Path("results/ablation.md")) -> None:
    m = result.metrics
    lo, hi = m["macro_f1_ci"]
    pc = m["per_class"]
    recall = next((v for k, v in m.items() if k.startswith("recall@")), None)
    cells = [step, result.split, result.config_hash[:8], f"{m['macro_f1']:.2f} [{lo:.2f}, {hi:.2f}]",
             *[f"{pc[c]['f1']:.2f} (n={pc[c]['support']})" for c in ("answer", "clarify", "escalate")],
             _f(m["unsafe_answer_rate"]), _f(m["over_ask_rate"]), _f(recall),
             _f(m.get("judged_correctness")), _f(m.get("citation_support")), _f(m.get("followup_targeting")),
             f"{result.cost_per_q:.4f}", _f(result.latency_p50), str(result.n_errors), decision, notes, result.run_id]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(HEADER + "\n" + SEP + "\n")
    with path.open("a") as fh:
        fh.write("| " + " | ".join(cells) + " |\n")
