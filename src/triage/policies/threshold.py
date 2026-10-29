from collections.abc import Callable

from triage.evals.metrics import over_ask_rate, over_escalation_rate
from triage.schemas import Action, Decision, GoldItem, Retrieved


def top_vector_cosine(retrieved: list[Retrieved]) -> float | None:
    cos = [r.vector_cosine for r in retrieved if r.vector_cosine is not None]
    if retrieved and not cos:
        raise ValueError("retrieved results carry no vector_cosine; A+ needs it (every retriever must keep it)")
    return max(cos) if cos else None


class ThresholdRAG:
    """Approach A+: escalate when the best first-stage vector cosine is below a threshold tuned on dev."""

    def __init__(self, base, threshold: float):
        self.base, self.threshold = base, threshold
        self.name = "A+"

    def decide(self, question: str, retrieved: list[Retrieved], repeat: int = 0) -> Decision:
        top = top_vector_cosine(retrieved)
        if top is None or top < self.threshold:
            shown = "no results" if top is None else f"top cosine {top:.2f} < {self.threshold:.2f}"
            return Decision(action=Action.ESCALATE, reason=shown)
        return self.base.decide(question, retrieved, repeat=repeat)


def tune_threshold(dev_items: list[GoldItem], retriever, metric: Callable, grid: list[float], limits: dict,
                   k: int = 5) -> float:
    """Dev-only. The base policy always answers, so a threshold turns answers into escalations; no LLM calls are needed."""
    if any(g.split != "dev" for g in dev_items):
        raise ValueError("threshold tuning accepts dev items only")
    tops = []
    for g in dev_items:
        found = top_vector_cosine(retriever.retrieve(g.text_raw or g.text, k))
        tops.append(-1.0 if found is None else found)
    y_true = [g.label.value for g in dev_items]
    best: tuple[float, float] | None = None
    for t in sorted(grid):
        y_pred = [Action.ESCALATE.value if top < t else Action.ANSWER.value for top in tops]
        oa, oe = over_ask_rate(y_true, y_pred), over_escalation_rate(y_true, y_pred)
        if oa is not None and oa > limits.get("over_ask_max", 1.0):
            continue
        if oe is not None and oe > limits.get("over_escalation_max", 1.0):
            continue
        score = metric(y_true, y_pred)
        if best is None or score > best[0]:
            best = (score, t)
    if best is None:
        raise ValueError("no threshold in the grid satisfies the limits")
    return best[1]
