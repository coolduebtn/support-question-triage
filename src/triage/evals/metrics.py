from sklearn.metrics import precision_recall_fscore_support

LABELS = ["answer", "clarify", "escalate"]


def _check(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred differ in length")
    if not y_true:
        raise ValueError("no items to score")


def per_class_prf(y_true: list[str], y_pred: list[str]) -> dict[str, dict]:
    """Predictions outside LABELS (e.g. "error") are counted as misses for the gold class."""
    _check(y_true, y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    return {lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, lab in enumerate(LABELS)}


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    prf = per_class_prf(y_true, y_pred)
    return sum(v["f1"] for v in prf.values()) / len(LABELS)


def _rate(y_true, y_pred, gold_in: set[str], pred_is: str) -> float | None:
    _check(y_true, y_pred)
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if t in gold_in]
    if not pairs:
        return None
    return sum(p == pred_is for _, p in pairs) / len(pairs)


def unsafe_answer_rate(y_true, y_pred):
    return _rate(y_true, y_pred, {"clarify", "escalate"}, "answer")


def over_ask_rate(y_true, y_pred):
    return _rate(y_true, y_pred, {"answer"}, "clarify")


def over_escalation_rate(y_true, y_pred):
    return _rate(y_true, y_pred, {"answer"}, "escalate")


def retrieval_metrics(gold_pages: list[set[str]], ranked_pages: list[list[str]], k: int) -> dict[str, float]:
    if not gold_pages or len(gold_pages) != len(ranked_pages):
        raise ValueError("need equal, non-empty gold and ranked lists")
    hit = recall = mrr = 0.0
    for gold, ranked in zip(gold_pages, ranked_pages):
        top = ranked[:k]
        found = gold & set(top)
        hit += bool(found)
        recall += len(found) / len(gold)
        mrr += next((1 / (i + 1) for i, p in enumerate(top) if p in gold), 0.0)
    n = len(gold_pages)
    return {"hit": hit / n, "recall": recall / n, "mrr": mrr / n}
