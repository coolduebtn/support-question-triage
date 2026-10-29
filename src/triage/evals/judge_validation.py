from sklearn.metrics import cohen_kappa_score

from triage.evals.stats import bootstrap_ci


def _kappa(human: list[str], judge: list[str]) -> float | None:
    if len(set(human) | set(judge)) < 2:
        return None
    return float(cohen_kappa_score(human, judge))


def _summ(human: list[bool], judge: list[bool], seed: int) -> dict:
    if not human:
        return {"n": 0, "agreement": None, "kappa": None, "kappa_ci": None, "confusion": {}}
    h, j = [str(x) for x in human], [str(x) for x in judge]
    ci = None
    try:
        ci = bootstrap_ci(h, j, lambda t, p: _kappa(t, p), n=1000, seed=seed)
    except ValueError:
        pass
    conf = {"tp": sum(a and b for a, b in zip(human, judge)), "tn": sum(not a and not b for a, b in zip(human, judge)),
            "fp": sum(not a and b for a, b in zip(human, judge)), "fn": sum(a and not b for a, b in zip(human, judge))}
    return {"n": len(human), "agreement": sum(a == b for a, b in zip(human, judge)) / len(human),
            "kappa": _kappa(h, j), "kappa_ci": ci, "confusion": conf}


def validate_judge(judge, items: list[dict], rubric: str = "answer_correct", seed: int = 13) -> dict:
    """Compare the judge with owner-graded items. Synthetic (deliberately wrong) items are reported separately."""
    verdicts: list[bool] = []
    for it in items:
        if rubric == "answer_correct":
            verdicts.append(judge.answer_correct(it["question"], it["reference"], it["draft"]).passed)
        elif rubric == "followup":
            verdicts.append(judge.followup_targets_missing(it["draft"], it["missing_detail"]).passed)
        else:
            raise ValueError(f"unknown rubric {rubric}")
    human = [bool(it["human"]) for it in items]
    syn = [bool(it.get("synthetic")) for it in items]
    pick = lambda flag: ([h for h, s in zip(human, syn) if s == flag], [v for v, s in zip(verdicts, syn) if s == flag])  # noqa: E731
    return {"all": _summ(human, verdicts, seed), "real": _summ(*pick(False), seed), "synthetic": _summ(*pick(True), seed)}
