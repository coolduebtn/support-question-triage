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


import random
import re
from pathlib import Path

SHEET_HEADER = """# Judge check sheet

For each item decide whether the draft answer is correct compared with the reference answer: same key facts or steps,
nothing contradicted. Ignore style and length. Write `correct` or `incorrect` after the verdict field.
Leave it empty if you do not want to grade that item.
"""


def build_validation_items(rows: list[dict], gold: dict, n_real: int, n_mismatched: int, seed: int) -> list[dict]:
    """Real items are system drafts for gold ANSWER questions (graded later by the owner). Mismatched items pair a question
    with a real draft that was written for a different question, so they are wrong by construction; they are marked synthetic."""
    eligible = sorted((r for r in rows if r.get("repeat", 0) == 0 and r["pred"] == "answer" and r.get("draft")
                       and r["qid"] in gold and gold[r["qid"]].reference_answer
                       and gold[r["qid"]].label.value == "answer"), key=lambda r: r["qid"])
    rng = random.Random(seed)
    rng.shuffle(eligible)
    real = [{"qid": r["qid"], "question": gold[r["qid"]].text, "reference": gold[r["qid"]].reference_answer,
             "draft": r["draft"], "human": None, "synthetic": False} for r in eligible[:n_real]]
    synthetic = []
    for r in eligible[:n_mismatched]:
        other = rng.choice([o for o in eligible if o["qid"] != r["qid"]])
        synthetic.append({"qid": r["qid"], "question": gold[r["qid"]].text, "reference": gold[r["qid"]].reference_answer,
                          "draft": other["draft"], "human": False, "synthetic": True})
    return real + synthetic


def write_validation_sheet(items: list[dict], path: Path) -> Path:
    parts = [SHEET_HEADER]
    for it in items:
        if it["synthetic"]:
            continue
        parts.append(f"\n## {it['qid']}\n\nQUESTION:\n{it['question']}\n\nREFERENCE ANSWER:\n{it['reference']}\n\n"
                     f"DRAFT ANSWER:\n{it['draft']}\n\nowner_verdict:\n")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(parts))
    return Path(path)


def read_validation_grades(path: Path) -> dict[str, bool]:
    grades: dict[str, bool] = {}
    qid = None
    for line in Path(path).read_text().splitlines():
        if line.startswith("## "):
            qid = line[3:].strip()
        m = re.match(r"owner_verdict:\s*(correct|incorrect)\s*$", line.strip(), re.I)
        if m and qid:
            grades[qid] = m.group(1).lower() == "correct"
    return grades
