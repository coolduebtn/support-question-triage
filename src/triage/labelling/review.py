import json
import math
import random
import re
from pathlib import Path
from urllib.parse import quote_plus

from triage.labelling.agreement import agreement
from triage.labelling.annotate import annotate
from triage.labelling.models import LabelRecord
from triage.labelling.verify import check_entailment, flag_possible_answer, verify_evidence
from triage.schemas import Action, Annotation, GoldItem, Question, ReviewItem

FLAGS_FOR_REVIEW = {"evidence_failed", "unsupported", "invalid_clarify", "possible_answer"}
SEARCH_URL = "https://cwiki.apache.org/confluence/dosearchsite.action?queryString="


class SplitNotFrozen(RuntimeError):
    pass


def draft_labels(questions: list[Question], tools, annotators: list, adjudicator, checker,
                 page_texts: dict[str, str], split_path: Path) -> list[LabelRecord]:
    split_path = Path(split_path)
    if not split_path.exists():
        raise SplitNotFrozen(f"{split_path} must exist before any annotation call")
    split = json.loads(split_path.read_text())
    records = []
    for q in questions:
        anns = [verify_evidence(annotate(q, tools, llm), page_texts) for llm in annotators]
        agreed = len({a.label for a in anns}) == 1
        adjudicated = False
        if agreed:
            final = anns[0]
        else:
            final = verify_evidence(annotate(q, tools, adjudicator), page_texts)
            adjudicated = True
        final = check_entailment(final, q.text_clean, checker)
        final = flag_possible_answer(final, q.text_clean, tools)
        flags = list(final.flags) + ([] if agreed else ["disputed"])
        records.append(LabelRecord(qid=q.qid, text=q.text_clean, text_raw=q.text_raw, split=split[q.qid], annotations=anns,
                                   final=final, agreed=agreed, adjudicated=adjudicated, flags=flags))
    return records


BLIND_HEADER = """# Blind labelling sheet

Read each question and decide what an IT support person should do, using the wiki search link if needed.
Write one word on the line that starts with the label field name (it is empty at first).

- answer: the wiki already contains the answer
- clarify: the wiki could answer it, but the question lacks a key detail (version, environment, error text)
- escalate: the wiki does not cover it

Please do not look at any other file while filling this in.
"""


def build_owner_blind_sheet(questions: list[Question], n: int, seed: int, path: Path) -> Path:
    qs = sorted(questions, key=lambda q: q.qid)
    chosen = random.Random(seed).sample(qs, min(n, len(qs)))
    parts = [BLIND_HEADER]
    for q in sorted(chosen, key=lambda q: q.qid):
        parts.append(f"\n## {q.qid}\n\n{q.text_clean}\n\nwiki search: {SEARCH_URL}{quote_plus(q.text_clean.splitlines()[0][:80])}\n\nowner_label:\n")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(parts))
    return Path(path)


def read_owner_labels(path: Path) -> dict[str, Action]:
    labels: dict[str, Action] = {}
    qid = None
    for line in Path(path).read_text().splitlines():
        if line.startswith("## "):
            qid = line[3:].strip()
        m = re.match(r"owner_label:\s*(answer|clarify|escalate)\s*$", line.strip(), re.I)
        if m and qid:
            labels[qid] = Action(m.group(1).lower())
    return labels


def build_review_queue(records: list[LabelRecord], owner_blind_qids: set[str], audit_n: int = 12,
                       seed: int = 13) -> list[ReviewItem]:
    queue: list[ReviewItem] = []
    pool: list[LabelRecord] = []
    for r in sorted(records, key=lambda r: r.qid):
        if r.qid in owner_blind_qids:
            continue
        if not r.agreed:
            queue.append(ReviewItem(qid=r.qid, annotations=r.annotations, reason="disputed"))
        elif FLAGS_FOR_REVIEW & set(r.flags):
            queue.append(ReviewItem(qid=r.qid, annotations=r.annotations, reason="evidence_failed"))
        else:
            pool.append(r)
    rng = random.Random(seed)
    rng.shuffle(pool)
    audit: list[LabelRecord] = []
    for lab in Action:
        audit += [r for r in pool if r.final.label == lab][:3]
    taken = {r.qid for r in audit}
    audit += [r for r in pool if r.qid not in taken][:max(0, audit_n - len(audit))]
    queue += [ReviewItem(qid=r.qid, annotations=r.annotations, reason="audit") for r in audit[:max(audit_n, len(audit))]]
    return queue


def apply_review(records: list[LabelRecord], owner_blind: dict[str, Action], owner_review: dict[str, Action],
                 adjudicated_qids: set[str] | None = None) -> list[GoldItem]:
    adjudicated_qids = adjudicated_qids or set()
    gold = []
    for r in sorted(records, key=lambda r: r.qid):
        if r.qid in owner_blind:
            label, source = owner_blind[r.qid], "owner_blind"
        elif r.qid in owner_review:
            label, source = owner_review[r.qid], "owner_review"
        elif r.adjudicated or r.qid in adjudicated_qids:
            label, source = r.final.label, "adjudicated"
        else:
            label, source = r.final.label, "agree"
        same = label == r.final.label
        gold.append(GoldItem(qid=r.qid, text=r.text, text_raw=r.text_raw, label=label,
                             evidence=r.final.evidence if same else [],
                             missing_detail=r.final.missing_detail if same else None,
                             reference_answer=r.final.reference_answer if same else None,
                             split=r.split, source=source))
    return gold


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    denom = 1 + z * z / n
    return max(0.0, (centre - half) / denom), min(1.0, (centre + half) / denom)


def owner_agreement(owner_blind: dict[str, str], llm_labels_before_review: dict[str, str]) -> dict:
    qids = sorted(set(owner_blind) & set(llm_labels_before_review))
    if not qids:
        raise ValueError("no overlap between owner blind labels and LLM labels")
    owner = [str(Action(owner_blind[q]).value) for q in qids]
    llm = [str(Action(llm_labels_before_review[q]).value) for q in qids]
    agree = sum(a == b for a, b in zip(owner, llm))
    kappa = agreement(owner, llm)["kappa"] if len(qids) >= 2 else None
    return {"n": len(qids), "agree": agree, "rate": agree / len(qids), "ci": _wilson(agree, len(qids)), "kappa": kappa}


def render_review_sheet(queue: list[ReviewItem], records: list[LabelRecord], path: Path) -> Path:
    """Shown to the owner only after the blind labels are recorded: includes the LLM label and its evidence."""
    by_qid = {r.qid: r for r in records}
    parts = ["# Review sheet\n\nFor each item write your own label after the label field, or leave it empty to accept the drafted label.\n"]
    for item in queue:
        r = by_qid[item.qid]
        parts.append(f"\n## {item.qid}\n\nreason: {item.reason}  | flags: {', '.join(r.flags) or 'none'}\n\n{r.text}\n")
        parts.append(f"\ndrafted label: {r.final.label.value}")
        for e in r.final.evidence:
            parts.append(f"\n> quote ({e.page_id}): {e.quote}")
        if r.final.missing_detail:
            parts.append(f"\nmissing detail: {r.final.missing_detail}")
        parts.append("\n\nowner_label:\n")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(parts))
    return Path(path)
