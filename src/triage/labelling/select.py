import json
import random
import re
from pathlib import Path

from triage.clean import clean_email
from triage.schemas import Question, Thread
from triage.sources.mailing_list import is_noise

MAX_CHARS = 6000
DUP_JACCARD = 0.8


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


NOT_SUPPORT = re.compile(
    r"^(proposal|looking for|community over code|travel assistance|job|hiring|welcome|introduc|"
    r"want to become|documentation highlight|\[?(announce|announcement|discuss|proposal)|participate in|"
    r"attempting to join|.*new contributor|step-by-step guide|try apache|.*landing-page)|"
    r"travel assistance|we are hiring|expert(ise)? (in|needed)|anniversary campaign",
    re.I)
QUESTION_CUES = re.compile(
    r"\?|\b(how|why|what|where|which|can i|could i|cannot|can't|unable|error|exception|fail(s|ed|ing)?|"
    r"not working|doesn't|does not|issue|problem|help|stuck|trying to|need to|want to understand)\b", re.I)


def looks_like_question(text: str) -> bool:
    """Cheap pre-filter so job ads, proposals and announcements are not labelled as support questions."""
    subject = text.splitlines()[0] if text else ""
    return not NOT_SUPPORT.search(subject) and bool(QUESTION_CUES.search(text))


def select_candidates(threads: list[Thread], n_target: int, seed: int) -> list[Question]:
    replied: list[tuple[Question, set[str]]] = []
    unreplied: list[tuple[Question, set[str]]] = []
    for t in sorted(threads, key=lambda t: t.tid):
        if is_noise(t.subject, t.body):
            continue
        cleaned = clean_email(t.body)
        if not cleaned and len(t.subject.split()) < 4:
            continue
        text = f"{t.subject}\n\n{cleaned}" if cleaned else t.subject
        if not looks_like_question(text):
            continue
        if len(text) > MAX_CHARS:
            continue
        question = Question(qid=f"q_{t.tid[:10]}", thread_id=t.tid, text_raw=f"{t.subject}\n\n{t.body}",
                            text_clean=text, permalink=t.permalink)
        (replied if t.n_replies > 0 else unreplied).append((question, _words(text)))
    rng = random.Random(seed)
    rng.shuffle(replied)
    rng.shuffle(unreplied)
    chosen: list[tuple[Question, set[str]]] = []
    pools = [replied, unreplied]
    while len(chosen) < n_target and any(pools):
        for pool in pools:
            while pool:
                cand = pool.pop()
                if all(len(cand[1] & w) / max(1, len(cand[1] | w)) < DUP_JACCARD for _, w in chosen):
                    chosen.append(cand)
                    break
            if len(chosen) >= n_target:
                break
    return [q for q, _ in chosen]


def assign_split(questions: list[Question], n_dev: int, seed: int, path: Path | None = None) -> dict[str, str]:
    qids = sorted(q.qid for q in questions)
    random.Random(seed).shuffle(qids)
    mapping = {qid: ("dev" if i < n_dev else "test") for i, qid in enumerate(qids)}
    mapping = dict(sorted(mapping.items()))
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(mapping, indent=1))
    return mapping
