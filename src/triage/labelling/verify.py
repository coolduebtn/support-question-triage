import re
import unicodedata

from triage.labelling.models import Entailment
from triage.schemas import Action, Annotation, Thread


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().lower()


def _with_flag(a: Annotation, flag: str) -> Annotation:
    return a if flag in a.flags else a.model_copy(update={"flags": [*a.flags, flag]})


def verify_evidence(a: Annotation, page_texts: dict[str, str], min_quote_chars: int = 40) -> Annotation:
    """Mechanical checks only. Never changes the label; failures are flagged for review."""
    out = a
    if a.label == Action.CLARIFY and not (a.missing_detail or "").strip():
        out = _with_flag(out, "invalid_clarify")
    if a.label in (Action.ANSWER, Action.CLARIFY) and not a.evidence:
        out = _with_flag(out, "evidence_failed")
    normalised = {pid: normalise(t) for pid, t in page_texts.items()}
    for ev in a.evidence:
        quote = normalise(ev.quote)
        if len(quote) < min_quote_chars or quote not in normalised.get(ev.page_id, ""):
            out = _with_flag(out, "evidence_failed")
    return out


def check_entailment(a: Annotation, question: str, llm) -> Annotation:
    if a.label == Action.ESCALATE or not a.evidence:
        return a
    quotes = "\n".join(f'- "{e.quote}"' for e in a.evidence)
    claim = ("These quotes answer the question." if a.label == Action.ANSWER else
             f"These quotes show the docs could answer once this detail is known: {a.missing_detail}.")
    messages = [{"role": "system", "content": "Answer strictly whether the claim is supported by the quotes."},
                {"role": "user", "content": f"QUESTION:\n{question}\n\nQUOTES:\n{quotes}\n\nCLAIM: {claim}"}]
    verdict: Entailment = llm.complete(messages, response_format=Entailment).parsed
    return a if verdict.supports else _with_flag(a, "unsupported")


def flag_possible_answer(a: Annotation, question: str, tools, threshold: float = 0.6) -> Annotation:
    if a.label != Action.ESCALATE:
        return a
    if any(s.score >= threshold for s in tools.search_wiki(question)):
        return _with_flag(a, "possible_answer")
    return a


def thread_outcome_signal(t: Thread) -> Action | None:
    """Weak signal from what happened in the thread. Reported, never used to set labels."""
    if t.n_replies == 0:
        return Action.ESCALATE
    first = (t.replies_text[0] if t.replies_text else "").strip()
    if first.endswith("?") and len(first) < 200:
        return Action.CLARIFY
    return Action.ANSWER if first else None
