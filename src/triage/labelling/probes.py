import random
import re
from collections import Counter

from pydantic import BaseModel

from triage.schemas import Action, GoldItem

SYSTEM = """You make a vaguer version of a real support question by DELETING words only.
Delete the specific details (exact error text, version numbers, entity / service / screen names, steps already tried)
so that the remaining question would need a follow-up before anyone could answer it. Rules:
- Only delete. Never add, reorder, replace or rephrase any word.
- Keep what the question is about, so it still reads like a real, short, vague support email.
- removed_detail: name in one short phrase what the reader would now have to ask for."""


class ProbeOut(BaseModel):
    degraded_text: str
    removed_detail: str


def _words(text: str) -> Counter:
    return Counter(re.findall(r"\w+", text.lower()))


def is_deletion_only(original: str, degraded: str) -> bool:
    d, o = _words(degraded), _words(original)
    return bool(d) and all(o[w] >= n for w, n in d.items())


def make_probe(item: GoldItem, llm) -> GoldItem | None:
    """A CLARIFY probe made by deleting words from a real ANSWER question. Rejected unless deletion-only."""
    out: ProbeOut = llm.complete([{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": f"REAL QUESTION:\n{item.text}"}],
                                 response_format=ProbeOut).parsed
    text = out.degraded_text.strip()
    if not out.removed_detail.strip() or not is_deletion_only(item.text, text) or _words(text) == _words(item.text):
        return None
    return GoldItem(qid=f"{item.qid}-probe", text=text, text_raw=text, label=Action.CLARIFY, evidence=item.evidence,
                    missing_detail=out.removed_detail.strip(), reference_answer=None, split=item.split,
                    source="synthetic_probe", synthetic=True, derived_from=item.qid)


def probes_from_gold(items: list[GoldItem], llm, n: int, seed: int) -> list[GoldItem]:
    eligible = sorted((g for g in items if not g.synthetic and g.label == Action.ANSWER), key=lambda g: g.qid)
    random.Random(seed).shuffle(eligible)
    probes: list[GoldItem] = []
    for g in eligible:
        if len(probes) >= n:
            break
        p = make_probe(g, llm)
        if p is not None:
            probes.append(p)
    return probes
