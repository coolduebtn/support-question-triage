import pytest

from triage.evals.metrics import macro_f1
from triage.policies.threshold import ThresholdRAG, tune_threshold
from triage.schemas import Action, Chunk, Decision, GoldItem, Retrieved


def r(cos, kind="cosine", score=None):
    c = Chunk(id="c", page_id="p", page_title="t", heading_path=["t"], text="x", url="u")
    return Retrieved(chunk=c, score=cos if score is None else score, score_kind=kind, vector_cosine=cos)


class Base:
    name = "A"

    def decide(self, question, retrieved, repeat=0):
        return Decision(action=Action.ANSWER, draft="d", citations=["c"])


def test_escalates_below_threshold():
    d = ThresholdRAG(Base(), 0.5).decide("q", [r(0.3)])
    assert d.action == Action.ESCALATE and "0.30" in d.reason


def test_delegates_at_or_above_threshold():
    assert ThresholdRAG(Base(), 0.5).decide("q", [r(0.5)]).action == Action.ANSWER


def test_empty_retrieval_escalates():
    assert ThresholdRAG(Base(), 0.1).decide("q", []).action == Action.ESCALATE


def test_uses_vector_cosine_even_when_primary_score_is_rrf():
    results = [r(0.2, kind="rrf", score=0.03), r(0.8, kind="rrf", score=0.01)]
    assert ThresholdRAG(Base(), 0.5).decide("q", results).action == Action.ANSWER


def test_missing_vector_cosine_everywhere_is_an_error_not_a_silent_answer():
    c = Chunk(id="c", page_id="p", page_title="t", heading_path=["t"], text="x", url="u")
    with pytest.raises(ValueError):
        ThresholdRAG(Base(), 0.5).decide("q", [Retrieved(chunk=c, score=1, score_kind="bm25")])


def gold(i, label, split="dev", text=None):
    return GoldItem(qid=f"q{i}", text=text or f"q{i}", label=label, evidence=[], missing_detail=None,
                    reference_answer=None, split=split, source="agree")


class Retr:
    cos = {"q1": 0.9, "q2": 0.85, "q3": 0.2, "q4": 0.25, "q5": 0.7, "q6": 0.1}

    def retrieve(self, query, k):
        return [r(self.cos[query])]


DEV = [gold(1, Action.ANSWER), gold(2, Action.ANSWER), gold(3, Action.ESCALATE), gold(4, Action.ESCALATE),
       gold(5, Action.CLARIFY), gold(6, Action.ESCALATE)]


def test_tune_picks_best_in_grid():
    t = tune_threshold(DEV, Retr(), macro_f1, [0.05, 0.5, 0.95], limits={})
    assert t == 0.5


def test_tune_respects_over_escalation_limit():
    t = tune_threshold(DEV, Retr(), macro_f1, [0.5, 0.95], limits={"over_escalation_max": 0.0})
    assert t == 0.5


def test_tune_rejects_test_items():
    with pytest.raises(ValueError):
        tune_threshold(DEV + [gold(7, Action.ANSWER, split="test", text="q1")], Retr(), macro_f1, [0.5], limits={})


def test_tune_raises_when_no_threshold_meets_limits():
    with pytest.raises(ValueError):
        tune_threshold(DEV, Retr(), macro_f1, [0.95], limits={"over_escalation_max": 0.0})
