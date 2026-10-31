import pytest

from triage.evals.judge import ClaimsOut, ClaimVerdict, Judge, VerdictOut, assert_judge_differs
from triage.evals.judge_validation import validate_judge
from triage.evals.runner import run_eval
from triage.llm import FakeLLM
from triage.schemas import Action, Chunk, Decision, GoldItem, Retrieved


def test_parses_verdict():
    j = Judge(FakeLLM([VerdictOut(passed=True, reason="matches reference")]))
    v = j.answer_correct("q", "ref", "draft")
    assert v.passed is True and v.reason == "matches reference" and not v.error


def test_fake_llm_raises_scripted_exception():
    with pytest.raises(RuntimeError):
        FakeLLM([RuntimeError("boom")]).complete([])


def test_invalid_output_retried_then_error_verdict():
    llm = FakeLLM([RuntimeError("bad json"), RuntimeError("bad json")])
    v = Judge(llm, attempts=2).answer_correct("q", "ref", "draft")
    assert v.error and v.passed is False and len(llm.calls) == 2


def test_invalid_output_then_success_on_retry():
    llm = FakeLLM([RuntimeError("bad json"), VerdictOut(passed=True, reason="ok")])
    assert Judge(llm, attempts=2).answer_correct("q", "ref", "draft").passed is True


def test_citation_supported_returns_one_verdict_per_claim():
    chunk = Chunk(id="c1", page_id="p", page_title="t", heading_path=["t"], text="Run the reset target.", url="u")
    out = ClaimsOut(claims=[ClaimVerdict(claim="Run reset", supported=True, reason="x"),
                            ClaimVerdict(claim="Restart twice", supported=False, reason="not in docs")])
    verdicts = Judge(FakeLLM([out])).citation_supported("Run reset. Restart twice.", [chunk])
    assert [v.passed for v in verdicts] == [True, False]


def test_followup_targets_missing():
    j = Judge(FakeLLM([VerdictOut(passed=True, reason="asks for version")]))
    assert j.followup_targets_missing("Which version?", "version").passed


def test_validate_reports_agreement_kappa_ci_and_n_split_by_synthetic():
    humans = [True] * 10 + [False] * 10
    judge_says = [True] * 8 + [False] * 2 + [False] * 8 + [True] * 2
    items = [{"question": "q", "reference": "r", "draft": f"d{i}", "human": h, "synthetic": i >= 15}
             for i, h in enumerate(humans)]
    llm = FakeLLM([VerdictOut(passed=s, reason="") for s in judge_says])
    out = validate_judge(Judge(llm), items, rubric="answer_correct", seed=1)
    assert out["all"]["n"] == 20 and out["all"]["agreement"] == pytest.approx(0.8)
    assert out["all"]["kappa"] == pytest.approx(0.6)
    lo, hi = out["all"]["kappa_ci"]
    assert lo <= 0.6 <= hi
    assert out["real"]["n"] == 15 and out["synthetic"]["n"] == 5


def test_judge_model_is_not_the_system_model():
    assert_judge_differs("gpt-a", "gpt-b")
    with pytest.raises(ValueError):
        assert_judge_differs("gpt-a", "gpt-a")


class FakeJudge:
    def evaluate(self, item, decision, retrieved):
        return {"correct": True if item.label == Action.ANSWER else None, "citation_support": 0.5,
                "followup": True if item.label == Action.CLARIFY else None}


class Retr:
    def retrieve(self, query, k):
        c = Chunk(id="c0", page_id="p1", page_title="t", heading_path=["t"], text="x", url="u")
        return [Retrieved(chunk=c, score=1, score_kind="cosine", vector_cosine=1)]


class Pol:
    name = "p"

    def decide(self, question, retrieved, repeat=0):
        return Decision(action=Action.ANSWER, draft="d", citations=["c0"])


def test_run_eval_records_judge_columns(tmp_path):
    gold = [GoldItem(qid=f"q{i}", text=f"t{i}", label=lab, evidence=[], missing_detail=None, reference_answer="r",
                     split="dev", source="agree") for i, lab in enumerate([Action.ANSWER, Action.ANSWER, Action.CLARIFY])]
    res = run_eval(Pol(), Retr(), gold, "dev", {"k": 5}, judge=FakeJudge(), results_dir=tmp_path)
    m = res.metrics
    assert m["judged_correctness"] == 1.0 and m["judged_correctness_n"] == 2
    assert m["citation_support"] == 0.5 and m["citation_support_n"] == 3
    assert m["followup_targeting"] is None and m["followup_targeting_n"] == 0


def test_judge_errors_are_excluded_not_counted_as_failures():
    gold = GoldItem(qid="q", text="t", label=Action.ANSWER, evidence=[], missing_detail=None, reference_answer="r",
                    split="dev", source="agree")
    j = Judge(FakeLLM([RuntimeError("x"), RuntimeError("x"), RuntimeError("x"), RuntimeError("x")]), attempts=2)
    out = j.evaluate(gold, Decision(action=Action.ANSWER, draft="d", citations=["c0"]), [])
    assert out["correct"] is None and out["citation_support"] is None


def _gold_item(i, label=Action.ANSWER):
    return GoldItem(qid=f"q{i}", text=f"question {i}", label=label, evidence=[], missing_detail=None,
                    reference_answer=f"reference {i}" if label == Action.ANSWER else None, split="dev", source="agree")


def test_build_validation_items_uses_real_drafts_and_marks_mismatched_ones_synthetic():
    from triage.evals.judge_validation import build_validation_items
    gold = {f"q{i}": _gold_item(i) for i in range(8)}
    rows = [{"qid": f"q{i}", "pred": "answer", "draft": f"draft {i}", "repeat": 0} for i in range(8)]
    items = build_validation_items(rows, gold, n_real=5, n_mismatched=2, seed=1)
    real = [x for x in items if not x["synthetic"]]
    syn = [x for x in items if x["synthetic"]]
    assert len(real) == 5 and len(syn) == 2
    assert all(x["draft"] == f"draft {x['qid'][1:]}" and x["reference"] == f"reference {x['qid'][1:]}" for x in real)
    for x in syn:  # a real draft written for a different question
        assert x["draft"] != f"draft {x['qid'][1:]}" and x["draft"].startswith("draft ") and x["human"] is False


def test_validation_sheet_roundtrip_and_blank_means_ungraded(tmp_path):
    from triage.evals.judge_validation import build_validation_items, read_validation_grades, write_validation_sheet
    gold = {f"q{i}": _gold_item(i) for i in range(6)}
    rows = [{"qid": f"q{i}", "pred": "answer", "draft": f"draft {i}", "repeat": 0} for i in range(6)]
    items = build_validation_items(rows, gold, n_real=4, n_mismatched=0, seed=2)
    path = write_validation_sheet(items, tmp_path / "sheet.md")
    text = path.read_text()
    assert "synthetic" not in text.lower() and "mismatch" not in text.lower()
    assert text.count("owner_verdict:") == 4
    path.write_text(text.replace("owner_verdict:", "owner_verdict: correct", 2).replace("owner_verdict:\n", "owner_verdict: incorrect\n", 1))
    grades = read_validation_grades(path)
    assert sorted(grades.values()).count(True) == 2 and sorted(grades.values()).count(False) == 1 and len(grades) == 3
