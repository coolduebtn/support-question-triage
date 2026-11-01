import json

import pytest

from conftest import pred, write_run  # noqa: F401
from triage.ui.data import APPROACHES, Bundle, load_bundle, verdict


def test_latest_run_per_approach_wins(tree):
    b = load_bundle(*tree, split="dev")
    assert b.predictions["A"]["q1"]["pred"] == "escalate"
    assert set(b.predictions) == {"A", "B"}


def test_approach_labels_cover_policy_retriever_and_cleaning():
    assert APPROACHES[("A", "vector", False)] == "A"
    assert APPROACHES[("B", "vector", False)] == "B"
    assert APPROACHES[("A+", "vector", False)] == "A+"


def test_question_text_is_cleaned_of_signature(tree):
    b = load_bundle(*tree, split="dev")
    assert "Bob" not in b.question_text("q1") and "reset" in b.question_text("q1")


def test_page_titles_resolve(tree):
    assert load_bundle(*tree, split="dev").page("p1") == ("Reset guide", "https://x/p1")
    assert load_bundle(*tree, split="dev").page("zzz") == ("zzz", "")


def test_qids_are_only_those_any_run_saw_sorted_by_gold(tree):
    b = load_bundle(*tree, split="dev")
    assert b.qids() == ["q1", "q2"] and b.gold("q2") == "clarify"


def test_verdict_labels():
    assert verdict({"gold": "answer", "pred": "answer", "error": None}) == "correct"
    assert verdict({"gold": "answer", "pred": "escalate", "error": None}) == "wrong"
    assert verdict({"gold": "answer", "pred": "error", "error": "429"}) == "error"
    assert verdict(None) == "not run"


def test_expected_pages_and_reference_answer_come_from_gold(tree):
    res, data = tree
    (data / "gold" / "gold.jsonl").write_text(json.dumps(
        {"qid": "q1", "label": "answer", "evidence": [{"page_id": "p1", "heading": "h", "quote": "q"}],
         "reference_answer": "Run the reset target.", "missing_detail": None}) + "\n")
    b = load_bundle(res, data, split="dev")
    assert b.expected_pages("q1") == ["p1"] and b.reference_answer("q1") == "Run the reset target."
    assert b.expected_pages("q2") == [] and b.reference_answer("q2") is None


def _gold_row(qid, label, pages=(), synthetic=False):
    return {"qid": qid, "label": label, "synthetic": synthetic,
            "evidence": [{"page_id": p, "heading": "h", "quote": "q" * 50} for p in pages]}


def test_retrieval_rows_score_hit_recall_and_mrr_on_answer_items_only(tree):
    res, data = tree
    write_run(res, "20261031-130000-eeee-dev", "A", retriever="hybrid", clean=True, preds=[
        pred("q1", "answer", "answer", retrieved_pages=["x", "p1"]), pred("q2", "clarify", "clarify")])
    (data / "gold" / "gold.jsonl").write_text(
        json.dumps(_gold_row("q1", "answer", ["p1", "p9"])) + "\n" + json.dumps(_gold_row("q2", "clarify")) + "\n")
    rows = {r["label"]: r for r in load_bundle(res, data, "dev").retrieval_rows()}
    assert rows["A + hybrid"] == {"label": "A + hybrid", "n": 1, "hit": 1.0, "recall": 0.5, "mrr": 0.5}


def test_risk_coverage_counts_answered_questions_that_should_not_have_been(tree):
    res, data = tree
    write_run(res, "20261031-130000-ffff-dev", "A", preds=[
        pred("q1", "answer", "answer", top_cosine=0.9), pred("q2", "escalate", "answer", top_cosine=0.5)])
    (data / "gold" / "gold.jsonl").write_text("")
    curve = load_bundle(res, data, "dev").risk_coverage("A")
    assert curve[0] == (0.5, 0.0) and curve[-1] == (1.0, 0.5)


def test_cause_hint_names_why_a_prediction_is_wrong(tree):
    res, data = tree
    write_run(res, "20261031-130000-gggg-dev", "B", preds=[
        pred("q1", "answer", "escalate", retrieved_pages=["zz"]), pred("q2", "clarify", "answer")])
    (data / "gold" / "gold.jsonl").write_text(
        json.dumps({**_gold_row("q1", "answer", ["p1"]), "text": "t", "split": "dev", "source": "agree", "reference_answer": None, "missing_detail": None}) + "\n"
        + json.dumps({**_gold_row("q2", "clarify"), "text": "t", "split": "dev", "source": "agree", "reference_answer": None, "missing_detail": None}) + "\n")
    b = load_bundle(res, data, "dev")
    assert b.cause("B", "q1") == "retrieval_miss" and b.cause("B", "q2") == "missed_ambiguity"
