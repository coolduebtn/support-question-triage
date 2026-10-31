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
