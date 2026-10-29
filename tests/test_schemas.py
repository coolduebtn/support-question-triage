import pytest
from pydantic import ValidationError

from triage.config import load_settings
from triage.schemas import ACTIONS, Action, Decision, GoldItem


def test_actions_order():
    assert ACTIONS == ["answer", "clarify", "escalate"]
    assert Action.ANSWER == "answer"


def test_gold_item_rejects_unknown_label():
    with pytest.raises(ValidationError):
        GoldItem(qid="q1", text="t", label="maybe", evidence=[], missing_detail=None,
                 reference_answer=None, split="dev", source="agree")


def test_gold_item_rejects_unknown_source():
    with pytest.raises(ValidationError):
        GoldItem(qid="q1", text="t", label="answer", evidence=[], missing_detail=None,
                 reference_answer=None, split="dev", source="owner")


def test_decision_allows_error_action():
    d = Decision(action="error", reason="api failed", error_detail="timeout")
    assert d.action == "error"
    assert d.citations == []


def test_load_settings_reads_env(monkeypatch):
    monkeypatch.setenv("TRIAGE_SYSTEM_MODEL", "x")
    monkeypatch.setenv("TRIAGE_CONTACT", "me@example.org")
    s = load_settings()
    assert s.system_model == "x"
    assert s.contact == "me@example.org"
    assert s.k == 5 and s.seed == 13
