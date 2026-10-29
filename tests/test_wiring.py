import json

from triage.cli import main
from triage.schemas import Action, Chunk, Decision, Retrieved
from triage.wiring import render_decision, run_config


def test_run_config_changes_with_prompt_and_index_hash():
    base = dict(policy="B", retriever="vector", clean_query=False, k=5, system_model="m", prompt="p1",
                snapshot_hash="s", index_hash="i", threshold=None, seed=13)
    a = run_config(**base)
    assert a == run_config(**base)
    assert run_config(**{**base, "prompt": "p2"}) != a and run_config(**{**base, "index_hash": "i2"}) != a


class Retr:
    def retrieve(self, q, k):
        c = Chunk(id="c1", page_id="p1", page_title="Reset guide", heading_path=["Reset guide"], text="x", url="https://wiki/p1")
        return [Retrieved(chunk=c, score=0.9, score_kind="cosine", vector_cosine=0.9)]


class Pol:
    name = "B"

    def decide(self, q, retrieved, repeat=0):
        return Decision(action=Action.ANSWER, reason="docs cover it", citations=["c1"], draft="Run the reset target.")


def test_ask_prints_action_reason_citation_urls_and_draft():
    from triage.wiring import run_ask
    out = run_ask("How do I reset?", Retr(), Pol())
    assert "ANSWER" in out and "docs cover it" in out and "https://wiki/p1" in out and "Run the reset target." in out


def test_ask_handles_empty_input():
    from triage.wiring import run_ask
    out = run_ask("   ", Retr(), Pol())
    assert "ESCALATE" in out and "empty" in out.lower()


def test_render_decision_shows_error_decisions():
    d = Decision(action="error", reason="decision step failed", error_detail="timeout")
    assert "ERROR" in render_decision(d, []) and "timeout" in render_decision(d, [])


def test_eval_without_key_exits_2(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["eval", "--policy", "A", "--split", "dev"]) == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err
