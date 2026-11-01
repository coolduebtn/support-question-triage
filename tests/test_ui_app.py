import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from conftest import pred, write_run

APP = str(Path(__file__).parent.parent / "src" / "triage" / "ui" / "app.py")


def test_app_renders_compare_and_scoreboard_from_saved_runs(tree, monkeypatch):
    res, data = tree
    write_run(res, "20261031-130000-dddd-dev", "A+", preds=[pred("q1", "answer", "answer")])
    monkeypatch.chdir(res.parent)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert len(at.tabs) == 4
    assert any("Question" in s.value for s in at.selectbox) or at.selectbox


def test_app_without_runs_shows_a_helpful_error(tmp_path, monkeypatch):
    (tmp_path / "results" / "runs").mkdir(parents=True)
    (tmp_path / "data").mkdir()
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception and "No saved dev runs" in at.error[0].value


def test_scoreboard_shows_retrieval_and_risk_coverage_sections(tree, monkeypatch):
    res, data = tree
    (data / "gold" / "gold.jsonl").write_text(json.dumps(
        {"qid": "q1", "text": "t", "label": "answer", "synthetic": False, "split": "dev", "source": "agree",
         "reference_answer": None, "missing_detail": None,
         "evidence": [{"page_id": "p1", "heading": "h", "quote": "q" * 50}]}) + "\n")
    monkeypatch.chdir(res.parent)
    at = AppTest.from_file(APP, default_timeout=30).run()
    text = " ".join(m.value for m in at.markdown)
    assert not at.exception and "Retrieval variants" in text and "Risk and coverage" in text


def test_test_split_notes_that_only_the_first_repeat_is_shown(tree, monkeypatch):
    res, data = tree
    write_run(res, "20261101-130000-tttt-test", "B", preds=[pred("q1", "answer", "answer")])
    monkeypatch.chdir(res.parent)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.selectbox[0].select("test").run()
    assert not at.exception and any("first repeat" in i.value for i in at.info)


def _ready_for_live(monkeypatch, res):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    (res.parent / "data" / "index" / "chroma").mkdir(parents=True)
    monkeypatch.chdir(res.parent)


def test_try_your_own_runs_ask_and_shows_its_output(tree, monkeypatch):
    import subprocess
    from types import SimpleNamespace

    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return SimpleNamespace(stdout="ACTION: ESCALATE\nREASON: test", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    res, data = tree
    _ready_for_live(monkeypatch, res)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_area[0].input("The export is failing.").run()
    at.button[0].click().run()
    assert not at.exception
    assert seen["cmd"][-3:-1] == ["triage.cli", "ask"] and seen["cmd"][-1] == "The export is failing."
    assert any("ACTION: ESCALATE" in c.value for c in at.code)


def test_try_your_own_explains_a_missing_key_and_disables_the_button(tree, monkeypatch):
    res, data = tree
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(res.parent)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_area[0].input("The export is failing.").run()
    assert not at.exception
    assert any("OPENAI_API_KEY" in w.value and "none was found" in w.value for w in at.warning)
    assert at.button[0].disabled


def test_try_your_own_explains_a_failed_call_in_plain_words(tree, monkeypatch):
    import subprocess
    from types import SimpleNamespace

    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: SimpleNamespace(
        returncode=1, stdout="", stderr="openai.RateLimitError: Error code: 429 insufficient_quota"))
    res, data = tree
    _ready_for_live(monkeypatch, res)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_area[0].input("The export is failing.").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("spend limit" in e.value for e in at.error)
