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
