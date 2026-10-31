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
