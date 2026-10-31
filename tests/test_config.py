import pytest

from triage.config import load_settings


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "TRIAGE_JUDGE_FALLBACK_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("TRIAGE_JUDGE_MODEL", "claude-sonnet-5-5")


def test_claude_judge_without_key_falls_back_to_openai_model(monkeypatch, capsys):
    monkeypatch.setenv("TRIAGE_JUDGE_FALLBACK_MODEL", "gpt-5.4")
    s = load_settings()
    assert s.judge_model == "gpt-5.4" and s.judge_fell_back is True
    assert "gpt-5.4" in capsys.readouterr().err


def test_claude_judge_with_key_is_used(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("TRIAGE_JUDGE_FALLBACK_MODEL", "gpt-5.4")
    s = load_settings()
    assert s.judge_model == "claude-sonnet-5-5" and s.judge_fell_back is False


def test_no_key_and_no_fallback_keeps_claude_model():
    assert load_settings().judge_model == "claude-sonnet-5-5"


def test_openai_judge_is_never_replaced(monkeypatch):
    monkeypatch.setenv("TRIAGE_JUDGE_MODEL", "gpt-5.5")
    monkeypatch.setenv("TRIAGE_JUDGE_FALLBACK_MODEL", "gpt-5.4")
    assert load_settings().judge_model == "gpt-5.5"
