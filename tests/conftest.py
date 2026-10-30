import pytest


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    """Tests must never read the developer's real .env (it holds a live API key)."""
    monkeypatch.setenv("TRIAGE_NO_DOTENV", "1")
