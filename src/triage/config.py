import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    k: int
    seed: int
    data_dir: Path
    results_dir: Path
    system_model: str
    judge_model: str
    annotator_models: list[str]
    embed_model: str
    contact: str
    judge_fell_back: bool = False


def load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env reader. Values already in the environment win. The file is gitignored."""
    path = Path(path)
    if os.environ.get("TRIAGE_NO_DOTENV") or not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)


def _judge_model() -> tuple[str, bool]:
    """A Claude judge needs an Anthropic key; without one use the OpenAI fallback model, if set."""
    model = os.environ.get("TRIAGE_JUDGE_MODEL", "")
    fallback = os.environ.get("TRIAGE_JUDGE_FALLBACK_MODEL", "")
    has_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if model.startswith("claude") and not has_key and fallback:
        print(f"note: no ANTHROPIC_API_KEY, judging with the OpenAI fallback {fallback}", file=sys.stderr)
        return fallback, True
    return model, False


def load_settings() -> Settings:
    root = Path(os.environ.get("TRIAGE_ROOT", Path.cwd()))
    judge_model, fell_back = _judge_model()
    return Settings(
        k=5,
        seed=13,
        data_dir=root / "data",
        results_dir=root / "results",
        system_model=os.environ.get("TRIAGE_SYSTEM_MODEL", ""),
        judge_model=judge_model,
        annotator_models=[m for m in os.environ.get("TRIAGE_ANNOTATOR_MODELS", "").split(",") if m],
        embed_model=os.environ.get("TRIAGE_EMBED_MODEL", ""),
        contact=os.environ.get("TRIAGE_CONTACT", ""),
        judge_fell_back=fell_back,
    )
