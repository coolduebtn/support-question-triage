import os
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


def load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env reader. Values already in the environment win. The file is gitignored."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)


def load_settings() -> Settings:
    root = Path(os.environ.get("TRIAGE_ROOT", Path.cwd()))
    return Settings(
        k=5,
        seed=13,
        data_dir=root / "data",
        results_dir=root / "results",
        system_model=os.environ.get("TRIAGE_SYSTEM_MODEL", ""),
        judge_model=os.environ.get("TRIAGE_JUDGE_MODEL", ""),
        annotator_models=[m for m in os.environ.get("TRIAGE_ANNOTATOR_MODELS", "").split(",") if m],
        embed_model=os.environ.get("TRIAGE_EMBED_MODEL", ""),
        contact=os.environ.get("TRIAGE_CONTACT", ""),
    )
