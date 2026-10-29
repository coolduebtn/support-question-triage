from pathlib import Path

START, END = "<!-- ablation:start -->", "<!-- ablation:end -->"


def insert_ablation(readme: Path, table: Path) -> None:
    """Copies results/ablation.md into the README between markers, so no number is typed by hand."""
    text = Path(readme).read_text()
    if START not in text or END not in text:
        raise ValueError("README is missing the ablation markers")
    body = Path(table).read_text().strip()
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    Path(readme).write_text(f"{head}{START}\n{body}\n{END}{tail}")
