import subprocess
from pathlib import Path

from triage.cli import main


def test_cli_unknown_subcommand_returns_nonzero(capsys):
    assert main(["nope"]) != 0


def test_cli_smoke_registered(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["smoke"]) in (0, 2)


def test_contact_not_in_repo():
    root = Path(__file__).resolve().parents[1]
    files = subprocess.run(["git", "ls-files", "src"], cwd=root, capture_output=True,
                           text=True, check=True).stdout.split()
    for f in files:
        for line in (root / f).read_text().splitlines():
            if "User-Agent" in line:
                assert "@" not in line, f"contact leaked in {f}: {line}"


def test_smoke_without_key_exits_2_with_message(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["smoke"]) == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_ingest_index_without_key_exits_2(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["ingest", "--index"]) == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_label_select_writes_questions_split_and_blind_sheet(tmp_path, monkeypatch):
    import json

    from triage.schemas import Thread
    monkeypatch.setenv("TRIAGE_ROOT", str(tmp_path))
    snap = tmp_path / "data" / "snapshot"
    snap.mkdir(parents=True)
    ts = [Thread(tid=f"t{i:03d}", subject=f"Export number {i} fails", body=f"How can I fix error {i} xq{i} zr{i} in the export?",
                 n_replies=i % 2, epoch=1, permalink="u") for i in range(12)]
    (snap / "threads.jsonl").write_text("".join(t.model_dump_json() + "\n" for t in ts))
    assert main(["label", "--select", "--n", "10", "--n-dev", "4", "--blind", "5"]) == 0
    gold = tmp_path / "data" / "gold"
    assert len((gold / "questions.jsonl").read_text().splitlines()) == 10
    split = json.loads((gold / "split.json").read_text())
    assert list(split.values()).count("dev") == 4
    assert (gold / "owner_blind.md").read_text().count("owner_label:") == 5


def test_collect_command_registered():
    from triage.cli import COMMANDS
    assert "collect" in COMMANDS


def test_label_select_refuses_to_overwrite_existing_split_without_force(tmp_path, monkeypatch, capsys):
    from triage.schemas import Thread
    monkeypatch.setenv("TRIAGE_ROOT", str(tmp_path))
    snap = tmp_path / "data" / "snapshot"
    snap.mkdir(parents=True)
    ts = [Thread(tid=f"t{i:03d}", subject=f"Export number {i} fails", body=f"How can I fix error {i} xq{i} zr{i}?",
                 n_replies=i % 2, epoch=1, permalink="u") for i in range(12)]
    (snap / "threads.jsonl").write_text("".join(t.model_dump_json() + "\n" for t in ts))
    assert main(["label", "--select", "--n", "10", "--n-dev", "4", "--blind", "5"]) == 0
    assert main(["label", "--select", "--n", "10", "--n-dev", "4", "--blind", "5"]) == 2
    assert "already exists" in capsys.readouterr().err
    blind = tmp_path / "data" / "gold" / "owner_blind.md"
    blind.write_text(blind.read_text().replace("owner_label:", "owner_label: answer", 1))
    assert main(["label", "--select", "--force", "--n", "10", "--n-dev", "4", "--blind", "5"]) == 2
    assert "owner labels" in capsys.readouterr().err
