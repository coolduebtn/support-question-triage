import subprocess
from pathlib import Path

from triage.cli import main


def test_cli_unknown_subcommand_returns_nonzero(capsys):
    assert main(["nope"]) != 0


def test_cli_smoke_stub_registered(capsys):
    assert main(["smoke"]) == 0


def test_contact_not_in_repo():
    root = Path(__file__).resolve().parents[1]
    files = subprocess.run(["git", "ls-files", "src"], cwd=root, capture_output=True,
                           text=True, check=True).stdout.split()
    for f in files:
        for line in (root / f).read_text().splitlines():
            if "User-Agent" in line:
                assert "@" not in line, f"contact leaked in {f}: {line}"
