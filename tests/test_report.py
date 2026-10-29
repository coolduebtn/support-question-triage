from triage.report import insert_ablation

START, END = "<!-- ablation:start -->", "<!-- ablation:end -->"


def test_insert_ablation_replaces_between_markers(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"# T\n{START}\nold\n{END}\ntail\n")
    table = tmp_path / "ablation.md"
    table.write_text("| step |\n|---|\n| A |\n")
    insert_ablation(readme, table)
    text = readme.read_text()
    assert "| A |" in text and "old" not in text and text.endswith("tail\n")
    insert_ablation(readme, table)
    assert readme.read_text().count("| A |") == 1


def test_insert_ablation_requires_markers_and_table(tmp_path):
    import pytest
    readme = tmp_path / "README.md"
    readme.write_text("no markers")
    table = tmp_path / "ablation.md"
    table.write_text("x")
    with pytest.raises(ValueError):
        insert_ablation(readme, table)
    readme.write_text(f"{START}\n{END}")
    with pytest.raises(FileNotFoundError):
        insert_ablation(readme, tmp_path / "missing.md")
