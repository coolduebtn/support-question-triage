import json
from dataclasses import dataclass, field
from pathlib import Path

from triage.clean import clean_email

APPROACHES = {
    ("A", "vector", False): "A",
    ("A+", "vector", False): "A+",
    ("B", "vector", False): "B",
    ("A", "vector", True): "A + cleaning",
    ("A", "hybrid", True): "A + hybrid",
    ("A", "rerank-minilm", True): "A + reranker",
}


def label_for(policy: str, retriever: str, clean: bool) -> str:
    return APPROACHES.get((policy, retriever, clean)) or f"{policy} {retriever}{' clean' if clean else ''}"


def verdict(row: dict | None) -> str:
    if row is None:
        return "not run"
    if row.get("error") or row["pred"] == "error":
        return "error"
    return "correct" if row["pred"] == row["gold"] else "wrong"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


@dataclass
class Bundle:
    predictions: dict[str, dict[str, dict]] = field(default_factory=dict)   # approach -> qid -> row
    metrics: dict[str, dict] = field(default_factory=dict)
    run_ids: dict[str, str] = field(default_factory=dict)
    _text: dict[str, str] = field(default_factory=dict)
    _pages: dict[str, tuple[str, str]] = field(default_factory=dict)
    _gold: dict[str, dict] = field(default_factory=dict)

    def qids(self) -> list[str]:
        seen = {q for rows in self.predictions.values() for q in rows}
        return sorted(seen, key=lambda q: (self.gold(q) or "", q))

    def gold(self, qid: str) -> str | None:
        for rows in self.predictions.values():
            if qid in rows:
                return rows[qid]["gold"]
        return None

    def question_text(self, qid: str) -> str:
        return clean_email(self._text.get(qid) or self._gold.get(qid, {}).get("text_raw", ""))

    def expected_pages(self, qid: str) -> list[str]:
        return list(dict.fromkeys(e["page_id"] for e in self._gold.get(qid, {}).get("evidence", [])))

    def reference_answer(self, qid: str) -> str | None:
        return self._gold.get(qid, {}).get("reference_answer")

    def page(self, page_id: str) -> tuple[str, str]:
        return self._pages.get(page_id, (page_id, ""))


def load_bundle(results_dir: Path, data_dir: Path, split: str = "dev") -> Bundle:
    b = Bundle()
    for run in sorted((Path(results_dir) / "runs").glob(f"*-{split}")):      # run ids sort by time, newest last
        cfg = json.loads((run / "config.json").read_text())["config"]
        label = label_for(cfg["policy"], cfg["retriever"], bool(cfg.get("clean_query")))
        if cfg.get("synthetic", "none") == "only":
            continue
        rows = {}
        for p in _jsonl(run / "predictions.jsonl"):
            if p.get("repeat", 0) == 0:
                rows[p["qid"]] = p
        b.predictions[label], b.run_ids[label] = rows, run.name
        b.metrics[label] = json.loads((run / "metrics.json").read_text())
    b._text = {q["qid"]: q["text_raw"] for q in _jsonl(Path(data_dir) / "gold" / "questions.jsonl")}
    b._pages = {c["page_id"]: (c["page_title"], c["url"]) for c in _jsonl(Path(data_dir) / "index" / "chunks.jsonl")}
    b._gold = {g["qid"]: g for g in _jsonl(Path(data_dir) / "gold" / "gold.jsonl")}
    return b
