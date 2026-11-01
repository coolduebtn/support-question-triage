import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from triage.clean import clean_email
from triage.evals.analysis import cause_hint, risk_coverage
from triage.schemas import GoldItem, RunResult

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

    def retrieval_rows(self) -> list[dict]:
        """Page-level hit, recall and MRR of each retrieval variant (policy A) on the answerable questions."""
        order = ["A", "A + cleaning", "A + hybrid", "A + reranker"]
        items = {q: [e["page_id"] for e in g["evidence"]] for q, g in self._gold.items()
                 if g["label"] == "answer" and g.get("evidence")}
        out = []
        for label in order:
            rows = self.predictions.get(label, {})
            qs = [q for q in items if q in rows]
            if not qs:
                continue
            hit = rec = mrr = 0.0
            for q in qs:
                gold_pages, pages = set(items[q]), rows[q]["retrieved_pages"]
                ranks = [i + 1 for i, p in enumerate(pages) if p in gold_pages]
                hit += bool(ranks)
                mrr += 1 / ranks[0] if ranks else 0.0
                rec += len(gold_pages & set(pages)) / len(gold_pages)
            out.append({"label": label, "n": len(qs), "hit": hit / len(qs), "recall": rec / len(qs), "mrr": mrr / len(qs)})
        return out

    def risk_coverage(self, label: str) -> list[tuple[float, float]]:
        rows = list(self.predictions.get(label, {}).values())
        if not rows:
            return []
        return risk_coverage(RunResult(run_id=label, config_hash="", split="", predictions=rows, metrics={},
                                       n_errors=0, cost_per_q=0.0, latency_p50=None))

    def cause(self, label: str, qid: str) -> str:
        row, g = self.predictions.get(label, {}).get(qid), self._gold.get(qid)
        if row is None or g is None:
            return "unknown"
        return cause_hint(GoldItem.model_validate(g), row)

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


def _key_in_env_file(path: Path) -> bool:
    if not path.exists():
        return False
    for line in path.read_text().splitlines():
        name, _, value = line.partition("=")
        if name.strip() == "OPENAI_API_KEY" and value.strip().strip("\"'"):
            return True
    return False


def live_problem(root: Path) -> str | None:
    """Why the 'Try your own' tab cannot run yet, in words a reviewer can act on; None when it can."""
    root = Path(root)
    has_key = bool(os.environ.get("OPENAI_API_KEY")) or (
        not os.environ.get("TRIAGE_NO_DOTENV") and _key_in_env_file(root / ".env"))
    if not has_key:
        return ("This tab needs an OpenAI API key, and none was found. Add a line `OPENAI_API_KEY=...` to a `.env` file "
                "in the project folder (keep it private and never commit it), then reload the page. "
                "Every other tab works without a key.")
    if not (root / "data" / "index" / "chroma").exists():
        return ("The documentation search index has not been built yet. Run `uv run triage ingest --index` "
                "(it needs the documentation snapshot), then reload the page.")
    return None


def explain_failure(stderr: str) -> str:
    """A plain-language reason for a failed `triage ask`, from its error output."""
    low = (stderr or "").lower()
    if "429" in low or "ratelimit" in low or "insufficient_quota" in low or "spend limit" in low:
        return "OpenAI refused the request because the key's rate or spend limit was reached. Wait a moment or raise the limit, then try again."
    if "401" in low or "incorrect api key" in low or "authentication" in low:
        return "OpenAI rejected the API key. Check the value of OPENAI_API_KEY in your .env file."
    if "openai_api_key" in low:
        return "No OpenAI API key was available to the command. Add OPENAI_API_KEY to your .env file."
    return "The triage command failed. Details below."
