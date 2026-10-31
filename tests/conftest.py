import json

import pytest


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    """Tests must never read the developer's real .env (it holds a live API key)."""
    monkeypatch.setenv("TRIAGE_NO_DOTENV", "1")


def write_run(root, run_id, policy, retriever="vector", clean=False, preds=(), n_errors=0):
    d = root / "runs" / run_id
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"config": {"policy": policy, "retriever": retriever,
                                                            "clean_query": clean}, "config_hash": run_id[-4:]}))
    (d / "metrics.json").write_text(json.dumps({"macro_f1": 0.5, "n_items": len(preds)}))
    (d / "predictions.jsonl").write_text("".join(json.dumps(p) + "\n" for p in preds))


def pred(qid, gold, p, **kw):
    return {"qid": qid, "repeat": 0, "gold": gold, "pred": p, "reason": "r", "draft": None, "citations": [],
            "missing_detail": None, "cost_usd": 0.001, "latency_s": 1.0, "error": None,
            "retrieved_pages": ["p1"], "judge": {}, **kw}


@pytest.fixture
def tree(tmp_path):
    res, data = tmp_path / "results", tmp_path / "data"
    write_run(res, "20261031-100000-aaaa-dev", "A", preds=[pred("q1", "answer", "answer")])
    write_run(res, "20261031-110000-bbbb-dev", "A", preds=[pred("q1", "answer", "escalate")])   # newer wins
    write_run(res, "20261031-120000-cccc-dev", "B", preds=[pred("q1", "answer", "answer"), pred("q2", "clarify", "clarify")])
    (data / "gold").mkdir(parents=True)
    (data / "index").mkdir()
    (data / "gold" / "questions.jsonl").write_text(
        json.dumps({"qid": "q1", "text_raw": "Hello,\n\nHow do I reset the thing in ofbiz?\n\nRegards,\nBob"}) + "\n"
        + json.dumps({"qid": "q2", "text_raw": "It fails."}) + "\n")
    (data / "index" / "chunks.jsonl").write_text(json.dumps(
        {"id": "p1-0", "page_id": "p1", "page_title": "Reset guide", "url": "https://x/p1", "text": "t",
         "heading_path": ["Reset guide"]}) + "\n")
    return res, data
