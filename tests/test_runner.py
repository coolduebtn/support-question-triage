import json

import pytest

from triage.evals.ablation import append_ablation_row
from triage.evals.runner import ConfigNotFrozenError, config_hash, run_eval
from triage.llm import FakeLLM
from triage.policies.vanilla import VanillaRAG
from triage.schemas import Action, Chunk, Decision, Evidence, GoldItem, Retrieved


def chunk(i, page):
    return Chunk(id=f"c{i}", page_id=page, page_title=page, heading_path=[page], text=f"text {i}", url=f"u{i}")


def gold(i, label, split="dev", page=None, text=None):
    ev = [Evidence(page_id=page, heading="H", quote="q" * 40)] if page else []
    return GoldItem(qid=f"q{i}", text=text or f"question {i}", label=label, evidence=ev, missing_detail=None,
                    reference_answer=None, split=split, source="agree")


class Retr:
    def __init__(self, pages=("p1",)):
        self.pages = pages

    def retrieve(self, query, k):
        return [Retrieved(chunk=chunk(i, p), score=1.0, score_kind="cosine", vector_cosine=1.0)
                for i, p in enumerate(self.pages[:k])]


class Pol:
    name = "pol"

    def __init__(self, fn=None):
        self.fn, self.repeats = fn, []

    def decide(self, question, retrieved, repeat=0):
        self.repeats.append(repeat)
        if self.fn:
            return self.fn(question)
        return Decision(action=Action.ANSWER, draft="d", citations=["c0"], cost_usd=0.01, latency_s=0.5)


CFG = {"policy": "pol", "retriever": "vec", "k": 5, "prompt": "v1"}
GOLD = [gold(1, Action.ANSWER, page="p1"), gold(2, Action.CLARIFY), gold(3, Action.ESCALATE), gold(4, Action.ANSWER, page="p9")]


def test_vanilla_always_answers():
    llm = FakeLLM(["Here is the answer."])
    d = VanillaRAG(llm, "sys").decide("how?", [Retrieved(chunk=chunk(0, "p1"), score=1, score_kind="cosine")])
    assert d.action == Action.ANSWER and d.draft == "Here is the answer." and d.citations == ["c0"]


def test_vanilla_answers_even_with_no_retrieval():
    assert VanillaRAG(FakeLLM(["x"]), "sys").decide("how?", []).action == Action.ANSWER


def test_runner_turns_policy_exception_into_error_prediction(tmp_path):
    def boom(q):
        if q == "question 2":
            raise RuntimeError("api down")
        return Decision(action=Action.ANSWER)

    res = run_eval(Pol(boom), Retr(), GOLD, "dev", CFG, results_dir=tmp_path)
    assert res.n_errors == 1
    assert [p["pred"] for p in res.predictions].count("error") == 1 and len(res.predictions) == 4


def test_runner_refuses_test_without_dev_run_then_allows(tmp_path):
    items = GOLD + [gold(5, Action.ANSWER, split="test", page="p1")]
    with pytest.raises(ConfigNotFrozenError):
        run_eval(Pol(), Retr(), items, "test", CFG, results_dir=tmp_path)
    run_eval(Pol(), Retr(), items, "dev", CFG, results_dir=tmp_path)
    res = run_eval(Pol(), Retr(), items, "test", CFG, results_dir=tmp_path)
    assert res.split == "test" and len(res.predictions) == 1


def test_config_hash_changes_when_prompt_changes():
    assert config_hash(CFG) == config_hash(dict(CFG))
    assert config_hash(CFG) != config_hash({**CFG, "prompt": "v2"})


def test_runner_writes_files_and_metrics(tmp_path):
    res = run_eval(Pol(), Retr(), GOLD, "dev", CFG, results_dir=tmp_path)
    d = tmp_path / "runs" / res.run_id
    assert (d / "predictions.jsonl").exists() and (d / "config.json").exists()
    metrics = json.loads((d / "metrics.json").read_text())
    assert metrics["macro_f1"] == pytest.approx(res.metrics["macro_f1"])
    assert json.loads((d / "config.json").read_text())["config_hash"] == config_hash(CFG)
    assert res.metrics["unsafe_answer_rate"] == pytest.approx(1.0)
    assert res.cost_per_q == pytest.approx(0.01)


def test_retrieval_metrics_on_answer_items_only(tmp_path):
    res = run_eval(Pol(), Retr(("p1",)), GOLD, "dev", CFG, results_dir=tmp_path)
    assert res.metrics["recall@5"] == pytest.approx(0.5)
    assert res.metrics["n_retrieval_items"] == 2


def test_repeats_use_distinct_repeat_values(tmp_path):
    pol = Pol()
    res = run_eval(pol, Retr(), GOLD, "dev", CFG, repeats=3, results_dir=tmp_path)
    assert sorted(set(pol.repeats)) == [0, 1, 2]
    assert len(res.predictions) == 4 and "macro_f1_repeats" in res.metrics


def test_latency_excludes_cached_calls(tmp_path):
    def fn(q):
        cached = q.endswith("1")
        return Decision(action=Action.ANSWER, latency_s=100.0 if cached else 2.0, cached=cached)

    res = run_eval(Pol(fn), Retr(), GOLD, "dev", CFG, results_dir=tmp_path)
    assert res.latency_p50 == pytest.approx(2.0)


def test_empty_question_escalated_by_runner_never_answered(tmp_path):
    items = [gold(1, Action.ANSWER, page="p1", text="   ")]
    res = run_eval(Pol(), Retr(), items, "dev", CFG, results_dir=tmp_path)
    assert res.predictions[0]["pred"] == "escalate"


def test_ablation_row_appended_not_overwritten(tmp_path):
    res = run_eval(Pol(), Retr(), GOLD, "dev", CFG, results_dir=tmp_path)
    path = tmp_path / "ablation.md"
    append_ablation_row(res, "A vector baseline", "baseline", "first", path=path)
    append_ablation_row(res, "A again", "keep", "second", path=path)
    lines = path.read_text().splitlines()
    assert lines[0].startswith("| step") and len(lines) == 4
    assert "A vector baseline" in lines[2] and "A again" in lines[3] and res.run_id in lines[3]


def test_clean_query_flag_selects_cleaned_text_over_raw(tmp_path):
    seen = []

    class Spy(Retr):
        def retrieve(self, query, k):
            seen.append(query)
            return super().retrieve(query, k)

    g = GoldItem(qid="q1", text="clean text", text_raw="raw text > quoted", label=Action.ANSWER, evidence=[],
                 missing_detail=None, reference_answer=None, split="dev", source="agree")
    run_eval(Pol(), Spy(), [g], "dev", {**CFG, "clean_query": False}, results_dir=tmp_path)
    run_eval(Pol(), Spy(), [g], "dev", {**CFG, "clean_query": True}, results_dir=tmp_path)
    assert seen == ["raw text > quoted", "clean text"]


def test_prediction_rows_record_top_cosine(tmp_path):
    res = run_eval(Pol(), Retr(), GOLD, "dev", CFG, results_dir=tmp_path)
    assert all(r["top_cosine"] == 1.0 for r in res.predictions)
