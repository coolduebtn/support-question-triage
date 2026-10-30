import pytest

from triage.evals.runner import run_eval
from triage.labelling.probes import ProbeOut, is_deletion_only, make_probe, probes_from_gold
from triage.llm import FakeLLM
from triage.schemas import Action, Decision, Evidence, GoldItem

ORIG = "Export fails with error NullPointerException in InvoiceExportService on 18.12.09 when I print the report."


def gold(i, label=Action.ANSWER, text=ORIG, split="dev", synthetic=False):
    ev = [Evidence(page_id="p1", heading="H", quote="q" * 40)] if label == Action.ANSWER else []
    return GoldItem(qid=f"q{i}", text=text, text_raw=text, label=label, evidence=ev, missing_detail=None,
                    reference_answer="ref", split=split, source="agree", synthetic=synthetic)


def test_deletion_only_accepts_pure_deletions_and_rejects_added_words():
    assert is_deletion_only(ORIG, "Export fails when I print the report.")
    assert is_deletion_only(ORIG, "Export fails with error when I print.")
    assert not is_deletion_only(ORIG, "Export fails with a weird crash when I print the report.")
    assert not is_deletion_only(ORIG, "")


def test_make_probe_requires_deletion_only_and_a_named_removed_detail():
    good = ProbeOut(degraded_text="Export fails when I print the report.", removed_detail="the exact error and version")
    assert make_probe(gold(1), FakeLLM([good])).text == "Export fails when I print the report."
    added = ProbeOut(degraded_text="Export fails with a weird crash.", removed_detail="error")
    assert make_probe(gold(1), FakeLLM([added])) is None
    unchanged = ProbeOut(degraded_text=ORIG, removed_detail="nothing")
    assert make_probe(gold(1), FakeLLM([unchanged])) is None
    no_detail = ProbeOut(degraded_text="Export fails.", removed_detail="  ")
    assert make_probe(gold(1), FakeLLM([no_detail])) is None


def test_probe_item_is_clarify_synthetic_and_traceable():
    out = ProbeOut(degraded_text="Export fails when I print the report.", removed_detail="error text and version")
    p = make_probe(gold(7, split="test"), FakeLLM([out]))
    assert p.qid == "q7-probe" and p.label == Action.CLARIFY and p.synthetic and p.derived_from == "q7"
    assert p.missing_detail == "error text and version" and p.split == "test" and p.source == "synthetic_probe"
    assert p.evidence and p.reference_answer is None


def test_probes_from_gold_uses_only_real_answer_items_and_keeps_split():
    items = [gold(1), gold(2, Action.ESCALATE), gold(3, Action.CLARIFY), gold(4, synthetic=True), gold(5, split="test")]
    out = ProbeOut(degraded_text="Export fails when I print the report.", removed_detail="detail")
    probes = probes_from_gold(items, FakeLLM([out, out]), n=5, seed=1)
    assert sorted(p.derived_from for p in probes) == ["q1", "q5"]
    assert {p.derived_from: p.split for p in probes} == {"q1": "dev", "q5": "test"}


class Retr:
    def retrieve(self, q, k):
        return []


class Pol:
    name = "p"

    def decide(self, q, r, repeat=0):
        return Decision(action=Action.CLARIFY, missing_detail="x", draft="Which version?")


def test_runner_excludes_synthetic_by_default_and_can_select_only_or_both(tmp_path):
    items = [gold(1), gold(2, Action.ESCALATE), gold(3, Action.CLARIFY, text="vague", synthetic=True)]
    base = {"k": 5}
    assert run_eval(Pol(), Retr(), items, "dev", base, results_dir=tmp_path).metrics["n_items"] == 2
    only = run_eval(Pol(), Retr(), items, "dev", {**base, "synthetic": "only"}, results_dir=tmp_path)
    both = run_eval(Pol(), Retr(), items, "dev", {**base, "synthetic": "both"}, results_dir=tmp_path)
    assert only.metrics["n_items"] == 1 and both.metrics["n_items"] == 3
