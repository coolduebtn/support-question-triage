import pytest

from triage.evals.analysis import compare_pair, error_table, risk_coverage, select_final
from triage.schemas import Action, Evidence, GoldItem, RunResult


def rows(pairs, retrieved=None, top=None):
    out = []
    for i, (g, p) in enumerate(pairs):
        out.append({"qid": f"q{i}", "gold": g, "pred": p, "reason": "", "retrieved_pages": (retrieved or {}).get(i, []),
                    "top_cosine": (top or {}).get(i, 0.5), "error": "boom" if p == "error" else None})
    return out


def result(pairs, cost=0.01, **kw):
    from triage.evals.metrics import macro_f1, over_ask_rate, over_escalation_rate
    yt, yp = [g for g, _ in pairs], [p for _, p in pairs]
    m = {"n_items": len(pairs), "macro_f1": macro_f1(yt, yp), "over_ask_rate": over_ask_rate(yt, yp),
         "over_escalation_rate": over_escalation_rate(yt, yp)}
    return RunResult(run_id="r", config_hash="h", split="dev", predictions=rows(pairs, **kw), metrics=m, n_errors=0,
                     cost_per_q=cost, latency_p50=1.0)


A, C, E = "answer", "clarify", "escalate"
GOOD = [(A, A)] * 5 + [(C, C)] * 3 + [(E, E)] * 3
OK = [(A, A)] * 5 + [(C, A)] * 3 + [(E, E)] * 3


def test_select_final_respects_limits():
    greedy = result([(A, C)] * 3 + [(A, A)] * 2 + [(C, C)] * 3 + [(E, E)] * 3)  # high over-ask
    limits = {"over_ask_max": 0.25, "over_escalation_max": 0.4, "cost_per_q_max": 0.02}
    assert select_final({"greedy": greedy, "ok": result(OK)}, limits, complexity={"greedy": 1, "ok": 2}) == "ok"


def test_select_final_raises_when_nothing_within_limits():
    with pytest.raises(ValueError):
        select_final({"x": result(OK, cost=1.0)}, {"cost_per_q_max": 0.02}, complexity={"x": 1})


def test_tie_within_mde_prefers_simpler():
    slightly_better = result(GOOD)
    simple = result(GOOD[:-1] + [(E, A)])  # one item worse, well inside the MDE
    assert select_final({"complex": slightly_better, "simple": simple}, {},
                        complexity={"complex": 3, "simple": 1}) == "simple"


def test_clear_winner_beats_simplicity():
    bad = result([(A, E)] * 5 + [(C, A)] * 3 + [(E, E)] * 3)
    assert select_final({"simple": bad, "complex": result(GOOD)}, {}, complexity={"simple": 1, "complex": 3}) == "complex"


def test_compare_pair_no_claim_inside_mde():
    a, b = result(GOOD), result(GOOD[:-1] + [(E, A)])
    out = compare_pair(a, b, seed=1)
    assert out["claim"] is False and out["mde"] > abs(out["delta"])


def test_compare_pair_claims_large_clear_gain():
    worse = result([(A, E)] * 15 + [(C, A)] * 10 + [(E, E)] * 10)
    better = result([(A, A)] * 15 + [(C, C)] * 10 + [(E, E)] * 10)
    out = compare_pair(worse, better, seed=1)
    assert out["delta"] > 0.5 and out["claim"] is True and out["mcnemar_p"] < 0.01


def test_error_table_lists_every_misclassified_item_including_errors():
    pairs = [(A, A), (A, E), (C, A), (E, A), (A, "error")]
    res = result(pairs, retrieved={1: ["p2"], 4: ["p1"]})
    gold = [GoldItem(qid=f"q{i}", text=f"text {i}", label=Action(g), evidence=[Evidence(page_id="p1", heading="h", quote="q" * 40)] if g == A else [],
                     missing_detail=None, reference_answer=None, split="dev", source="agree") for i, (g, _) in enumerate(pairs)]
    table = error_table(res, gold)
    assert [t["qid"] for t in table] == ["q1", "q2", "q3", "q4"]
    causes = {t["qid"]: t["cause_hint"] for t in table}
    assert causes["q1"] == "retrieval_miss" and causes["q2"] == "missed_ambiguity"
    assert causes["q3"] == "unsafe_answer" and causes["q4"] == "error"


def test_risk_coverage_is_monotone_in_coverage():
    pairs = [(A, A), (A, A), (E, A), (E, A), (C, A)]
    res = result(pairs, top={0: 0.9, 1: 0.8, 2: 0.4, 3: 0.3, 4: 0.2})
    curve = risk_coverage(res)
    covs = [c for c, _ in curve]
    assert covs == sorted(covs) and covs[-1] == 1.0
    assert curve[0][1] == 0.0 and curve[-1][1] == pytest.approx(3 / 5)


def test_select_final_respects_unsafe_answer_limit():
    always_answers = result([(A, A)] * 5 + [(C, A)] * 3 + [(E, A)] * 3)
    safer = result(OK)
    safer.metrics["unsafe_answer_rate"] = 0.5
    always_answers.metrics["unsafe_answer_rate"] = 1.0
    assert select_final({"A": always_answers, "B": safer}, {"unsafe_answer_max": 0.7},
                        complexity={"A": 1, "B": 2}) == "B"
