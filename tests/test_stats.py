import pytest

from triage.evals.metrics import macro_f1
from triage.evals.stats import (bootstrap_ci, mcnemar_exact, min_detectable_effect,
                                paired_bootstrap_diff)

Y_TRUE = ["answer"] * 8 + ["clarify"] * 6 + ["escalate"] * 6
Y_PRED = (["answer"] * 6 + ["clarify"] * 2 + ["clarify"] * 4 + ["escalate"] * 2 + ["escalate"] * 5 + ["answer"])


def test_bootstrap_seeded_and_contains_estimate():
    lo, hi = bootstrap_ci(Y_TRUE, Y_PRED, macro_f1, n=500, seed=1)
    assert (lo, hi) == bootstrap_ci(Y_TRUE, Y_PRED, macro_f1, n=500, seed=1)
    assert lo <= macro_f1(Y_TRUE, Y_PRED) <= hi


def test_stratified_bootstrap_keeps_all_classes():
    seen = []

    def metric(t, p):
        seen.append(set(t))
        return 0.0

    bootstrap_ci(["answer"] * 10 + ["clarify"] * 2 + ["escalate"] * 2,
                 ["answer"] * 14, metric, n=50, seed=3, stratify=True)
    assert all(s == {"answer", "clarify", "escalate"} for s in seen)


def test_mcnemar_known_table():
    a = [True] * 10 + [False] * 2 + [True] * 20
    b = [False] * 10 + [True] * 2 + [True] * 20
    assert mcnemar_exact(a, b) == pytest.approx(0.03857, abs=1e-4)


def test_mcnemar_no_discordant_pairs_is_one():
    assert mcnemar_exact([True, False], [True, False]) == 1.0


def test_paired_bootstrap_diff_detects_clear_gain():
    better = list(Y_TRUE)
    out = paired_bootstrap_diff(Y_TRUE, Y_PRED, better, macro_f1, n=500, seed=2)
    assert out["delta"] > 0 and out["ci"][0] > 0 and out["improves"] is True


def test_paired_bootstrap_diff_same_predictions_not_improvement():
    out = paired_bootstrap_diff(Y_TRUE, Y_PRED, list(Y_PRED), macro_f1, n=200, seed=2)
    assert out["delta"] == 0 and out["improves"] is False


def test_min_detectable_effect_shrinks_with_n():
    assert min_detectable_effect(10) > min_detectable_effect(40) > 0


def test_empty_inputs_raise():
    with pytest.raises(ValueError):
        bootstrap_ci([], [], macro_f1)
    with pytest.raises(ValueError):
        mcnemar_exact([], [])
