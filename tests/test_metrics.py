import pytest
from sklearn.metrics import f1_score

from triage.evals.metrics import (macro_f1, over_ask_rate, over_escalation_rate, per_class_prf,
                                  retrieval_metrics, unsafe_answer_rate)

LABELS = ["answer", "clarify", "escalate"]


def test_macro_f1_perfect():
    y = ["answer", "clarify", "escalate", "answer"]
    assert macro_f1(y, y) == 1.0


def test_macro_f1_missing_pred_class_zero_not_error():
    y_true = ["answer", "clarify", "escalate", "clarify"]
    y_pred = ["answer", "answer", "escalate", "answer"]
    got = macro_f1(y_true, y_pred)
    assert got == pytest.approx(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))
    assert per_class_prf(y_true, y_pred)["clarify"]["f1"] == 0.0


def test_error_prediction_counts_wrong():
    y_true = ["answer", "escalate"]
    assert macro_f1(y_true, ["answer", "error"]) < macro_f1(y_true, ["answer", "escalate"])
    assert per_class_prf(y_true, ["answer", "error"])["escalate"]["recall"] == 0.0


def test_matches_sklearn_fixed_labels():
    y_true = ["answer", "answer", "clarify", "escalate", "escalate", "clarify"]
    y_pred = ["answer", "clarify", "clarify", "escalate", "answer", "clarify"]
    assert macro_f1(y_true, y_pred) == pytest.approx(
        f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))


def test_per_class_has_support_counts():
    prf = per_class_prf(["answer", "answer", "clarify"], ["answer", "clarify", "clarify"])
    assert prf["answer"]["support"] == 2 and prf["clarify"]["support"] == 1 and prf["escalate"]["support"] == 0


def test_unsafe_rate_example():
    y_true = ["clarify", "clarify", "escalate", "escalate", "escalate", "answer"]
    y_pred = ["answer", "clarify", "answer", "escalate", "escalate", "answer"]
    assert unsafe_answer_rate(y_true, y_pred) == pytest.approx(2 / 5)


def test_over_ask_and_over_escalation_examples():
    y_true = ["answer", "answer", "answer", "answer", "clarify"]
    y_pred = ["clarify", "answer", "escalate", "escalate", "clarify"]
    assert over_ask_rate(y_true, y_pred) == pytest.approx(1 / 4)
    assert over_escalation_rate(y_true, y_pred) == pytest.approx(2 / 4)


def test_rate_returns_none_on_empty_denominator():
    assert unsafe_answer_rate(["answer"], ["answer"]) is None
    assert over_ask_rate(["clarify"], ["clarify"]) is None


def test_recall_mrr_example():
    gold = [{"p1"}, {"p2", "p3"}, {"p9"}]
    ranked = [["p1", "p5"], ["p7", "p3", "p2"], ["p4", "p5"]]
    m = retrieval_metrics(gold, ranked, k=2)
    assert m["hit"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert m["mrr"] == pytest.approx((1 + 0.5 + 0) / 3)


def test_empty_input_raises():
    with pytest.raises(ValueError):
        macro_f1([], [])
    with pytest.raises(ValueError):
        retrieval_metrics([], [], k=5)
    with pytest.raises(ValueError):
        macro_f1(["answer"], ["answer", "clarify"])
