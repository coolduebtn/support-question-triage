import math
from collections.abc import Callable

import numpy as np
from scipy.stats import binomtest


def _strata(y_true: list[str], stratify: bool) -> list[np.ndarray]:
    arr = np.array(y_true)
    if not stratify:
        return [np.arange(len(arr))]
    return [np.where(arr == c)[0] for c in sorted(set(y_true))]


def _resample(strata: list[np.ndarray], rng) -> np.ndarray:
    return np.concatenate([rng.choice(idx, size=len(idx), replace=True) for idx in strata])


def bootstrap_ci(y_true, y_pred, metric: Callable, n: int = 2000, seed: int = 13, stratify: bool = True,
                 alpha: float = 0.05) -> tuple[float, float]:
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("need equal, non-empty inputs")
    rng = np.random.default_rng(seed)
    strata = _strata(y_true, stratify)
    t, p = np.array(y_true, dtype=object), np.array(y_pred, dtype=object)
    vals = []
    for _ in range(n):
        idx = _resample(strata, rng)
        v = metric(list(t[idx]), list(p[idx]))
        if v is not None:
            vals.append(v)
    if not vals:
        raise ValueError("metric undefined on every resample")
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_bootstrap_diff(y_true, pred_a, pred_b, metric: Callable, n: int = 2000, seed: int = 13,
                          stratify: bool = True, alpha: float = 0.05) -> dict:
    """delta = metric(pred_b) - metric(pred_a), resampling the same items for both systems."""
    if not y_true or not (len(y_true) == len(pred_a) == len(pred_b)):
        raise ValueError("need equal, non-empty inputs")
    rng = np.random.default_rng(seed)
    strata = _strata(y_true, stratify)
    t, a, b = (np.array(x, dtype=object) for x in (y_true, pred_a, pred_b))
    diffs = []
    for _ in range(n):
        idx = _resample(strata, rng)
        va, vb = metric(list(t[idx]), list(a[idx])), metric(list(t[idx]), list(b[idx]))
        if va is not None and vb is not None:
            diffs.append(vb - va)
    lo, hi = float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))
    delta = metric(list(y_true), list(pred_b)) - metric(list(y_true), list(pred_a))
    return {"delta": delta, "ci": (lo, hi), "improves": lo > 0}


def mcnemar_exact(correct_a: list[bool], correct_b: list[bool]) -> float:
    if not correct_a or len(correct_a) != len(correct_b):
        raise ValueError("need equal, non-empty inputs")
    b = sum(x and not y for x, y in zip(correct_a, correct_b))
    c = sum(y and not x for x, y in zip(correct_a, correct_b))
    if b + c == 0:
        return 1.0
    return float(binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue)


def min_detectable_effect(n: int, discordant_rate: float = 0.25) -> float:
    """Approximate smallest paired accuracy difference detectable with 80% power at alpha 0.05
    (z = 1.96 + 0.84), assuming a share `discordant_rate` of items where the two systems disagree."""
    if n <= 0:
        raise ValueError("n must be positive")
    return 2.8 * math.sqrt(discordant_rate / n)
