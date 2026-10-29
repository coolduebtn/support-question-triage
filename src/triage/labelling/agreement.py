from triage.evals.metrics import LABELS


def agreement(a: list[str], b: list[str]) -> dict:
    if not a or len(a) != len(b):
        raise ValueError("need equal, non-empty label lists")
    n = len(a)
    raw = sum(x == y for x, y in zip(a, b)) / n
    pa = {c: sum(x == c for x in a) / n for c in LABELS}
    pb = {c: sum(y == c for y in b) / n for c in LABELS}
    pe = sum(pa[c] * pb[c] for c in LABELS)
    kappa = None if pe >= 1 else (raw - pe) / (1 - pe)
    pi = {c: (pa[c] + pb[c]) / 2 for c in LABELS}
    pe_g = sum(p * (1 - p) for p in pi.values()) / (len(LABELS) - 1)
    ac1 = None if pe_g >= 1 else (raw - pe_g) / (1 - pe_g)
    confusion = {x: {y: 0 for y in LABELS} for x in LABELS}
    for x, y in zip(a, b):
        confusion[x][y] += 1
    return {"raw": raw, "kappa": kappa, "ac1": ac1, "confusion": confusion, "n": n}
