"""Judge metrics against gold labels.

label  = Alice's answer is actually correct.
pred   = the judge said "correct".
Positive class is "Alice correct", so FPR = wrong answers the judge accepted (false accepts).
Bootstraps resample whole questions, because Alice's samples of one question are correlated.
"""

import math
import random
from collections import defaultdict
from collections.abc import Callable, Sequence


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def summarize(labels: Sequence[bool], preds: Sequence[bool], probs: Sequence[float] | None = None) -> dict:
    tp = sum(l and p for l, p in zip(labels, preds))
    tn = sum(not l and not p for l, p in zip(labels, preds))
    fp = sum(not l and p for l, p in zip(labels, preds))
    fn = sum(l and not p for l, p in zip(labels, preds))
    n = len(labels)
    tpr, tnr = _rate(tp, tp + fn), _rate(tn, tn + fp)
    out = {
        "n": n,
        "accuracy": _rate(tp + tn, n),
        "balanced_accuracy": (tpr + tnr) / 2 if tpr is not None and tnr is not None else None,
        "false_accept_rate": _rate(fp, fp + tn),
        "false_reject_rate": _rate(fn, fn + tp),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }
    if probs is not None:
        out["brier"] = _rate(sum((p - float(l)) ** 2 for l, p in zip(labels, probs)), n)
    return out


def accuracy(labels: Sequence[bool], preds: Sequence[bool]) -> float:
    return sum(l == p for l, p in zip(labels, preds)) / len(labels)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def flips(labels: Sequence[bool], preds_a: Sequence[bool], preds_b: Sequence[bool]) -> dict:
    """How judging changed per item going from condition a to condition b."""
    ra = [l == p for l, p in zip(labels, preds_a)]
    rb = [l == p for l, p in zip(labels, preds_b)]
    return {
        "both_right": sum(x and y for x, y in zip(ra, rb)),
        "both_wrong": sum(not x and not y for x, y in zip(ra, rb)),
        "wrong_to_right": sum(not x and y for x, y in zip(ra, rb)),
        "right_to_wrong": sum(x and not y for x, y in zip(ra, rb)),
    }


def cluster_bootstrap_ci(
    stat: Callable[[list[int]], float],
    clusters: Sequence[str],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI for stat(indices), resampling clusters with replacement."""
    by_cluster: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(clusters):
        by_cluster[c].append(i)
    groups = list(by_cluster.values())
    rng = random.Random(seed)
    values = []
    for _ in range(n_boot):
        idx = [i for g in rng.choices(groups, k=len(groups)) for i in g]
        values.append(stat(idx))
    values.sort()
    lo = values[int(alpha / 2 * n_boot)]
    hi = values[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return lo, hi


def paired_comparison(
    labels: Sequence[bool],
    preds_a: Sequence[bool],
    preds_b: Sequence[bool],
    clusters: Sequence[str],
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Accuracy change from condition a to b on the same items."""

    def delta(idx: list[int]) -> float:
        la = [labels[i] for i in idx]
        return accuracy(la, [preds_b[i] for i in idx]) - accuracy(la, [preds_a[i] for i in idx])

    f = flips(labels, preds_a, preds_b)
    lo, hi = cluster_bootstrap_ci(delta, clusters, n_boot, seed)
    return {
        "n": len(labels),
        "delta_accuracy": delta(list(range(len(labels)))),
        "ci95": [lo, hi],
        "mcnemar_p": mcnemar_exact(f["wrong_to_right"], f["right_to_wrong"]),
        "flips": f,
    }


def auroc(labels: Sequence[bool], probs: Sequence[float]) -> float | None:
    """P(a random correct answer gets a higher p_correct than a random wrong one); ties count half."""
    n_pos = sum(bool(l) for l in labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return None
    # Mann-Whitney U from midranks: O(n log n), so bootstrapping it stays fast.
    order = sorted(range(len(probs)), key=lambda i: probs[i])
    ranks = [0.0] * len(probs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and probs[order[j + 1]] == probs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    rank_sum = sum(r for r, l in zip(ranks, labels) if l)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def paired_auroc(
    labels: Sequence[bool],
    probs_a: Sequence[float],
    probs_b: Sequence[float],
    clusters: Sequence[str],
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """AUROC change from condition a to b on the same items, with a question-clustered bootstrap CI."""

    def delta(idx: list[int]) -> float:
        la = [labels[i] for i in idx]
        a, b = auroc(la, [probs_a[i] for i in idx]), auroc(la, [probs_b[i] for i in idx])
        return 0.0 if a is None or b is None else b - a

    lo, hi = cluster_bootstrap_ci(delta, clusters, n_boot, seed)
    return {"delta_auroc": delta(list(range(len(labels)))), "ci95": [lo, hi]}
