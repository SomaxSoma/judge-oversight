import pytest

from oversight import metrics


def test_summarize_hand_computed():
    # labels: Alice correct?  preds: judge said "correct"?
    labels = [True, True, True, False, False, False, False, True]
    preds = [True, True, False, True, False, False, True, True]
    probs = [0.9, 0.8, 0.4, 0.6, 0.1, 0.2, 0.7, 1.0]
    s = metrics.summarize(labels, preds, probs)
    assert (s["tp"], s["fn"], s["fp"], s["tn"]) == (3, 1, 2, 2)
    assert s["accuracy"] == pytest.approx(5 / 8)
    assert s["false_accept_rate"] == pytest.approx(2 / 4)
    assert s["false_reject_rate"] == pytest.approx(1 / 4)
    assert s["balanced_accuracy"] == pytest.approx((3 / 4 + 2 / 4) / 2)
    expected_brier = sum((p - l) ** 2 for l, p in zip(labels, probs)) / 8
    assert s["brier"] == pytest.approx(expected_brier)


def test_summarize_single_class_has_no_balanced_accuracy():
    s = metrics.summarize([True, True], [True, False])
    assert s["balanced_accuracy"] is None
    assert s["false_accept_rate"] is None


def test_mcnemar_exact():
    # b=1, c=6: p = 2 * (C(7,0) + C(7,1)) / 2^7 = 0.125
    assert metrics.mcnemar_exact(1, 6) == pytest.approx(0.125)
    assert metrics.mcnemar_exact(6, 1) == pytest.approx(0.125)
    assert metrics.mcnemar_exact(0, 0) == 1.0
    assert metrics.mcnemar_exact(5, 5) == 1.0


def test_flips_and_paired_comparison():
    labels = [True, False, True, False, True, False]
    a = [False, True, True, False, False, True]  # right on items 2, 3
    b = [True, False, True, True, True, True]  # right on items 0, 1, 2, 4
    f = metrics.flips(labels, a, b)
    assert f == {"both_right": 1, "both_wrong": 1, "wrong_to_right": 3, "right_to_wrong": 1}
    c = metrics.paired_comparison(labels, a, b, clusters=["q1", "q1", "q2", "q2", "q3", "q3"], n_boot=500)
    assert c["delta_accuracy"] == pytest.approx(4 / 6 - 2 / 6)
    lo, hi = c["ci95"]
    assert lo <= c["delta_accuracy"] <= hi


def test_cluster_bootstrap_is_deterministic():
    clusters = ["a", "a", "b", "c", "c", "d"]
    stat = lambda idx: len(idx)  # noqa: E731
    assert metrics.cluster_bootstrap_ci(stat, clusters, 200, seed=1) == metrics.cluster_bootstrap_ci(
        stat, clusters, 200, seed=1
    )
