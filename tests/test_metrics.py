import numpy as np
import pytest

from calaudit.metrics import (
    brier_binary,
    coverage_at,
    ece,
    ltt_threshold,
    reliability,
    split_conformal_selective,
    top_tie_share,
)


def test_ece_zero_when_confidence_matches_frequency():
    # 10 items at 0.8 with 8 right, 10 items at 0.4 with 4 right: perfectly calibrated per bin.
    c = [0.8] * 10 + [0.4] * 10
    y = [1] * 8 + [0] * 2 + [1] * 4 + [0] * 6
    assert ece(c, y) == pytest.approx(0.0)


def test_ece_known_value():
    # One bin, confidence 0.9, accuracy 0.5 -> ECE 0.4.
    assert ece([0.9] * 4, [1, 0, 1, 0]) == pytest.approx(0.4)


def test_ece_puts_one_and_boundaries_in_expected_bins():
    # 1.0 lands in the last bin, 0.5 in the (0.4, 0.5] bin: two separate bins, each perfectly calibrated.
    assert ece([1.0, 1.0, 0.5, 0.5], [1, 1, 1, 0]) == pytest.approx(0.0)


def test_brier():
    assert brier_binary([1.0, 0.0, 0.5], [1, 1, 0]) == pytest.approx((0 + 1 + 0.25) / 3)


def test_reliability_last_bin_includes_one():
    rows = reliability([0.995, 1.0, 0.65], [1, 0, 1])
    assert [r["n"] for r in rows] == [1, 2]
    assert rows[-1]["agreement"] == pytest.approx(0.5)


def test_coverage_at_respects_ties():
    # Two items tied at 1.0 with one wrong: no threshold reaches 100%, even though the first item is right.
    assert coverage_at([1.0, 1.0, 0.9], [1, 0, 1], 1.0) == 0.0
    # Without the tie the top item alone qualifies.
    assert coverage_at([1.0, 0.99, 0.9], [1, 0, 1], 1.0) == pytest.approx(1 / 3)


def test_coverage_at_full_when_everything_right():
    assert coverage_at([0.6, 0.7, 0.8], [1, 1, 1], 0.99) == 1.0


def test_ltt_rejects_when_no_threshold_is_safe():
    c = np.linspace(0.5, 1.0, 200)
    y = np.zeros(200, bool)
    assert ltt_threshold(c, y, alpha=0.05, delta=0.1) == float("inf")


def test_ltt_accepts_everything_when_all_right_and_n_is_large():
    c = np.full(2000, 0.97)
    y = np.ones(2000, bool)
    t = ltt_threshold(c, y, alpha=0.05, delta=0.1)
    assert t <= 0.97  # every grid point at or below 0.97 accepts all 2000 items with zero errors
    assert (c >= t).all()


def test_ltt_controls_risk_on_synthetic_calibrated_data():
    # Calibrated model: P(right | conf) = conf. Target 95% agreement among accepted items.
    rng = np.random.default_rng(1)
    alpha, delta, bad, trials = 0.05, 0.1, 0, 200
    for _ in range(trials):
        c = rng.uniform(0.5, 1.0, 4000)
        y = rng.uniform(size=4000) < c
        t = ltt_threshold(c[:2000], y[:2000], alpha, delta)
        if np.isfinite(t):
            # True risk among accepted, computed analytically from the calibrated model.
            true_risk = 1 - (t + 1.0) / 2
            bad += int(true_risk > alpha)
    assert bad / trials <= delta


def test_split_conformal_keeps_groups_together_and_reports_coverage():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(2000), 5)
    c = rng.uniform(0.5, 1.0, len(groups))
    y = rng.uniform(size=len(groups)) < c
    out = split_conformal_selective(groups, c, y, alpha=0.1, delta=0.1, splits=30)
    # A calibrated uniform(0.5, 1) model reaches 90% agreement above t = 0.8, i.e. 40% coverage at best.
    assert 0.2 < out["coverage_mean"] <= 0.42
    assert out["agreement_mean"] >= 0.89
    assert out["violation_rate"] <= 0.3


def test_top_tie_share():
    assert top_tie_share([1.0, 1.0, 1.0, 0.99, 0.5]) == pytest.approx(0.6)
    assert top_tie_share([0.9, 0.8]) == 0.0
