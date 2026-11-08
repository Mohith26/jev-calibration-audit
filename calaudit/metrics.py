"""Calibration and selective-prediction metrics. Pure numpy/scipy, no I/O."""

from __future__ import annotations

import numpy as np
from scipy.stats import binom


def ece(confidence, correct, bins: int = 10) -> float:
    """Expected calibration error with equal-width bins on [0, 1] (first bin closed on the left)."""
    c = np.asarray(confidence, float)
    y = np.asarray(correct, float)
    if len(c) == 0:
        return float("nan")
    idx = np.clip(np.ceil(c * bins).astype(int) - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(c[m].mean() - y[m].mean())
    return float(total)


def reliability(confidence, correct, edges=(0.0, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)) -> list[dict]:
    """Rows of (bin, n, mean confidence, observed agreement). The last bin includes 1.0."""
    c = np.asarray(confidence, float)
    y = np.asarray(correct, float)
    rows = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        last = i == len(edges) - 2
        m = (c >= lo) & ((c <= hi) if last else (c < hi))
        if m.any():
            rows.append({"lo": lo, "hi": hi, "n": int(m.sum()), "confidence": float(c[m].mean()),
                         "agreement": float(y[m].mean())})
    return rows


def coverage_at(confidence, correct, target: float) -> float:
    """Largest fraction of items, taken in order of decreasing confidence, whose agreement is >= target.

    This is the in-sample (oracle) risk-coverage point. Ties are kept together: a threshold accepts every
    item at or above it, so we only evaluate cut points between distinct confidence values.
    """
    c = np.asarray(confidence, float)
    y = np.asarray(correct, float)
    if len(c) == 0:
        return float("nan")
    order = np.argsort(-c, kind="stable")
    c, y = c[order], y[order]
    running = np.cumsum(y) / np.arange(1, len(y) + 1)
    boundary = np.r_[c[1:] != c[:-1], True]  # last index of each tie group
    ok = np.where(boundary & (running >= target))[0]
    return float((ok.max() + 1) / len(c)) if len(ok) else 0.0


GRID = np.round(np.linspace(0.5, 1.0, 101), 3)


def ltt_threshold(confidence, correct, alpha: float, delta: float, grid=GRID) -> float:
    """Learn-then-Test threshold (Angelopoulos et al., 2021) with a Bonferroni correction over a fixed grid.

    For each candidate threshold t in the grid, test H0: "error rate among items with confidence >= t is
    above alpha" with the exact binomial p-value P(Binom(n, alpha) <= errors). Every t whose p-value is
    <= delta / len(grid) is certified; return the lowest certified t (most coverage). With exchangeable
    calibration and test items, P(true error rate among accepted items > alpha) <= delta.
    Returns +inf (accept nothing) if no threshold is certified.
    """
    c = np.asarray(confidence, float)
    y = np.asarray(correct, bool)
    grid = np.asarray(grid, float)
    accepted = c[None, :] >= grid[:, None]
    n = accepted.sum(axis=1)
    errors = (accepted & ~y[None, :]).sum(axis=1)
    with np.errstate(invalid="ignore"):
        p = np.where(n > 0, binom.cdf(errors, n, alpha), 1.0)
    certified = grid[p <= delta / len(grid)]
    return float(certified.min()) if len(certified) else float("inf")


def split_conformal_selective(groups, confidence, correct, alpha: float, delta: float,
                              splits: int = 200, seed: int = 0) -> dict:
    """Repeated 50/50 splits by group (case): LTT threshold on one half, evaluated on the other.

    violation_rate counts splits whose held-out agreement on accepted items fell below 1 - alpha. It includes
    finite-sample noise of the held-out half, so it can exceed delta slightly even when the guarantee holds.

    Splitting by case keeps every question of a case on the same side, so the guarantee is not inflated
    by near-duplicate questions within a case.
    """
    g = np.asarray(groups)
    c = np.asarray(confidence, float)
    y = np.asarray(correct, bool)
    unique = np.unique(g)
    rng = np.random.default_rng(seed)
    coverage, agreement, violations = [], [], 0
    for _ in range(splits):
        cal_groups = rng.choice(unique, size=len(unique) // 2, replace=False)
        cal = np.isin(g, cal_groups)
        t = ltt_threshold(c[cal], y[cal], alpha, delta)
        test = ~cal
        accepted = test & (c >= t)
        cov = accepted.sum() / test.sum()
        coverage.append(cov)
        if accepted.any():
            agr = y[accepted].mean()
            agreement.append(agr)
            violations += int(agr < 1 - alpha)
    return {
        "coverage_mean": float(np.mean(coverage)),
        "coverage_p10": float(np.percentile(coverage, 10)),
        "coverage_p90": float(np.percentile(coverage, 90)),
        "agreement_mean": float(np.mean(agreement)) if agreement else float("nan"),
        "violation_rate": violations / splits,
        "splits": splits,
    }


def brier_binary(confidence, correct) -> float:
    """Brier score of the confidence as a forecast of agreement."""
    c = np.asarray(confidence, float)
    y = np.asarray(correct, float)
    return float(np.mean((c - y) ** 2))


def top_tie_share(confidence) -> float:
    """Share of answers tied at the model's single most common confidence value at or above 0.99."""
    c = np.round(np.asarray(confidence, float), 6)
    high = c[c >= 0.99]
    if len(high) == 0:
        return 0.0
    values, counts = np.unique(high, return_counts=True)
    return float(counts.max() / len(c))
