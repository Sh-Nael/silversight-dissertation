"""Significance tests named before any result existed (Chapter 3, Section 3.6).

* Diebold-Mariano on paired per-origin losses, HAC (Newey-West) variance with
  h-1 lags for overlapping h-step targets, and the Harvey-Leybourne-Newbold (1997)
  small-sample correction with Student-t reference. Primary inference.
* Wilcoxon signed-rank on per-fold mean losses: consistency across the 8 folds.
  Robustness only, since n = 8 is low-powered (stated openly, D-03).
* Paired sign-flip permutation test on loss differences within blocks, as the
  distribution-free backstop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class TestResult:
    stat: float
    p_value: float
    mean_diff: float
    n: int


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1) -> TestResult:
    """H0: equal expected loss. mean_diff < 0 means model A is better. Two-sided p."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[~np.isnan(d)]
    n = len(d)
    dbar = d.mean()
    u = d - dbar
    lags = max(h - 1, 0)
    gamma = [np.dot(u[k:], u[: n - k]) / n for k in range(lags + 1)]
    var = gamma[0] + 2 * sum((1 - k / (lags + 1)) * gamma[k] for k in range(1, lags + 1))
    if var <= 0:
        return TestResult(float("nan"), float("nan"), float(dbar), n)
    dm = dbar / np.sqrt(var / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return TestResult(float(stat), float(p), float(dbar), n)


def wilcoxon_folds(fold_loss_a: np.ndarray, fold_loss_b: np.ndarray) -> TestResult:
    a, b = np.asarray(fold_loss_a, float), np.asarray(fold_loss_b, float)
    if np.allclose(a, b, rtol=0, atol=1e-15):  # identical forecasts: nothing to test
        return TestResult(0.0, 1.0, 0.0, len(a))
    res = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    return TestResult(float(res.statistic), float(res.pvalue), float((a - b).mean()), len(a))


def permutation_test(loss_a: np.ndarray, loss_b: np.ndarray, block: int = 21,
                     n_perm: int = 10_000, seed: int = 0) -> TestResult:
    """Sign-flip test on block means of the loss difference (blocks keep serial dependence)."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[~np.isnan(d)]
    nb = len(d) // block
    means = d[: nb * block].reshape(nb, block).mean(axis=1)
    obs = means.mean()
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(n_perm, nb))
    null = (flips * means).mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return TestResult(float(obs), float(p), float(d.mean()), len(d))
