"""Statistical-level tests (Table 3.1): the significance tests behave correctly.

On simulated forecasts of equal quality the tests must reject about 5% of the time
(correct size); on a planted real difference they must reject almost always (power).
"""

from __future__ import annotations

import numpy as np

from xaglab.eval.metrics import brier
from xaglab.eval.stats import diebold_mariano, permutation_test, wilcoxon_folds


def _forecasts(rng, n, edge_b=0.0):
    """Outcomes with true P(up) varying over time; A and B are noisy copies of it."""
    p_true = np.clip(0.52 + 0.08 * np.sin(np.arange(n) / 40) + rng.normal(0, 0.03, n), 0.05, 0.95)
    y = (rng.random(n) < p_true).astype(float)
    pa = np.clip(p_true + rng.normal(0, 0.10, n), 0.01, 0.99)
    pb = np.clip(p_true + rng.normal(0, 0.10 * (1 - edge_b), n), 0.01, 0.99)
    return brier(pa, y), brier(pb, y)


def test_dm_has_correct_size_under_equal_quality():
    rng = np.random.default_rng(42)
    rejections = sum(diebold_mariano(*_forecasts(rng, 4000)).p_value < 0.05 for _ in range(400))
    assert 0.02 <= rejections / 400 <= 0.09


def test_dm_detects_a_planted_improvement():
    rng = np.random.default_rng(7)
    powers = [diebold_mariano(*_forecasts(rng, 4000, edge_b=0.6)) for _ in range(50)]
    assert np.mean([r.p_value < 0.05 for r in powers]) > 0.9
    assert all(r.mean_diff > 0 for r in powers)  # A minus B > 0: B is better


def test_dm_hac_widens_with_overlapping_losses():
    rng = np.random.default_rng(3)
    d = rng.normal(0, 1, 5000)
    overlapping = np.convolve(d, np.ones(5) / 5, mode="same")  # MA(4): like 5-day targets
    naive = diebold_mariano(overlapping + 0.02, np.zeros(5000), h=1)
    hac = diebold_mariano(overlapping + 0.02, np.zeros(5000), h=5)
    assert abs(hac.stat) < abs(naive.stat)


def test_wilcoxon_and_permutation_directions():
    a = np.array([0.250, 0.249, 0.251, 0.248, 0.250, 0.247, 0.249, 0.250])
    b = a - 0.002
    assert wilcoxon_folds(b, a).p_value < 0.05
    rng = np.random.default_rng(0)
    la, lb = _forecasts(rng, 4000, edge_b=0.6)
    assert permutation_test(lb, la, n_perm=2000).p_value < 0.05
    la, lb = _forecasts(rng, 4000)
    assert permutation_test(lb, la, n_perm=2000).p_value > 0.01
