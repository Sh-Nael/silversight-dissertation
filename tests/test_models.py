from __future__ import annotations

import numpy as np

from xaglab.models.base import Calibrator
from xaglab.models.tuning import VolatilityTask


def test_qlike_scale_is_the_closed_form_minimum():
    rng = np.random.default_rng(0)
    sigma2 = np.exp(rng.normal(-8, 0.5, 3000))
    rv = sigma2 * rng.chisquare(1, 3000)          # one-day realised variance
    pred = np.full(3000, -1.27)                   # E[log chi2_1] = -1.27: the log bias
    task = VolatilityTask(rv, base=sigma2)        # relative target: log(rv / sigma2)
    rows = np.arange(3000)
    c = task.scale(pred, rows)
    best = task.score(pred, rows)
    for k in (0.8, 0.9, 1.1, 1.25):
        f = k * task.variance(pred, rows, c)
        assert np.mean(np.log(f) + rv / f) > best
    assert 2.5 < c < 4.5                          # recovers ~exp(1.27) = 3.56


def test_platt_calibration_fixes_overconfidence():
    rng = np.random.default_rng(1)
    p_true = rng.uniform(0.4, 0.6, 5000)
    y = (rng.random(5000) < p_true).astype(float)
    overconfident = np.clip(0.5 + 3 * (p_true - 0.5), 0.01, 0.99)
    cal = Calibrator.fit(overconfident, y)
    assert cal.fitted and 0.2 < cal.a < 0.5       # shrinks the logit roughly 3x
