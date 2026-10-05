"""Tests for the selective-prediction evaluation (accuracy against coverage, D-31)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from xaglab.eval.report import causal_selection, ranked_selection, selective


def _run(p: np.ndarray, y: np.ndarray, h: int = 1):
    idx = pd.bdate_range("2010-01-04", periods=len(p))
    return SimpleNamespace(predictions=pd.DataFrame({f"p_up{h}": p, f"up{h}": y}, index=idx))


def test_causal_selection_never_looks_ahead():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0, 0.1, 1000)
    a = causal_selection(conf, 0.2)
    changed = conf.copy()
    changed[600:] = rng.uniform(0, 0.1, 400) * 10  # a different future
    b = causal_selection(changed, 0.2)
    assert np.array_equal(a[:600], b[:600])
    assert not a[:63].any()                             # no reference yet -> nothing selected
    assert abs(a[63:].mean() - 0.2) < 0.03               # about the target share on i.i.d. input


def test_a_constant_forecast_has_no_confident_days():
    conf = np.r_[np.full(300, 0.02), np.full(300, 0.03)]  # changes once, like climatology at a refit
    sel = causal_selection(conf, 0.1)
    assert sel.sum() <= 252                                # only the days after the step, until they fill the window
    assert not causal_selection(np.full(500, 0.02), 0.1).any()


def test_ranked_selection_takes_the_top_share():
    conf = np.array([0.01, 0.05, 0.02, 0.04, 0.03, 0.00, 0.06, 0.07, 0.08, 0.09])
    sel = ranked_selection(conf, 0.3)
    assert sel.sum() == 3 and set(np.flatnonzero(sel)) == {7, 8, 9}
    assert ranked_selection(conf, 1.0).all()
    assert not ranked_selection(np.full(10, 0.02), 0.3).any()   # ties: nothing is more confident


def test_selective_by_hand():
    # 10 days, 4 confident and right (2 up calls, 2 down calls); 6 unsure, 3 right.
    p = np.array([0.9, 0.1, 0.8, 0.2, 0.51, 0.51, 0.51, 0.49, 0.49, 0.49])
    y = np.array([1, 0, 1, 0, 1, 1, 0, 1, 1, 0], float)
    t = selective(_run(p, y), 1, coverages=(0.4, 1.0), mode="ranked").set_index("coverage_target")
    assert t.loc[0.4, "n"] == 4 and t.loc[0.4, "accuracy"] == 1.0
    assert t.loc[0.4, "up_calls"] == 0.5 and t.loc[0.4, "always_up"] == 0.5 and t.loc[0.4, "edge"] == 0.5
    assert t.loc[1.0, "accuracy"] == 0.7                 # 4 + 3 right out of 10
    assert t.loc[1.0, "always_up"] == 0.6
    # two-sided scorecard (D-32): up calls are days 0,2,4,5,6 (4 of 5 went up);
    # down calls are days 1,3,7,8,9 (3 of 5 went down)
    assert t.loc[1.0, "up_n"] == 5 and t.loc[1.0, "up_right"] == 0.8 and t.loc[1.0, "up_base"] == 0.6
    assert t.loc[1.0, "down_n"] == 5 and t.loc[1.0, "down_right"] == 0.6 and t.loc[1.0, "down_base"] == 0.4
    assert t.loc[0.4, "up_right"] == 1.0 and t.loc[0.4, "down_right"] == 1.0


def test_scoreboard_splits_up_and_down_calls():
    from xaglab.eval.metrics import direction_summary

    d = direction_summary(np.array([0.6, 0.7, 0.4, 0.3]), np.array([1.0, 0, 0, 0]))
    assert d["up_calls"] == 0.5 and d["up_precision"] == 0.5 and d["down_precision"] == 1.0


def test_riding_the_drift_is_not_an_up_side_edge():
    rng = np.random.default_rng(5)
    n = 3000
    y = (rng.uniform(size=n) < 0.56).astype(float)        # a rising market
    p = 0.5 + rng.uniform(0.0, 0.1, n)                     # always calls up, no information
    row = selective(_run(p, y), 1, coverages=(0.3,)).iloc[0]
    assert row["up_right"] > 0.53 and row["p_up"] > 0.05  # right often, but only the base rate
    assert row["down_n"] == 0


def test_edge_test_has_power_and_size():
    rng = np.random.default_rng(1)
    n = 3000
    y = (rng.uniform(size=n) < 0.52).astype(float)       # silver-like base rate
    skill = np.where(y == 1, 0.5 + rng.uniform(0, 0.1, n), 0.5 - rng.uniform(0, 0.1, n))
    noisy = np.where(rng.uniform(size=n) < 0.6, skill, 1 - skill)  # right 60% of the time
    good = selective(_run(noisy, y), 1, coverages=(0.3,)).iloc[0]
    assert good["accuracy"] > 0.55 and good["p_edge"] < 0.05

    # no information: calls "down" half the time on a rising market, so it loses to always-up;
    # the one-sided test must not report an edge
    random = selective(_run(rng.uniform(0.45, 0.55, n), y), 1, coverages=(0.3,)).iloc[0]
    assert random["edge"] < 0 and random["p_edge"] > 0.5 and random["p_coin"] > 0.05

    # size: 40 independent no-skill models (random probabilities) on a fair-coin market,
    # where always-up is no better than chance; the edge test must reject at 5% rarely
    rejections = 0
    for k in range(40):
        r = np.random.default_rng(100 + k)
        yy = (r.uniform(size=1500) < 0.5).astype(float)
        rejections += selective(_run(r.uniform(0.4, 0.6, 1500), yy), 1, coverages=(0.3,)).iloc[0]["p_edge"] < 0.05
    assert rejections <= 5
