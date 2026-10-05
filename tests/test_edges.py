"""Tests for the edge-dynamics analysis (Phase 6.3, D-40)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from xaglab.eval import edges as E

EDGES = [f"{d}@{lag}" for d in ("gold", "copper", "dxy", "real_yield_10y", "vix") for lag in (1, 5, 21)]


def _run(actives: list[list[str]], start="2010-01-04"):
    """A run-like object: one refit per month (21 business days), the given active sets."""
    idx = pd.bdate_range(start, periods=21 * len(actives), name="date")
    refit = np.repeat(np.arange(len(actives)), 21)
    pred = pd.DataFrame({"fold": 0, "refit": refit}, index=idx)
    diags = [{"fold": 0, "refit": j, "selection": {
        "active": a, "reliability": {e: (0.6 if e in a else 0.3) for e in EDGES},
        "contribution_mnats": {e: 1.0 for e in EDGES}, "clock": {"tau": j, "rate": 0.1, "shift": 0.01}}}
        for j, a in enumerate(actives)]
    return SimpleNamespace(run_id="fake", predictions=pred, diagnostics=diags)


def test_timeline_and_cluster_activity_by_hand():
    run = _run([["gold@1", "dxy@21", "vix@1"], ["copper@5"], []])
    tl = E.timeline(run)
    assert list(tl.index) == [pd.Timestamp("2010-01-04"), pd.Timestamp("2010-02-02"), pd.Timestamp("2010-03-03")]
    assert E.edges_of(tl) == EDGES and list(tl["n_edges"]) == [3, 1, 0]
    assert bool(tl["on:gold@1"].iloc[0]) and not bool(tl["on:gold@1"].iloc[1])
    act = E.cluster_activity(tl)
    assert np.isclose(act["act:monetary"].iloc[0], 2 / 9)       # gold@1, dxy@21 of 9 monetary edges
    assert np.isclose(act["act:risk"].iloc[0], 1 / 3) and np.isclose(act["act:industrial"].iloc[1], 1 / 3)
    assert np.isclose(act["rel:monetary"].iloc[0], (2 * 0.6 + 7 * 0.3) / 9)


def test_regime_flags_skip_regimes_before_the_test_period():
    dates = pd.DatetimeIndex(["2011-06-01", "2015-06-01", "2020-03-02", "2022-07-01"])
    f = E.regime_flags(dates)
    assert "2008 GFC" not in f.columns and "2013 taper tantrum" in f.columns
    assert list(f["stress"]) == [True, False, True, True]


def test_circular_shift_test_finds_a_planted_regime_effect_and_not_noise():
    rng = np.random.default_rng(0)
    n = 200
    inside = np.zeros(n, bool)
    inside[40:60] = True
    inside[120:135] = True
    planted = rng.normal(0, 0.1, n) + 0.3 * inside
    res = E.circular_shift_test(planted, inside, "greater")
    assert res["diff"] > 0.2 and res["p"] <= 0.01 and res["n_inside"] == 35
    assert E.circular_shift_test(planted, inside, "less")["p"] > 0.9
    # persistent noise (a random walk) with no regime effect: the test must reject rarely
    rejections = 0
    for k in range(40):
        r = np.random.default_rng(100 + k)
        walk = np.cumsum(r.normal(size=n))
        rejections += E.circular_shift_test(walk, inside, "greater")["p"] < 0.05
    assert rejections <= 5


def test_persistence_by_hand():
    run = _run([["gold@1"], ["gold@1", "vix@1"], ["vix@1"], ["vix@1"], []])
    per, overall = E.persistence(E.timeline(run))
    g = per.set_index("edge")
    assert np.isclose(g.loc["gold@1", "share_active"], 0.4) and g.loc["gold@1", "switches"] == 1
    assert np.isclose(g.loc["gold@1", "mean_spell_months"], 2.0)
    assert np.isclose(g.loc["vix@1", "share_active"], 0.6) and g.loc["vix@1", "mean_spell_months"] == 3.0
    # Jaccard: {g}|{g,v} = 1/2; {g,v}|{v} = 1/2; {v}|{v} = 1; {v}|{} = 0
    assert np.isclose(overall["mean_jaccard"], (0.5 + 0.5 + 1 + 0) / 4)
    assert overall["months_unchanged"] == 1 and np.isclose(overall["mean_edges_changed"], (1 + 1 + 0 + 1) / 4)
