"""Leakage tests: the property the whole evaluation rests on.

For a set of cut dates T, everything strictly after T is replaced with a different
future (prices rescaled by random noise, driver rows deleted, unscheduled events
added). Nothing dated <= T may change: no feature, no event flag, no label whose
window closes by T, no fold boundary. A failure here is a build failure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xaglab.data.panel import build_panel_from_frames
from xaglab.eval.folds import make_calendar
from xaglab.features.build import HORIZONS, build_features

CUTS = ["2008-10-10", "2013-06-28", "2020-03-16", "2024-12-31"]


def _alternative_future(raw: dict, events: pd.DataFrame, cut: pd.Timestamp, seed: int):
    rng = np.random.default_rng(seed)
    out = {}
    for key, df in raw.items():
        df = df.copy()
        after = df.index > cut
        noise = np.exp(rng.normal(0, 0.05, after.sum()))
        for c in ("open", "high", "low", "close", "adj_close"):
            if c in df:
                df.loc[after, c] = df.loc[after, c].to_numpy() * noise
        if "volume" in df:
            df["volume"] = df["volume"].astype(float)
            df.loc[after, "volume"] = df.loc[after, "volume"].to_numpy() * rng.uniform(0.2, 5, after.sum())
        if key != "silver":  # silver defines the calendar; drop random driver rows after the cut
            drop = df.index[after][rng.random(after.sum()) < 0.05]
            df = df.drop(drop)
        out[key] = df
    extra = pd.DataFrame({"date": pd.bdate_range(cut + pd.Timedelta(days=1), periods=10).strftime("%Y-%m-%d"),
                          "event": ["fomc", "cpi"] * 5, "scheduled": False, "source": "test"})
    return out, pd.concat([events, extra], ignore_index=True)


@pytest.mark.parametrize("cut", CUTS)
def test_nothing_before_cut_depends_on_the_future(raw, calendar, features, cut):
    cut = pd.Timestamp(cut)
    events = calendar.read("events.csv")
    alt_raw, alt_events = _alternative_future(raw, events, cut, seed=hash(cut) % 2**32)
    alt = build_features(build_panel_from_frames(alt_raw, alt_events))

    past = features.index[features.index <= cut]
    for key, frame in features.nodes.items():
        pd.testing.assert_frame_equal(frame.loc[past], alt.nodes[key].loc[past],
                                      check_exact=True, obj=f"features[{key}]")
    pd.testing.assert_frame_equal(features.events.loc[past], alt.events.loc[past],
                                  check_exact=True, obj="event features")

    # Labels are forward-looking by definition: only those whose window closes by the
    # cut must be unchanged.
    pos_cut = features.index.get_loc(past[-1])
    for h in HORIZONS:
        known = features.index[: pos_cut - h + 1]
        cols = [f"ret{h}", f"up{h}", f"rv{h}"]
        pd.testing.assert_frame_equal(features.labels.loc[known, cols], alt.labels.loc[known, cols],
                                      check_exact=True, obj=f"labels h={h}")


def test_perturbation_actually_changes_the_future(raw, calendar, features):
    """Guard against a vacuous leakage test."""
    cut = pd.Timestamp(CUTS[1])
    alt_raw, alt_events = _alternative_future(raw, calendar.read("events.csv"), cut, seed=1)
    alt = build_features(build_panel_from_frames(alt_raw, alt_events))
    future = features.index[features.index > cut]
    assert not features.nodes["silver"].loc[future].equals(alt.nodes["silver"].loc[future])


def test_training_rows_never_see_labels_from_prediction_window(features):
    cal = make_calendar(features.index, features.labels, train_start=features.warmup_end())
    H = max(HORIZONS)
    for r in cal.all_refits():
        # Row t's label is known at t + H; it must be known at the refit origin.
        assert r.train_end + H <= r.origin
        assert r.inner_end + H < r.val_start
        assert r.train_start < r.inner_end < r.val_start <= r.train_end < r.origin <= r.predict_end
