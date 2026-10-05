"""Harness integration tests: views hide what they must, runs are complete and aligned."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xaglab.eval.folds import make_calendar
from xaglab.eval.harness import ModelSpec, run
from xaglab.models.base import FitView, Forecaster, PredictView, empty_forecast
from xaglab.models.baselines import Climatology


@pytest.fixture(scope="module")
def cal(features):
    return make_calendar(features.index, features.labels, train_start=features.warmup_end())


def test_views_expose_nothing_beyond_their_cutoff(features, cal):
    r = cal.refits(cal.folds[3])[7]
    fv = FitView.build(features, r, seed=0)
    assert fv.data.index[-1] == features.index[r.train_end]
    assert all(len(f) == r.train_end + 1 for f in fv.data.nodes.values())
    pv = PredictView.build(features, r)
    assert pv.data.index[-1] == features.index[r.predict_end]
    assert pv.data.labels.isna().all().all()  # no labels at prediction time, ever


class _Spy(Forecaster):
    """Records what it was shown; predicts the base rate."""

    name = "spy"
    stateless = False

    def __init__(self):
        self.seen = []

    def fit(self, view: FitView) -> None:
        self.seen.append((view.refit.origin, len(view.data.index)))

    def predict(self, view: PredictView) -> pd.DataFrame:
        out = empty_forecast(view.dates)
        out[["p_up1", "p_up5"]] = 0.5
        return out


def test_run_covers_every_test_origin_once(features, cal):
    res = run(ModelSpec(Climatology, {}, "climatology"), features, cal, save=False)
    p = res.predictions
    expected = sum(f.test_end - f.test_start + 1 for f in cal.folds)
    assert len(p) == expected == 4141 and p.index.is_unique
    assert p[["up1", "up5", "rv1", "rv5"]].notna().all().all()
    assert sorted(p["fold"].unique()) == list(range(8))
    assert res.manifest["data"]["dissertation_v1"] and res.manifest["code"]["commit"]


def test_spy_never_receives_rows_past_the_purge(features, cal):
    spy = _Spy()
    spec = ModelSpec(_Spy, {}, "spy")
    from xaglab.eval.harness import _one_refit

    for r in cal.all_refits()[:30]:
        _one_refit(spy, features, r, 0)
    for origin, n_rows in spy.seen:
        assert n_rows == origin - cal.purge + 1
    assert spec.build().name == "spy"


def test_bad_forecasts_are_rejected(features, cal):
    class Broken(Climatology):
        name = "broken"

        def predict(self, view):
            out = super().predict(view)
            out["p_up1"] = 1.5
            return out

    with pytest.raises(ValueError, match="outside"):
        run(ModelSpec(Broken, {}, "broken"), features, cal, save=False)
    assert np.isfinite(0.0)


def test_ensemble_averages_members_and_round_trips(features, cal, tmp_path):
    from xaglab.eval.ensemble import ensemble
    from xaglab.eval.harness import ModelSpec, load_run, run, save_run
    from xaglab.models.baselines import EWMA, Climatology

    a = run(ModelSpec(Climatology, {}, "climatology"), features, cal, save=False, folds=[1])
    b = run(ModelSpec(EWMA, {"lam": 0.94}, "ewma"), features, cal, save=False, folds=[1])
    e = ensemble([a, b], "ens")
    assert np.allclose(e.predictions["var1"], (a.predictions["var1"] + b.predictions["var1"]) / 2)
    assert np.allclose(e.predictions["p_up5"], a.predictions["p_up5"])   # identical members' direction
    assert e.predictions["up1"].equals(a.predictions["up1"])              # labels carried over
    path = save_run(e, tmp_path)
    back = load_run(path)
    assert back.manifest["model"] == "ens" and [m["model"] for m in back.manifest["members"]] == ["climatology", "ewma"]
    assert back.predictions.equals(e.predictions) and back.tuning is None
    with pytest.raises(ValueError, match="different origins"):
        ensemble([a, run(ModelSpec(EWMA, {}, "ewma"), features, cal, save=False, folds=[2])])
