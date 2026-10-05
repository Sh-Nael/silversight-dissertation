"""Tests for the shared tuning framework (D-26)."""

from __future__ import annotations

import json
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from xaglab.eval.folds import make_calendar
from xaglab.eval.harness import ModelSpec, develop, run
from xaglab.models.baselines import PooledLinear
from xaglab.models.tuning import (
    DirectionTask,
    VolatilityTask,
    cv_blocks,
    ewma_variance,
    tune,
    tuning_points,
)


def test_cv_blocks_are_consecutive_purged_and_capped():
    blocks = cv_blocks(train_start=100, train_end=3099, purge=5, n_blocks=3, block=252)
    assert [b.val.stop - b.val.start for b in blocks] == [252, 252, 252]
    assert blocks[-1].val.stop - 1 == 3099                      # last block ends the span
    for a, b in pairwise(blocks):
        assert b.val.start == a.val.stop                        # consecutive
    for b in blocks:
        assert b.train.start == 100 and b.train.stop - 1 + 5 < b.val.start  # purged
    small = cv_blocks(train_start=0, train_end=749, purge=5, n_blocks=3, block=252)
    assert sum(b.val.stop - b.val.start for b in small) <= 375  # capped at half
    with pytest.raises(ValueError):
        cv_blocks(0, 80, purge=5)


def test_tuning_points_yearly():
    origins = list(range(1000, 1000 + 21 * 30, 21))              # 30 monthly refits
    idx = tuning_points(origins, every=252)
    assert idx[0] == 0 and idx[11] == 0 and idx[12] == 12 and idx[24] == 24
    assert all(origins[i] - origins[idx[i]] < 252 for i in range(30))


def test_ewma_variance_is_causal():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 500))
    base = ewma_variance(r)
    r2 = r.copy()
    r2.iloc[300:] *= 10
    assert np.array_equal(base[:300], ewma_variance(r2)[:300])


def test_relative_volatility_target_adapts_to_a_level_shift():
    """Volatility doubles at day 1000. A model that learned only the average level
    (raw) is stale afterwards; the same model on the relative target tracks it."""
    rng = np.random.default_rng(1)
    sigma = np.r_[np.full(1000, 0.01), np.full(500, 0.02)]
    r = pd.Series(rng.normal(0, sigma))
    rv = (r.shift(-1) ** 2).to_numpy()
    base = ewma_variance(r)
    task = VolatilityTask(rv, base)
    train, test = slice(100, 990), np.arange(1050, 1490)
    z_const = np.nanmean(task.target(train))                    # relative model: one number
    rel = task.variance(np.full(len(test), z_const), test, task.scale(np.full(890, z_const), np.arange(100, 990)))
    raw = np.full(len(test), np.nanmean(rv[train]))             # raw model: training average
    q = lambda f: np.mean(np.log(f) + rv[test] / f)
    assert q(rel) < q(raw) - 0.3


def test_tune_is_deterministic_and_improves_on_a_planted_signal():
    from xaglab.models.baselines import LogisticLearner

    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(1500, 5)), columns=list("abcde"))
    y = (rng.random(1500) < 1 / (1 + np.exp(-0.8 * X["a"]))).astype(float).to_numpy()
    task = DirectionTask(y)
    blocks = cv_blocks(0, 1499, purge=5)
    r1 = tune(LogisticLearner(), task, X, blocks, n_trials=15, seed=3)
    r2 = tune(LogisticLearner(), task, X, blocks, n_trials=15, seed=3)
    assert r1.params == r2.params and r1.cv_loss == r2.cv_loss
    assert r1.cv_loss < np.log(2) - 0.03                        # beats a coin flip clearly


def test_tuned_model_end_to_end_on_one_fold(features):
    cal = make_calendar(features.index, features.labels, train_start=features.warmup_end())
    spec = ModelSpec(PooledLinear, {"n_trials": 3}, "linear")
    res = run(spec, features, cal, n_jobs=4, save=False, folds=[1])
    assert len(res.predictions) == cal.folds[1].test_end - cal.folds[1].test_start + 1
    assert res.tuning and len(res.tuning) == 3                  # 25 refits -> tuning at 0, 12, 24
    for t in res.tuning:
        assert pd.Timestamp(t["train_end"]) < pd.Timestamp(t["origin"])
        assert set(t["results"]) == {"direction_h1", "volatility_h1", "direction_h5", "volatility_h5"}
    p = res.predictions
    assert p[["p_up1", "p_up5", "var1", "var5"]].notna().all().all()


def test_development_run_then_exact_reuse_in_the_test_run(features, tmp_path):
    cal = make_calendar(features.index, features.labels, train_start=features.warmup_end())
    spec = ModelSpec(PooledLinear, {"n_trials": 2}, "linear")
    dev = develop(spec, features, cal, n_jobs=3, folds=[1], base=tmp_path)
    assert dev.manifest["stage"] == "development" and len(dev.tuning) == 3
    assert (dev.path / "tuning.json").exists() and not (dev.path / "predictions.parquet").exists()

    res = run(spec, features, cal, n_jobs=4, save=False, folds=[1], tuning_from=dev.path)
    assert res.manifest["tuning"]["from"] == dev.run_id
    assert [t["results"] for t in res.tuning] == [t["results"] for t in json.loads(
        (dev.path / "tuning.json").read_text())]                 # the settings judged are the settings tested

    with pytest.raises(ValueError, match="model_kwargs"):     # different keyword arguments
        run(ModelSpec(PooledLinear, {"n_trials": 3}, "linear"), features, cal, save=False, folds=[1],
            tuning_from=dev.path)
    changed = type("LinearOtherDefaults", (PooledLinear,), {"config": lambda self: {"changed": True}})
    with pytest.raises(ValueError, match="model_config"):     # same kwargs, different effective defaults
        run(ModelSpec(changed, {"n_trials": 2}, "linear"), features, cal, save=False, folds=[1],
            tuning_from=dev.path)
    with pytest.raises(ValueError, match="no tuning point"):  # tuning points the dev run never had
        run(spec, features, cal, save=False, folds=[2], tuning_from=dev.path)
