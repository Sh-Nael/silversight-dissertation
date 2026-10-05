"""Tests for the neural training machinery (Phase 3, step 3.2).

Run on the CPU (XAGLAB_DEVICE=cpu) so seeds reproduce exactly and the suite needs no GPU.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from xaglab.eval.folds import make_calendar
from xaglab.eval.harness import _check
from xaglab.eval.report import dev_paired, dev_scoreboard
from xaglab.models.base import FitView, PredictView
from xaglab.models.neural import trainer
from xaglab.models.neural.data import prepare
from xaglab.models.neural.forecaster import TunedNeural
from xaglab.models.neural.learner import NeuralLearner
from xaglab.models.neural.losses import qlike_rel
from xaglab.models.neural.nets import SequenceNet
from xaglab.models.tuning import DirectionTask, JointTask, VolatilityTask, cv_blocks, cv_predict


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setenv("XAGLAB_DEVICE", "cpu")


def test_windows_look_back_only():
    X = pd.DataFrame({"a": np.arange(10.0)})
    prep = prepare(X, slice(0, 10), window=3, device=torch.device("cpu"))
    w = prep.windows(torch.tensor([0, 5]))[:, :, 0]
    z = lambda v: (v - 4.5) / 4.5  # robust scaling: median 4.5, IQR 4.5
    assert torch.allclose(w[1], torch.tensor([z(3), z(4), z(5)], dtype=torch.float32))
    assert torch.allclose(w[0], torch.tensor([z(0)] * 3, dtype=torch.float32))  # repeats the first row


def test_scaler_uses_fit_rows_only_and_marks_missing():
    X = pd.DataFrame({"a": [1.0, 2, 3, 4, 1000], "v": [1.0, np.nan, 3, 4, 5]})
    p1 = prepare(X, slice(0, 4), 2, torch.device("cpu"))
    X2 = X.copy()
    X2.loc[4, "a"] = -5000  # a different "future" row
    p2 = prepare(X2, slice(0, 4), 2, torch.device("cpu"))
    assert p1.scaler.center.equals(p2.scaler.center)
    assert p1.columns == ["a", "v", "v__missing"] and float(p1.X[1, 2]) == 1.0


def test_qlike_rel_is_minimised_at_log_ratio():
    r = torch.tensor(2.5)
    zs = torch.linspace(-2, 3, 501)
    assert abs(float(zs[torch.argmin(qlike_rel(zs, r))]) - float(torch.log(r))) < 0.02


def test_early_stop_split_is_purged():
    fit, val = trainer.early_stop_split(np.arange(1000))
    assert len(val) == 150 and fit[-1] + trainer.PURGE < val[0]


def _memory_problem(n=1600, seed=0):
    """up(t+1) = 1 when x(t-3) > 0 (needs memory); volatility ratio depends on |x(t)|."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    noise = rng.normal(size=n)
    X = pd.DataFrame({"x": x, "noise": noise})
    lag3 = np.r_[np.full(3, np.nan), x[:-3]]
    up = (lag3 > 0).astype(float)
    ratio = np.exp(0.8 * np.abs(x) - 0.4)
    y = {"up1": up, "up5": up, "vol1": np.log(ratio), "vol5": np.log(ratio)}
    return X, y, up, ratio


def test_learner_learns_a_planted_memory_signal():
    X, y, up, ratio = _memory_problem()
    learner = NeuralLearner(window=8, max_epochs=300, patience=25)
    params = {"hidden": 16, "layers": 1, "dropout": 0.0, "lr": 3e-3, "weight_decay": 1e-5, "lam": 0.5}
    train, test = np.arange(10, 1300), np.arange(1300, 1600)
    # a 3-network ensemble, the deployed unit: after the all-rows refit (D-30) a single
    # network ends on its last epoch, which is noisier than an early-stopped checkpoint
    fitted = learner.fit_rows(X, train, {k: v[train] for k, v in y.items()}, params, seed=0, n_seeds=3)
    pred = learner.predict_rows(fitted, X, test)
    acc = np.mean((pred["up1"] > 0.5) == (up[test] > 0.5))
    assert acc > 0.9, acc                                            # memory was learned
    assert np.corrcoef(pred["vol1"], np.log(ratio[test]))[0, 1] > 0.8   # volatility head too
    assert fitted.info[0]["best_epoch"] <= fitted.info[0]["epochs"] <= 300


def test_same_seed_same_forecast_on_cpu():
    X, y, *_ = _memory_problem(n=600)
    learner = NeuralLearner(window=5, max_epochs=5, patience=5)
    params = {"hidden": 8, "layers": 1, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4, "lam": 1.0}
    rows = np.arange(10, 500)
    a = learner.predict_rows(learner.fit_rows(X, rows, {k: v[rows] for k, v in y.items()}, params, 7), X, np.arange(500, 600))
    b = learner.predict_rows(learner.fit_rows(X, rows, {k: v[rows] for k, v in y.items()}, params, 7), X, np.arange(500, 600))
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_ensemble_averages_probabilities_and_variances():
    class Const(torch.nn.Module):
        def __init__(self, logit, z):
            super().__init__()
            self.logit, self.z = logit, z

        def forward(self, x):
            n = x.shape[0]
            return {"logit1": torch.full((n,), self.logit), "logit5": torch.full((n,), self.logit),
                    "z1": torch.full((n,), self.z), "z5": torch.full((n,), self.z)}

    prep = prepare(pd.DataFrame({"a": np.arange(5.0)}), slice(0, 5), 2, torch.device("cpu"))
    out = trainer.predict([Const(0.0, 0.0), Const(np.log(3.0), np.log(3.0))], prep, np.arange(5))
    assert np.allclose(out["up1"], (0.5 + 0.75) / 2)
    assert np.allclose(out["vol1"], np.log((1 + 3) / 2))


def test_joint_task_scores_sequential_learner_out_of_fold():
    X, _y, up, ratio = _memory_problem(n=900)
    base = np.ones(len(X))
    task = JointTask({1: DirectionTask(up), 5: DirectionTask(up)},
                     {1: VolatilityTask(ratio - 1e-6, base), 5: VolatilityTask(ratio - 1e-6, 5 * base)})
    blocks = cv_blocks(10, 899, purge=5, n_blocks=2, block=150)
    learner = NeuralLearner(window=6, max_epochs=8, patience=8)
    params = {"hidden": 8, "layers": 1, "dropout": 0.0, "lr": 3e-3, "weight_decay": 1e-5, "lam": 0.5}
    pred, rows = cv_predict(learner, task, X, blocks, params, seed=1)
    assert set(pred) == {"up1", "up5", "vol1", "vol5"} and len(rows) == 300
    assert np.isfinite(task.score(pred, rows))


def test_joint_task_parts_and_total_by_hand():
    y = np.array([1.0, 0.0, 1.0, 1.0])
    rv, base = np.array([2.0, 1.0, 4.0, 1.0]), np.ones(4)
    task = JointTask({1: DirectionTask(y), 5: DirectionTask(y)},
                     {1: VolatilityTask(rv, base), 5: VolatilityTask(rv, base)})
    rows = np.arange(4)
    pred = {"up1": np.full(4, 0.5), "up5": np.full(4, 0.75), "vol1": np.zeros(4), "vol5": np.zeros(4)}
    c = task.components(pred, rows)
    assert np.isclose(c["direction_h1"], np.log(2))
    assert np.isclose(c["direction_h5"], -(3 * np.log(0.75) + np.log(0.25)) / 4)
    # z = 0 -> f = base * scale, scale = mean(rv / base) = 2; QLIKE = mean(ln 2 + rv / 2)
    assert np.isclose(c["volatility_h1"], np.log(2) + np.mean(rv) / 2)
    assert np.isclose(task.score(pred, rows),
                      c["direction_h1"] + c["direction_h5"] + 0.5 * (c["volatility_h1"] + c["volatility_h5"]))


def test_tuned_neural_through_one_real_refit(features):
    cal = make_calendar(features.index, features.labels, train_start=features.warmup_end())
    refit = cal.all_refits(None)[0]
    m = TunedNeural("gru", window=10, windows=(10,), n_seeds=2, tune_seeds=1, n_trials=2, max_epochs=2, patience=2)
    tuned = m.tune(FitView.build(features, refit, 0))
    assert set(tuned) == {"joint", "direction_h1", "direction_h5", "volatility_h1", "volatility_h5"}
    parts = {k: v["cv_loss"] for k, v in tuned.items()}
    assert np.isclose(parts["joint"], parts["direction_h1"] + parts["direction_h5"]
                      + 0.5 * (parts["volatility_h1"] + parts["volatility_h5"]))
    m.set_tuned(tuned)
    m.fit(FitView.build(features, refit, 0))
    view = PredictView.build(features, refit)
    pred = _check(m.predict(view), view.dates, m.name)      # harness checks: dates, ranges, positivity
    assert pred.notna().all().all() and m.name == "gru"
    assert len(m.diagnostics()["best_epoch"]) == 2


def test_dev_scores_derive_the_joint_score_and_pair_models():
    from types import SimpleNamespace as NS

    def fake(model, points):
        return NS(manifest={"model": model}, tuning=[
            {"origin": o, "results": {k: {"cv_loss": v} for k, v in r.items()}} for o, r in points.items()])

    lin = fake("linear", {"2010": {"direction_h1": 0.69, "direction_h5": 0.70,
                                   "volatility_h1": -7.0, "volatility_h5": -5.0}})
    nn_ = fake("lstm", {"2010": {"joint": -4.62, "direction_h1": 0.68, "direction_h5": 0.70,
                                 "volatility_h1": -7.0, "volatility_h5": -5.0}})
    board = dev_scoreboard([lin, nn_]).set_index(["model", "task"])["mean"]
    assert np.isclose(board["linear", "joint"], 0.69 + 0.70 - 6.0)
    paired = dev_paired([lin, nn_], "linear").set_index("task")
    assert np.isclose(paired.loc["joint", "mean_diff"], -4.62 - (1.39 - 6.0))
    assert paired.loc["direction_h1", "won"] == 1 and paired.loc["volatility_h1", "won"] == 0


def test_training_without_holdout_runs_exactly_the_given_epochs():
    X, y, *_ = _memory_problem(n=400)
    rows = np.arange(10, 400)
    prep = prepare(X, rows, 5, torch.device("cpu"))
    from xaglab.models.neural.learner import _targets_for
    tgt = _targets_for({k: v[rows] for k, v in y.items()}, rows, len(X), torch.device("cpu"))
    build = lambda: SequenceNet(prep.X.shape[1], 8, 1, 0.0, "lstm")
    res = trainer.train(build, prep, tgt, rows, None, trainer.TrainConfig(max_epochs=3), seed=0)
    assert res.epochs == 3 and res.best_epoch == 3


def test_refit_on_all_rows_and_searched_window():
    X, y, *_ = _memory_problem(n=700)
    rows = np.arange(10, 600)
    params = {"hidden": 8, "layers": 1, "dropout": 0.0, "lr": 3e-3, "weight_decay": 1e-5, "lam": 0.5, "window": 7}
    yy = {k: v[rows] for k, v in y.items()}
    held = NeuralLearner(window=5, max_epochs=6, patience=6, refit_all=False).fit_rows(X, rows, yy, params, 1)
    full = NeuralLearner(window=5, max_epochs=6, patience=6, refit_all=True).fit_rows(X, rows, yy, params, 1)
    assert held.window == full.window == 7                      # the searched window wins over the default
    assert held.info[0]["best_epoch"] == full.info[0]["best_epoch"]
    a = NeuralLearner().predict_rows(held, X, np.arange(600, 700))
    b = NeuralLearner().predict_rows(full, X, np.arange(600, 700))
    assert not np.allclose(a["up1"], b["up1"])                  # the all-rows refit is a different network

    old = NeuralLearner(refit_all=False, windows=(60,), max_weight_decay=1e-2, max_dropout=0.4)
    import optuna
    trial = optuna.create_study().ask()
    assert "window" not in old.space(trial)                     # round 1 (D-28) is reproducible
