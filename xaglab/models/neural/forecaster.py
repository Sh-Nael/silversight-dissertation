"""TunedNeural: a multi-task network as a walk-forward forecaster (the `lstm` and `gru` models).

It follows the same tuning framework as `linear` and `gbm` (D-26), with one
difference: a single network produces all four forecasts, so there is **one** search per
tuning point, on the joint negative log-likelihood (D-28), instead of four.

Lifecycle driven by the harness:
  tune(view)        at a tuning point: Optuna TPE over the learner's space, each trial
                    scored out-of-fold on 3 purged rolling blocks -> settings, joint
                    development score and its four parts
  set_tuned(result) before each refit: which settings to use
  fit(view)         each monthly refit: out-of-fold predictions of ``n_seeds``-network
                    ensembles -> Platt calibration (per horizon) and variance scale (per
                    horizon); then an ``n_seeds`` ensemble trained on the whole training span
  predict(view)     forecasts for the refit's origins; windows look back from each origin

Each trial is scored as it would be deployed: a ``tune_seeds``-network ensemble (default
5, D-29). Measured at the last tuning point, the seed-to-seed spread of a single network's
joint score (sd 0.0135) was as large as the spread between settings (0.0146), so with one
network per trial the search would mostly be choosing seeds.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from xaglab.models.base import Calibrator, FitView, Forecaster, PredictView, empty_forecast
from xaglab.models.neural.learner import NeuralLearner
from xaglab.models.tuning import (
    DirectionTask,
    JointTask,
    VolatilityTask,
    cv_blocks,
    cv_predict,
    ewma_variance,
    tune,
)

HORIZONS = (1, 5)


def joint_task(data) -> JointTask:
    """The four targets for a FeatureSet: up1/up5 and relative variance at 1 and 5 days."""
    y, s2 = data.labels, ewma_variance(data.nodes["silver"]["r1"])
    return JointTask({h: DirectionTask(y[f"up{h}"].to_numpy(float)) for h in HORIZONS},
                     {h: VolatilityTask(y[f"rv{h}"].to_numpy(float), h * s2) for h in HORIZONS})


class TunedNeural(Forecaster):
    tunable = True

    def __init__(self, cell: str = "lstm", net: str = "sequence", window: int = 60, n_seeds: int = 5,
                 tune_seeds: int = 5, n_trials: int = 20, n_blocks: int = 3, block: int = 252,
                 tune_every: int = 252, max_epochs: int = 150, patience: int = 12,
                 refit_all: bool = True, windows: tuple[int, ...] = (20, 60),
                 max_weight_decay: float = 1e-1, max_dropout: float = 0.5, half_life: float | None = None):
        self.name = cell if net == "sequence" else net
        self.cell, self.net, self.window = cell, net, window
        self.n_seeds, self.tune_seeds, self.n_trials = n_seeds, tune_seeds, n_trials
        self.n_blocks, self.block, self.tune_every = n_blocks, block, tune_every
        self.max_epochs, self.patience = max_epochs, patience
        self.refit_all, self.windows = refit_all, tuple(windows)
        self.max_weight_decay, self.max_dropout = max_weight_decay, max_dropout
        self.half_life = half_life
        self.tuned: dict[str, dict] = {}

    def config(self) -> dict:
        return {"net": self.net, "cell": self.cell, "window": self.window, "windows": list(self.windows),
                "half_life_rows": self.half_life,
                "refit_all": self.refit_all, "max_weight_decay": self.max_weight_decay,
                "max_dropout": self.max_dropout, "n_seeds": self.n_seeds,
                "tune_seeds": self.tune_seeds, "n_trials": self.n_trials, "cv_blocks": self.n_blocks,
                "block": self.block, "tune_every": self.tune_every, "max_epochs": self.max_epochs,
                "patience": self.patience, "sampler": "optuna TPE",
                "objective": "joint NLL: logloss1 + logloss5 + 0.5 (QLIKE1 + QLIKE5)",
                "volatility_target": "log(rv / (h * EWMA variance)), EWMA lambda 0.94",
                "calibration": "Platt on out-of-fold ensemble predictions, refreshed every refit"}

    def _learner(self, n_seeds: int) -> NeuralLearner:
        return NeuralLearner(self.net, self.cell, self.window, n_seeds, self.max_epochs, self.patience,
                             self.refit_all, self.windows, self.max_weight_decay, self.max_dropout,
                             half_life=self.half_life)

    def _blocks(self, view: FitView):
        r = view.refit
        return cv_blocks(r.train_start, r.train_end, purge=r.origin - r.train_end,
                         n_blocks=self.n_blocks, block=self.block)

    # ---- lifecycle
    def tune(self, view: FitView) -> dict[str, dict]:
        res = tune(self._learner(self.tune_seeds), joint_task(view.data), view.data.pooled(),
                   self._blocks(view), self.n_trials, seed=view.seed * 7919 + 17)
        out: dict[str, dict[str, Any]] = {
            "joint": {"params": res.params, "cv_loss": res.cv_loss, "n_trials": res.n_trials,
                      "seconds": round(res.seconds, 2)}}
        # The parts of the chosen trial, under the same keys as linear/gbm's searches.
        out.update({k: {"cv_loss": v} for k, v in (res.components or {}).items()})
        return out

    def set_tuned(self, tuned: dict[str, dict]) -> None:
        self.tuned = tuned

    def fit(self, view: FitView) -> None:
        if not self.tuned:
            raise RuntimeError(f"{self.name}: fit() called before set_tuned()")
        X = view.data.pooled()
        self.cols = list(X.columns)
        task, params = joint_task(view.data), self.tuned["joint"]["params"]
        self.learner = self._learner(self.n_seeds)
        self._before_oof(self.learner)
        oof, rows = cv_predict(self.learner, task, X, self._blocks(view), params, view.seed)
        self.calib = {h: Calibrator.fit(oof[f"up{h}"], task.direction[h].y[rows]) for h in HORIZONS}
        self.scale = {h: task.volatility[h].scale(oof[f"vol{h}"], rows) for h in HORIZONS}
        self._before_final(self.learner)
        self.fitted = self.learner.fit_rows(X, view.train, task.target(view.train), params, view.seed)
        self._after_final()

    # hooks for stateful subclasses (AIM-DG): what the out-of-fold fits and the final fit know
    def _before_oof(self, learner: NeuralLearner) -> None: ...
    def _before_final(self, learner: NeuralLearner) -> None: ...
    def _after_final(self) -> None: ...

    def predict(self, view: PredictView) -> pd.DataFrame:
        X = view.data.pooled()[self.cols]
        pred = self.learner.predict_rows(self.fitted, X, view.rows)
        s2 = ewma_variance(view.data.nodes["silver"]["r1"])[view.rows]
        out = empty_forecast(view.dates)
        for h in HORIZONS:
            out[f"p_up{h}"] = self.calib[h](pred[f"up{h}"])
            out[f"var{h}"] = h * s2 * np.exp(pred[f"vol{h}"]) * self.scale[h]
        return out

    def diagnostics(self) -> dict:
        info = self.fitted.info
        return {"var_scale": {f"volatility_h{h}": c for h, c in self.scale.items()},
                "calibration": {f"direction_h{h}": [c.a, c.b] for h, c in self.calib.items()},
                "best_epoch": [int(i["best_epoch"]) for i in info],
                "epochs": [int(i["epochs"]) for i in info],
                "train_seconds": round(sum(i["seconds"] for i in info), 2)}
