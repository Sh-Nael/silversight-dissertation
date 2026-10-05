"""Baseline ladder (Objective 3). Each rung isolates one claim.

  climatology   base rate + average variance: "no information at all"
  ewma          climatology direction + RiskMetrics EWMA variance: the classic vol floor
  ar_garch      AR(1) mean + GARCH(1,1) Student-t variance on silver alone (the ARIMA/
                GARCH rung): do the drivers add anything beyond silver's own history?
  linear        L2 logistic regression (direction) + ridge regression (volatility) on
                all 95 pooled features: the flat linear model
  gbm           gradient-boosted trees on the same pooled features: the flat nonlinear model

The LSTM, static-graph GNN and AIM-DG rungs live in ``xaglab.models.neural`` and
``xaglab.models.graph``.

``linear`` and ``gbm`` are *tuned* models (:class:`TunedPooled`), using the shared
tuning framework in :mod:`xaglab.models.tuning` (D-26): yearly Optuna search on
3 purged rolling-origin validation blocks, monthly refits that refresh calibration and
the variance scale from out-of-fold predictions, and volatility learned relative to a
causal EWMA baseline. Every tunable model, neural ones included, follows the same rules.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from xaglab.features.scaling import RobustScaler
from xaglab.models.base import Calibrator, FitView, Forecaster, PredictView, empty_forecast
from xaglab.models.tuning import (
    DirectionTask,
    Learner,
    TuneResult,
    VolatilityTask,
    cv_blocks,
    cv_predict,
    ewma_variance,
    tune,
)

HORIZONS = (1, 5)


# --------------------------------------------------------------------------- naive

class Climatology(Forecaster):
    name = "climatology"

    def fit(self, view: FitView) -> None:
        y = view.data.labels.iloc[view.train]
        self.p = {h: float(y[f"up{h}"].mean()) for h in HORIZONS}
        self.v = {h: float(y[f"rv{h}"].mean()) for h in HORIZONS}

    def predict(self, view: PredictView) -> pd.DataFrame:
        out = empty_forecast(view.dates)
        for h in HORIZONS:
            out[f"p_up{h}"], out[f"var{h}"] = self.p[h], self.v[h]
        return out


class EWMA(Climatology):
    """RiskMetrics: sigma^2_t = lam * sigma^2_{t-1} + (1 - lam) * r_t^2. var_h = h * sigma^2."""

    name = "ewma"

    def __init__(self, lam: float = 0.94):
        self.lam = lam

    def config(self) -> dict:
        return {"lambda": self.lam}

    def predict(self, view: PredictView) -> pd.DataFrame:
        out = super().predict(view)
        r = view.data.nodes["silver"]["r1"].fillna(0.0)
        s2 = (r**2).ewm(alpha=1 - self.lam, adjust=False).mean()
        for h in HORIZONS:
            out[f"var{h}"] = h * s2.iloc[view.rows].to_numpy()
        return out


# --------------------------------------------------------------------------- AR-GARCH

class ARGarch(Forecaster):
    """AR(1)-GARCH(1,1) with Student-t errors on silver's daily log returns (in %).

    Parameters are estimated on training returns only; forecasts at each origin then
    filter the fixed-parameter model forward through returns up to that origin.
    """

    name = "ar_garch"

    def config(self) -> dict:
        return {"mean": "AR(1)", "vol": "GARCH(1,1)", "dist": "t"}

    def fit(self, view: FitView) -> None:
        from arch import arch_model

        r = 100 * view.data.nodes["silver"]["r1"].iloc[view.train].dropna()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = arch_model(r, mean="AR", lags=1, vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
        self.params = res.params

    def predict(self, view: PredictView) -> pd.DataFrame:
        from arch import arch_model
        from scipy import stats

        r = 100 * view.data.nodes["silver"]["r1"].dropna()
        am = arch_model(r, mean="AR", lags=1, vol="GARCH", p=1, q=1, dist="t")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = am.fix(self.params).forecast(horizon=5, start=view.dates[0], reindex=False)
        mean = fc.mean.reindex(view.dates).to_numpy()  # h.1 .. h.5, in %
        var = fc.variance.reindex(view.dates).to_numpy()
        nu = float(self.params["nu"])
        t_scale = np.sqrt((nu - 2) / nu)  # standardised-t -> t quantile scale
        out = empty_forecast(view.dates)
        out["p_up1"] = stats.t.sf(-mean[:, 0] / (np.sqrt(var[:, 0]) * t_scale), df=nu)
        m5, v5 = mean.sum(axis=1), var.sum(axis=1)
        out["p_up5"] = stats.norm.sf(-m5 / np.sqrt(v5))  # 5-day sum: CLT, normal
        out["var1"] = var[:, 0] / 1e4
        out["var5"] = v5 / 1e4
        return out


# --------------------------------------------------------------------------- tuned pooled models

class TunedPooled(Forecaster):
    """A model with one direction learner and one volatility learner per horizon,
    on the 95 pooled features, tuned and calibrated by the shared framework.

    Lifecycle driven by the harness:
      tune(view)        at a tuning point: search each (horizon, task) -> settings
      set_tuned(result) before each refit: which settings to use
      fit(view)         each monthly refit: out-of-fold predictions with the current
                        settings -> Platt calibration / variance scale; then fit on the
                        whole training span
      predict(view)     forecasts for the refit's origins
    """

    tunable = True
    direction_learner: Learner
    volatility_learner: Learner

    def __init__(self, n_trials: int = 40, n_blocks: int = 3, block: int = 252, tune_every: int = 252,
                 extras: str | None = None):
        self.n_trials, self.n_blocks, self.block, self.tune_every = n_trials, n_blocks, block, tune_every
        self.extras = extras  # None | "silver" | "all": the MACD and Bollinger block (D-39)
        if extras is not None:
            self.name = f"{type(self).name}_ta" + ("_all" if extras == "all" else "")
        self.tuned: dict[str, dict] = {}

    def config(self) -> dict:
        return {"extras": self.extras,
                "n_trials": self.n_trials, "cv_blocks": self.n_blocks, "block": self.block,
                "tune_every": self.tune_every, "sampler": "optuna TPE",
                "volatility_target": "log(rv / (h * EWMA variance)), EWMA lambda 0.94",
                "calibration": "Platt on out-of-fold predictions, refreshed every refit"}

    # ---- helpers
    def _tasks(self, data, h: int) -> dict[str, Any]:
        y = data.labels
        base = h * ewma_variance(data.nodes["silver"]["r1"])
        return {"direction": DirectionTask(y[f"up{h}"].to_numpy(float)),
                "volatility": VolatilityTask(y[f"rv{h}"].to_numpy(float), base)}

    def _learner(self, kind: str) -> Learner:
        return self.direction_learner if kind == "direction" else self.volatility_learner

    def _blocks(self, view: FitView):
        r = view.refit
        return cv_blocks(r.train_start, r.train_end, purge=r.origin - r.train_end,
                         n_blocks=self.n_blocks, block=self.block)

    # ---- lifecycle
    def tune(self, view: FitView) -> dict[str, dict]:
        X = view.data.pooled(extras=self.extras)
        blocks = self._blocks(view)
        out = {}
        for h in HORIZONS:
            for kind, task in self._tasks(view.data, h).items():
                res: TuneResult = tune(self._learner(kind), task, X, blocks, self.n_trials,
                                       seed=view.seed * 7919 + 13 * h + (kind == "volatility"))
                out[f"{kind}_h{h}"] = {"params": res.params, "cv_loss": res.cv_loss,
                                       "n_trials": res.n_trials, "seconds": round(res.seconds, 2)}
        return out

    def set_tuned(self, tuned: dict[str, dict]) -> None:
        self.tuned = tuned

    def fit(self, view: FitView) -> None:
        if not self.tuned:
            raise RuntimeError(f"{self.name}: fit() called before set_tuned()")
        X = view.data.pooled(extras=self.extras)
        self.cols = list(X.columns)
        blocks = self._blocks(view)
        self.fitted, self.calib, self.scale = {}, {}, {}
        for h in HORIZONS:
            for kind, task in self._tasks(view.data, h).items():
                key = f"{kind}_h{h}"
                params, learner = self.tuned[key]["params"], self._learner(kind)
                oof, rows = cv_predict(learner, task, X, blocks, params, view.seed)
                if kind == "direction":
                    self.calib[key] = Calibrator.fit(oof, task.y[rows])
                else:
                    self.scale[key] = task.scale(oof, rows)
                self.fitted[key] = learner.fit(X.iloc[view.train], task.target(view.train),
                                               params, view.seed)

    def predict(self, view: PredictView) -> pd.DataFrame:
        X = view.data.pooled(extras=self.extras)[self.cols]
        Xp = X.iloc[view.rows]
        out = empty_forecast(view.dates)
        for h in HORIZONS:
            d, v = f"direction_h{h}", f"volatility_h{h}"
            out[f"p_up{h}"] = self.calib[d](self.direction_learner.predict(self.fitted[d], Xp))
            base = h * ewma_variance(view.data.nodes["silver"]["r1"])[view.rows]
            z = self.volatility_learner.predict(self.fitted[v], Xp)
            out[f"var{h}"] = base * np.exp(z) * self.scale[v]
        return out

    def diagnostics(self) -> dict:
        return {"var_scale": self.scale,
                "calibration": {k: [c.a, c.b] for k, c in self.calib.items()}}


class _Scaled:
    """Robust scaling fitted on the rows a learner is trained on, applied forward."""

    @staticmethod
    def prepare(X: pd.DataFrame) -> tuple[RobustScaler, np.ndarray]:
        sc = RobustScaler.fit(X)
        return sc, sc.transform_np(X)


class LogisticLearner(_Scaled):
    def space(self, trial) -> dict:
        return {"C": trial.suggest_float("C", 1e-5, 10.0, log=True)}

    def fit(self, X, y, params, seed):
        from sklearn.linear_model import LogisticRegression

        sc, Z = self.prepare(X)
        return sc, LogisticRegression(C=params["C"], max_iter=3000).fit(Z, y.astype(int))

    def predict(self, fitted, X):
        sc, m = fitted
        return m.predict_proba(sc.transform_np(X))[:, 1]


class RidgeLearner(_Scaled):
    def space(self, trial) -> dict:
        return {"alpha": trial.suggest_float("alpha", 1e-2, 1e6, log=True)}

    def fit(self, X, y, params, seed):
        from sklearn.linear_model import Ridge

        sc, Z = self.prepare(X)
        return sc, Ridge(alpha=params["alpha"]).fit(Z, y)

    def predict(self, fitted, X):
        sc, m = fitted
        return m.predict(sc.transform_np(X))


def _gbm_space(trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "max_iter": trial.suggest_int("max_iter", 10, 500, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 3, 63, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 500, log=True),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        "max_features": trial.suggest_float("max_features", 0.2, 1.0),
    }


class GBMClassifierLearner:
    """Trees handle missing values natively, so no scaling or filling."""

    def space(self, trial) -> dict:
        return _gbm_space(trial)

    def fit(self, X, y, params, seed):
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(early_stopping=False, random_state=seed,
                                              **params).fit(X, y.astype(int))

    def predict(self, fitted, X):
        return fitted.predict_proba(X)[:, 1]


class GBMRegressorLearner:
    def space(self, trial) -> dict:
        return _gbm_space(trial)

    def fit(self, X, y, params, seed):
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(early_stopping=False, random_state=seed,
                                             **params).fit(X, y)

    def predict(self, fitted, X):
        return fitted.predict(X)


class PooledLinear(TunedPooled):
    """L2 logistic regression (direction) + ridge regression (volatility), tuned."""

    name = "linear"
    direction_learner = LogisticLearner()
    volatility_learner = RidgeLearner()


class PooledGBM(TunedPooled):
    """Histogram gradient boosting for both tasks, tuned."""

    name = "gbm"
    direction_learner = GBMClassifierLearner()
    volatility_learner = GBMRegressorLearner()
