"""The forecasting contract shared by the research harness and the app.

Every model, from climatology to AIM-DG, implements :class:`Forecaster`. The harness
calls ``fit`` at each refit origin and ``predict`` for the origins that fit serves;
the app will call the same two methods nightly. What a model returns is a forecast
frame indexed by origin date with the columns in :data:`FORECAST_COLUMNS`. A column a
model does not produce is NaN (a direction-only model has no variance forecast).

Leakage is prevented by construction, not by convention: ``fit`` receives a
:class:`FitView` holding no row after the last training row, and ``predict``
receives a :class:`PredictView` whose labels have been removed entirely.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from xaglab.eval.folds import Refit
from xaglab.features.build import FeatureSet

#: p_up{h}: calibrated P(log return over (t, t+h] > 0); var{h}: forecast of the
#: realised variance (sum of squared daily log returns) over (t, t+h].
FORECAST_COLUMNS = ("p_up1", "p_up5", "var1", "var5")


def _slice_features(fs: FeatureSet, stop: int, with_labels: bool) -> FeatureSet:
    idx = fs.index[:stop]
    labels = fs.labels.iloc[:stop] if with_labels else fs.labels.iloc[:stop] * np.nan
    return FeatureSet(
        index=idx,
        nodes={k: v.iloc[:stop] for k, v in fs.nodes.items()},
        events=None if fs.events is None else fs.events.iloc[:stop],
        labels=labels,
        extras=None if fs.extras is None else {k: v.iloc[:stop] for k, v in fs.extras.items()},
    )


@dataclass(frozen=True)
class FitView:
    """What a model may see when fitting at a refit origin: rows up to ``train_end`` only."""

    data: FeatureSet
    refit: Refit
    seed: int

    @classmethod
    def build(cls, fs: FeatureSet, refit: Refit, seed: int) -> FitView:
        return cls(_slice_features(fs, refit.train_end + 1, with_labels=True), refit, seed)

    @property
    def train(self) -> slice:
        return self.refit.train

    @property
    def inner(self) -> slice:
        return self.refit.inner

    @property
    def val(self) -> slice:
        return self.refit.val

    def inner_for(self, h: int) -> slice:
        return self.refit.inner_for(h)

    def val_for(self, h: int) -> slice:
        return self.refit.val_for(h)


@dataclass(frozen=True)
class PredictView:
    """What a model may see when predicting: features up to the last origin, no labels."""

    data: FeatureSet
    rows: slice  # origins to forecast (positions)

    @classmethod
    def build(cls, fs: FeatureSet, refit: Refit) -> PredictView:
        return cls(_slice_features(fs, refit.predict_end + 1, with_labels=False), refit.predict)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.data.index[self.rows]


class Forecaster(ABC):
    """Base class. Subclasses set ``name`` and implement ``fit``/``predict``."""

    name: str = "forecaster"
    #: True if fit() fully resets state, so refits may run in parallel processes.
    stateless: bool = True
    #: True if the model has a tuning stage (tune / set_tuned / tune_every), see
    #: xaglab.models.tuning. The harness then tunes before fitting.
    tunable: bool = False

    @abstractmethod
    def fit(self, view: FitView) -> None: ...

    @abstractmethod
    def predict(self, view: PredictView) -> pd.DataFrame: ...

    def diagnostics(self) -> dict[str, Any]:
        """Optional per-refit extras persisted by the harness (e.g. active edges)."""
        return {}

    def config(self) -> dict[str, Any]:
        """Hyperparameters recorded with every run."""
        return {}


def empty_forecast(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=dates, columns=list(FORECAST_COLUMNS))


@dataclass
class Calibrator:
    """Platt scaling on the logit, fitted on validation predictions only."""

    a: float = 1.0
    b: float = 0.0
    fitted: bool = field(default=False)

    @classmethod
    def fit(cls, p: np.ndarray, y: np.ndarray) -> Calibrator:
        from sklearn.linear_model import LogisticRegression

        ok = ~(np.isnan(p) | np.isnan(y))
        if ok.sum() < 30 or len(np.unique(y[ok])) < 2:
            return cls()
        z = _logit(p[ok]).reshape(-1, 1)
        lr = LogisticRegression(C=1.0).fit(z, y[ok].astype(int))
        return cls(float(lr.coef_[0, 0]), float(lr.intercept_[0]), True)

    def __call__(self, p: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return p
        return 1.0 / (1.0 + np.exp(-(self.a * _logit(p) + self.b)))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))
