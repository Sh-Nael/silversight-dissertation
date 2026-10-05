"""Tuning framework shared by every tunable model (D-26).

The standard is real-world deployment: at each refit, tune and calibrate the model
the way you would before using it that day, with only the data known that day.

Three ideas, each a separate piece of this module:

1. **Rolling-origin cross-validation inside the training span** (:func:`cv_blocks`).
   Candidate settings are scored on ``n_blocks`` consecutive validation blocks at the
   end of the training span, each preceded by its own expanding, purged training set,
   and the losses are averaged. One block (the old rule) is too noisy a judge.

2. **Tune rarely, refit often.** Hyperparameters are searched (Optuna, TPE) only at
   *tuning points*, about once a year (:func:`tuning_points`). Every monthly refit
   reuses the latest settings but refreshes what must follow the current market:
   probability calibration and the variance scale, from out-of-fold predictions.

3. **Volatility relative to today's level** (:class:`VolatilityTask`). Models predict
   log(rv / baseline), where the baseline is a causal EWMA of silver's variance.
   A shift in the volatility level is absorbed by the baseline at once instead of
   waiting for a refit to notice it.

A model plugs in by supplying :class:`Learner` objects: a search space, a fit and a
predict. Everything else (blocks, search, out-of-fold predictions, scoring) is here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import optuna
import pandas as pd

from xaglab.eval.metrics import logloss

optuna.logging.set_verbosity(optuna.logging.WARNING)

#: Floor for realised variance before taking logs: (0.1%)^2.
RV_FLOOR = 1e-6
#: RiskMetrics decay for the causal variance baseline.
EWMA_LAMBDA = 0.94


# --------------------------------------------------------------------------- blocks

@dataclass(frozen=True)
class CVBlock:
    train: slice  # expanding inner training rows (purged before val)
    val: slice    # validation rows


def cv_blocks(train_start: int, train_end: int, purge: int, n_blocks: int = 3,
              block: int = 252, cap: float = 0.5) -> list[CVBlock]:
    """Consecutive validation blocks ending at ``train_end``, oldest first.

    Each block is ``block`` rows long, shrunk if needed so that all blocks together
    use at most ``cap`` of the training rows. Block k trains on everything from
    ``train_start`` up to ``purge`` rows before its validation starts.
    """
    n_train = train_end - train_start + 1
    length = min(block, int(n_train * cap) // n_blocks)
    if length < 20:
        raise ValueError(f"training span too short for {n_blocks} validation blocks")
    out = []
    for k in range(n_blocks):
        v_end = train_end - (n_blocks - 1 - k) * length
        v_start = v_end - length + 1
        out.append(CVBlock(slice(train_start, v_start - purge), slice(v_start, v_end + 1)))
    return out


def tuning_points(origins: list[int], every: int = 252) -> list[int]:
    """Index into ``origins`` (sorted refit origins) of each refit's tuning point.

    The first refit is a tuning point; so is every refit at least ``every`` origins
    after the previous tuning point. Returns, for each refit, the index of the refit
    whose tuning it uses.
    """
    out, last = [], None
    for i, o in enumerate(origins):
        if last is None or o - origins[last] >= every:
            last = i
        out.append(last)
    return out


# --------------------------------------------------------------------------- tasks

class Task(Protocol):
    kind: str

    def target(self, rows: slice) -> np.ndarray: ...
    def score(self, pred: np.ndarray, rows: np.ndarray) -> float: ...


@dataclass
class DirectionTask:
    """Binary outcome up_h; learners output probabilities; scored by log-loss."""

    y: np.ndarray
    kind: str = "direction"

    def target(self, rows: slice) -> np.ndarray:
        return self.y[rows]

    def score(self, pred: np.ndarray, rows: np.ndarray) -> float:
        return float(logloss(pred, self.y[rows]).mean())


def ewma_variance(r: pd.Series, lam: float = EWMA_LAMBDA) -> np.ndarray:
    """Causal RiskMetrics variance: value at t uses returns up to and including t."""
    return (r.fillna(0.0) ** 2).ewm(alpha=1 - lam, adjust=False).mean().to_numpy()


@dataclass
class VolatilityTask:
    """Realised variance over h days, learned relative to a causal baseline.

    target = log(rv + floor) - log(baseline), baseline = h * EWMA variance at t.
    Forecast = baseline * exp(prediction) * c, with c the QLIKE-optimal scale.
    """

    rv: np.ndarray
    base: np.ndarray
    kind: str = "volatility"

    def target(self, rows: slice) -> np.ndarray:
        return np.log(self.rv[rows] + RV_FLOOR) - np.log(self.base[rows])

    def scale(self, pred: np.ndarray, rows: np.ndarray) -> float:
        f = self.base[rows] * np.exp(pred)
        return float(np.mean(self.rv[rows] / f))

    def variance(self, pred: np.ndarray, rows: np.ndarray | slice, c: float) -> np.ndarray:
        return self.base[rows] * np.exp(pred) * c

    def score(self, pred: np.ndarray, rows: np.ndarray) -> float:
        c = self.scale(pred, rows)
        f = self.variance(pred, rows, c)
        return float(np.mean(np.log(f) + self.rv[rows] / f))


@dataclass
class JointTask:
    """All four forecasts from one network: direction and volatility at 1 and 5 days.

    Targets and predictions are dicts keyed ``up1, up5, vol1, vol5``. Direction
    predictions are probabilities; volatility predictions are relative log-variances
    ``z`` (as in VolatilityTask). The score is the **joint negative log-likelihood**:
    log-loss for each direction head plus one half of QLIKE for each volatility head,
    since QLIKE is (up to a constant) twice the Gaussian negative log-likelihood. All
    four terms are then in the same units, so no arbitrary weight is needed.
    """

    direction: dict[int, DirectionTask]
    volatility: dict[int, VolatilityTask]
    kind: str = "joint"

    def target(self, rows: slice) -> dict[str, np.ndarray]:
        out = {f"up{h}": t.target(rows) for h, t in self.direction.items()}
        out.update({f"vol{h}": t.target(rows) for h, t in self.volatility.items()})
        return out

    def components(self, pred: dict[str, np.ndarray], rows: np.ndarray) -> dict[str, float]:
        """The four parts, keyed like the single-task searches of TunedPooled
        (``direction_h1`` … ``volatility_h5``), so development scores line up across models."""
        out = {f"direction_h{h}": t.score(pred[f"up{h}"], rows) for h, t in self.direction.items()}
        out.update({f"volatility_h{h}": t.score(pred[f"vol{h}"], rows) for h, t in self.volatility.items()})
        return out

    def score(self, pred: dict[str, np.ndarray], rows: np.ndarray) -> float:
        c = self.components(pred, rows)
        return float(sum(v if k.startswith("direction") else 0.5 * v for k, v in c.items()))


# --------------------------------------------------------------------------- learners

class Learner(Protocol):
    """One estimator for one task. Must not look at rows it is not given."""

    def space(self, trial: optuna.Trial) -> dict[str, Any]: ...
    def fit(self, X: pd.DataFrame, y: np.ndarray, params: dict, seed: int) -> Any: ...
    def predict(self, fitted: Any, X: pd.DataFrame) -> np.ndarray: ...


class SequentialLearner(Protocol):
    """A learner that needs history before each row (sequence models).

    It receives the whole feature matrix, which never extends past the training cut-off,
    plus the row positions it may learn from or must predict. Windows for a row may look
    back into earlier rows (features are causal); targets are only read for ``rows``.
    """

    sequential: bool  # True

    def space(self, trial: optuna.Trial) -> dict[str, Any]: ...
    def fit_rows(self, X: pd.DataFrame, rows: slice, y: Any, params: dict, seed: int) -> Any: ...
    def predict_rows(self, fitted: Any, X: pd.DataFrame, rows: slice) -> Any: ...


def _concat(parts: list) -> Any:
    if isinstance(parts[0], dict):
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return np.concatenate(parts)


def cv_predict(learner: Learner | SequentialLearner, task: Task, X: pd.DataFrame,
               blocks: list[CVBlock], params: dict, seed: int) -> tuple[Any, np.ndarray]:
    """Out-of-fold predictions over all validation blocks, and their row positions.

    Predictions are arrays, or dicts of arrays for multi-output (joint) learners.
    """
    preds, rows = [], []
    for b in blocks:
        if getattr(learner, "sequential", False):
            fitted = learner.fit_rows(X, b.train, task.target(b.train), params, seed)
            preds.append(learner.predict_rows(fitted, X, b.val))
        else:
            fitted = learner.fit(X.iloc[b.train], task.target(b.train), params, seed)
            preds.append(learner.predict(fitted, X.iloc[b.val]))
        rows.append(np.arange(b.val.start, b.val.stop))
    return _concat(preds), np.concatenate(rows)


@dataclass
class TuneResult:
    params: dict[str, Any]
    cv_loss: float       # mean out-of-fold loss of the chosen settings (development score)
    n_trials: int
    seconds: float
    components: dict[str, float] | None = None  # multi-output tasks: each part's loss


def tune(learner: Learner, task: Task, X: pd.DataFrame, blocks: list[CVBlock],
         n_trials: int, seed: int) -> TuneResult:
    """Search the learner's space with Optuna TPE, minimising the out-of-fold loss."""
    t0 = time.perf_counter()

    def objective(trial: optuna.Trial) -> float:
        params = learner.space(trial)
        trial.set_user_attr("params", params)  # full set, incl. fixed values
        pred, rows = cv_predict(learner, task, X, blocks, params, seed)
        if hasattr(task, "components"):
            trial.set_user_attr("components", task.components(pred, rows))
        return task.score(pred, rows)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_trial
    return TuneResult(dict(best.user_attrs["params"]), float(study.best_value), n_trials,
                      time.perf_counter() - t0, best.user_attrs.get("components"))
