"""NeuralLearner: plugs a network into the shared tuning framework (a SequentialLearner).

At every fit it: fits the scaler on the given training rows; builds multi-task targets
for those rows only; splits off purged early-stopping rows; trains `n_seeds` networks.
Predictions for any rows use the same scaler and average the ensemble. Everything a fit
learns comes from the rows it was given (the leakage rule), and windows only look back.

Round 2 of the neural development (D-30, declared before its results):
* ``refit_all``: after early stopping has chosen the number of epochs, each network is
  retrained from scratch on *all* the given rows for that many epochs, so the newest
  15% of the training span (up to ~2.5 years) is learned from, as a deployed model would.
* ``windows``: the window length is a searched setting (default {20, 60} days).
* wider regularisation: weight decay up to ``max_weight_decay`` (0.1), dropout up to
  ``max_dropout`` (0.5).
Round 1 is reproduced by ``refit_all=False, windows=(60,), max_weight_decay=1e-2,
max_dropout=0.4``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from xaglab.features.scaling import RobustScaler
from xaglab.models.neural import trainer
from xaglab.models.neural.data import prepare, rows_array
from xaglab.models.neural.nets import LAGS, SequenceNet, StaticGraphNet, column_groups

#: Supernet edge dropout: per-sample keep-rate q ~ U(0.2, 1) (D-34 decision 4).
EDGE_DROPOUT = (0.2, 1.0)


@dataclass
class Fitted:
    scaler: RobustScaler
    missing_cols: list[str]
    nets: list[nn.Module]
    info: list[dict[str, float]]  # per seed: best_epoch, epochs, seconds
    window: int
    stage1: list[nn.Module] | None = None  # early-stopped networks (not trained on the tail), if kept
    extra: dict[str, Any] | None = None     # model-specific results (AIM-DG's selection)


def _targets_for(y: dict[str, np.ndarray], rows: np.ndarray, n: int, dev: torch.device) -> dict[str, torch.Tensor]:
    """Full-length target tensors, NaN except on `rows` (JointTask gives log r for vol)."""
    out = {}
    for h in (1, 5):
        up = np.full(n, np.nan, np.float32)
        up[rows] = y[f"up{h}"]
        r = np.full(n, np.nan, np.float32)
        r[rows] = np.exp(y[f"vol{h}"])
        out[f"up{h}"] = torch.as_tensor(up, device=dev)
        out[f"r{h}"] = torch.as_tensor(r, device=dev)
    return out


class NeuralLearner:
    sequential = True
    #: keep the early-stopped networks (trained without the tail) alongside the all-rows refit;
    #: AIM-DG selects edges with them on the tail, which they have not seen
    keep_stage1 = False

    def __init__(self, net: str = "sequence", cell: str = "lstm", window: int = 60, n_seeds: int = 1,
                 max_epochs: int = 150, patience: int = 12, refit_all: bool = True,
                 windows: tuple[int, ...] = (20, 60), max_weight_decay: float = 1e-1,
                 max_dropout: float = 0.5, half_life: float | None = None):
        self.net, self.cell, self.window = net, cell, window
        self.n_seeds, self.max_epochs, self.patience = n_seeds, max_epochs, patience
        self.refit_all, self.windows = refit_all, tuple(windows)
        self.max_weight_decay, self.max_dropout = max_weight_decay, max_dropout
        self.half_life = half_life  # rows; None = every training row counts the same

    def space(self, trial) -> dict[str, Any]:
        out = {
            "hidden": trial.suggest_categorical("hidden", [16, 32, 64]),
            "layers": trial.suggest_int("layers", 1, 2),
            "dropout": trial.suggest_float("dropout", 0.0, self.max_dropout),
            "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, self.max_weight_decay, log=True),
            "lam": trial.suggest_float("lam", 0.1, 2.0, log=True),
        }
        if len(self.windows) > 1:
            out["window"] = trial.suggest_categorical("window", list(self.windows))
        if self.net in ("static_gnn", "supernet"):
            out["gnn_layers"] = trial.suggest_int("gnn_layers", 1, 2)
        return out

    def _lookback(self, window: int) -> int:
        """Days of history each sample needs: W, plus 20 for the graph's lag-21 states."""
        return window + (max(LAGS) - 1 if self.net in ("static_gnn", "supernet") else 0)

    def _builder(self, columns: list[str], params: dict, window: int) -> Callable[[], nn.Module]:
        if self.net == "sequence":
            return lambda: SequenceNet(len(columns), params["hidden"], params["layers"], params["dropout"], self.cell)
        if self.net in ("static_gnn", "supernet"):
            groups = column_groups(columns)
            drop = EDGE_DROPOUT if self.net == "supernet" else None
            return lambda: StaticGraphNet(groups, params["hidden"], params["layers"], params["gnn_layers"],
                                          params["dropout"], self.cell, window, edge_dropout=drop)
        raise ValueError(f"unknown network {self.net!r}")

    def fit_rows(self, X: pd.DataFrame, rows: slice, y: dict[str, np.ndarray], params: dict, seed: int,
                 n_seeds: int | None = None) -> Fitted:
        dev = trainer.device()
        r = rows_array(rows)
        window = int(params.get("window", self.window))
        prep = prepare(X, r, self._lookback(window), dev)
        tgt = _targets_for(y, r, len(X), dev)
        fit_r, val_r = trainer.early_stop_split(r)
        cfg = trainer.TrainConfig(lr=params["lr"], weight_decay=params["weight_decay"], lam=params["lam"],
                                  max_epochs=self.max_epochs, patience=self.patience)
        build = self._builder(prep.columns, params, window)
        w = None if self.half_life is None else trainer.recency_weights(r, self.half_life, len(X), dev)
        nets, info, stage1 = [], [], []
        for k in range(n_seeds or self.n_seeds):
            res = trainer.train(build, prep, tgt, fit_r, val_r, cfg, seed=seed * 101 + k, weights=w)
            stage1.append(res.net)
            seconds = res.seconds
            if self.refit_all:  # same seed, all rows, the epoch count early stopping chose
                res2 = trainer.train(build, prep, tgt, r, None, replace(cfg, max_epochs=max(res.best_epoch, 1)),
                                     seed=seed * 101 + k, weights=w)
                seconds += res2.seconds
                res.net = res2.net
            nets.append(res.net)
            info.append({"best_epoch": res.best_epoch, "epochs": res.epochs, "seconds": seconds})
        return Fitted(prep.scaler, prep.missing_cols, nets, info, window,
                      stage1=stage1 if self.keep_stage1 else None)

    def predict_rows(self, fitted: Fitted, X: pd.DataFrame, rows: slice) -> dict[str, np.ndarray]:
        prep = prepare(X, rows, self._lookback(fitted.window), trainer.device(), fitted.scaler, fitted.missing_cols)
        return trainer.predict(fitted.nets, prep, rows_array(rows))
