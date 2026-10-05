"""From the feature matrix to what a network trains on.

* The robust scaler is fitted on the rows the network learns from, and applied to the
  whole matrix, which never extends past the training cut-off, so nothing leaks.
* Missing values become 0 after scaling (the median). Columns that are ever missing in
  the fit rows get an extra 0/1 "was missing" column, so the network can tell a
  masked volume day from an ordinary one.
* A sample at row t is the window of rows t-W+1 … t. Rows before the first one repeat
  the first row (in practice training starts a year after the data does).
* Targets: up1/up5 as 0/1; volatility as the ratio r = (rv + floor) / baseline with
  baseline = h × causal EWMA variance (the relative target of D-26).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from xaglab.features.scaling import RobustScaler
from xaglab.models.tuning import RV_FLOOR, ewma_variance

HORIZONS = (1, 5)


@dataclass
class Prepared:
    """Scaled inputs on the device, ready for windowing."""

    X: torch.Tensor          # [N, F] float32
    columns: list[str]
    scaler: RobustScaler
    missing_cols: list[str]  # columns that get a "was missing" indicator
    window: int

    def windows(self, idx: torch.Tensor) -> torch.Tensor:
        """[B] row positions → [B, W, F] windows ending at each row."""
        offs = torch.arange(-self.window + 1, 1, device=idx.device)
        rows = (idx[:, None] + offs[None, :]).clamp(min=0)
        return self.X[rows]


def prepare(X: pd.DataFrame, fit_rows: slice | np.ndarray, window: int, device: torch.device,
            scaler: RobustScaler | None = None, missing_cols: list[str] | None = None) -> Prepared:
    """Scale X with a scaler fitted on `fit_rows` (or a given one) and move it to `device`."""
    fit = X.iloc[fit_rows]
    if scaler is None:
        scaler = RobustScaler.fit(fit)
        missing_cols = [c for c in X.columns if fit[c].isna().any()]
    Z = scaler.transform(X, fill=0.0)
    if missing_cols:
        ind = X[missing_cols].isna().astype(np.float32).add_suffix("__missing")
        Z = pd.concat([Z, ind], axis=1)
    t = torch.as_tensor(Z.to_numpy(dtype=np.float32, copy=True), device=device)
    return Prepared(t, list(Z.columns), scaler, missing_cols or [], window)


def targets(labels: pd.DataFrame, silver_r1: pd.Series, device: torch.device) -> dict[str, torch.Tensor]:
    """Multi-task targets for every row (NaN where a label is unknown)."""
    base = ewma_variance(silver_r1)
    out: dict[str, torch.Tensor] = {}
    for h in HORIZONS:
        out[f"up{h}"] = torch.as_tensor(labels[f"up{h}"].to_numpy(np.float32), device=device)
        ratio = (labels[f"rv{h}"].to_numpy() + RV_FLOOR) / (h * base)
        out[f"r{h}"] = torch.as_tensor(ratio.astype(np.float32), device=device)
    return out


def rows_array(rows: slice | np.ndarray) -> np.ndarray:
    return np.arange(rows.start, rows.stop) if isinstance(rows, slice) else np.asarray(rows)
