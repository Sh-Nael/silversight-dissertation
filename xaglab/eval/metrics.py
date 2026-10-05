"""Per-observation losses and summary metrics.

Direction is judged first by proper scoring rules (Brier, log-loss) because they
reward honest probabilities (D-16); hit rate, balanced accuracy and AUC are
reported alongside. Volatility uses QLIKE in the Patton (2011) form
``log(f) + y/f``, robust to noisy realised-variance proxies and defined when the
realised value is zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EPS = 1e-6


def brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (p - y) ** 2


def logloss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def qlike(f: np.ndarray, y: np.ndarray) -> np.ndarray:
    f = np.maximum(f, 1e-12)
    return np.log(f) + y / f


def direction_summary(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    ok = ~(np.isnan(p) | np.isnan(y))
    p, y = p[ok], y[ok]
    if len(p) == 0:
        return {}
    pred = p > 0.5
    tp = ((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)
    tn = ((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)
    return {
        "n": len(p),
        "brier": float(brier(p, y).mean()),
        "logloss": float(logloss(p, y).mean()),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float((tp + tn) / 2),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "mean_p": float(p.mean()),
        "base_rate": float(y.mean()),
        # two-sided scorecard (D-32): silver is traded both ways, so each side is judged
        "up_calls": float(pred.mean()),
        "up_precision": float(y[pred].mean()) if pred.any() else float("nan"),
        "down_precision": float(1 - y[~pred].mean()) if (~pred).any() else float("nan"),
    }


def volatility_summary(f: np.ndarray, y: np.ndarray) -> dict[str, float]:
    ok = ~(np.isnan(f) | np.isnan(y))
    f, y = f[ok], y[ok]
    if len(f) == 0:
        return {}
    return {
        "n": len(f),
        "qlike": float(qlike(f, y).mean()),
        "rmse_vol": float(np.sqrt(np.mean((np.sqrt(f) - np.sqrt(y)) ** 2))),
        "mz_r2": float(np.corrcoef(f, y)[0, 1] ** 2) if f.std() > 0 else float("nan"),
    }


def summarise(pred: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Summary table from a harness prediction frame (forecast + label columns)."""
    groups = [("all", pred)] if by is None else list(pred.groupby(by))
    rows = []
    for g, df in groups:
        for h in (1, 5):
            d = direction_summary(df[f"p_up{h}"].to_numpy(float), df[f"up{h}"].to_numpy(float))
            v = volatility_summary(df[f"var{h}"].to_numpy(float), df[f"rv{h}"].to_numpy(float))
            rows.append({"group": g, "h": h, **d, **{f"vol_{k}": x for k, x in v.items() if k != "n"}})
    return pd.DataFrame(rows)
