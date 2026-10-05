"""A synthetic market with a planted regime break, for AIM-DG's integration test (step 4.7).

Silver's next-day direction is driven by **gold four days earlier** (the gold lag-5 edge)
until the break, and by **copper today** (the copper lag-1 edge) after it. The other drivers
are noise. Volatility labels carry no signal, so only direction separates the masks.

Everything else matches the real data's shape: node names and feature columns, labels, and
monthly refits with an expanding window, a 5-day purge and 21-day prediction blocks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xaglab.eval.folds import Refit
from xaglab.features.build import FeatureSet

DRIVERS = ("gold", "copper", "dxy", "real_yield_10y", "vix")


def planted_break_market(n: int = 1300, brk: int = 900, noise: float = 0.3, seed: int = 0) -> FeatureSet:
    """``up1`` at t is 1 when gold's return at t−4 (before ``brk``) or copper's at t (from
    ``brk`` on), plus a little noise, is positive."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n)
    r = {k: rng.normal(0, 0.01, n) for k in ("silver", *DRIVERS)}
    lag_gold = np.r_[np.zeros(4), r["gold"][:-4]]
    score = np.where(np.arange(n) < brk, lag_gold, r["copper"]) + noise * 0.01 * rng.normal(size=n)
    up = (score > 0).astype(float)
    up[:4] = np.nan
    nodes = {k: pd.DataFrame({"r1": v, "r5": pd.Series(v).rolling(5, min_periods=1).sum().to_numpy()}, index=idx)
             for k, v in r.items()}
    labels = pd.DataFrame({"up1": up, "up5": up,
                           "rv1": 1e-4 * np.exp(0.3 * rng.normal(size=n)),
                           "rv5": 5e-4 * np.exp(0.3 * rng.normal(size=n)),
                           "ret1": 0.0, "ret5": 0.0}, index=idx)
    return FeatureSet(index=idx, nodes=nodes, events=None, labels=labels)


def monthly_refits(first_origin: int, n_refits: int, train_start: int = 30, every: int = 21,
                   purge: int = 5) -> list[Refit]:
    """Monthly refits in calendar order: expanding window, purged, 21-day prediction blocks."""
    out = []
    for j in range(n_refits):
        origin = first_origin + j * every
        end = origin - purge
        out.append(Refit(fold=0, j=j, origin=origin, train_start=train_start, train_end=end,
                         val_start=end - 60, inner_end=end - 65, predict_end=origin + every - 1))
    return out
