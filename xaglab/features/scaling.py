"""Robust scaling fitted strictly inside a training window, then applied forward.

Median/IQR rather than mean/std because silver's returns are heavy-tailed (one
2011- or 2026-style day would dominate a standard deviation). Scaled values are
clipped at +-``clip`` so a single extreme day cannot saturate a network.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RobustScaler:
    center: pd.Series
    scale: pd.Series
    clip: float = 5.0

    @classmethod
    def fit(cls, x: pd.DataFrame, clip: float = 5.0) -> RobustScaler:
        center = x.median()
        iqr = x.quantile(0.75) - x.quantile(0.25)
        # Near-constant or binary columns: fall back to std, then to 1.
        scale = iqr.where(iqr > 1e-12, x.std()).where(lambda s: s > 1e-12, 1.0)
        return cls(center.fillna(0.0), scale.fillna(1.0), clip)

    def transform(self, x: pd.DataFrame, fill: float | None = 0.0) -> pd.DataFrame:
        z = ((x - self.center) / self.scale).clip(-self.clip, self.clip)
        return z if fill is None else z.fillna(fill)

    def transform_np(self, x: pd.DataFrame, fill: float = 0.0) -> np.ndarray:
        return self.transform(x, fill).to_numpy(dtype=np.float32)
