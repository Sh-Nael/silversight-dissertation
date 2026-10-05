"""Known-answer tests for feature primitives, labels and scaling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from xaglab.features import price as P
from xaglab.features.build import make_labels
from xaglab.features.scaling import RobustScaler


def _bars(o, h, l, c, ok=True):
    idx = pd.bdate_range("2021-01-04", periods=len(c))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "bar_ok": ok}, index=idx)


def test_candle_anatomy():
    b = _bars([10, 10], [12, 10], [8, 10], [11, 10], ok=[True, False])
    assert P.close_location(b).tolist() == [0.5, 0.0]   # (22-12-8)/4, flat bar -> 0
    assert P.body_fraction(b).tolist() == [0.25, 0.0]


def test_garman_klass_single_bar():
    b = _bars([100.0], [110.0], [90.0], [105.0])
    expected = np.sqrt(0.5 * np.log(110 / 90) ** 2 - (2 * np.log(2) - 1) * np.log(1.05) ** 2)
    got = np.exp(P.garman_klass_vol(b, window=1, min_periods=1).iloc[0]) - 1e-8
    assert abs(got - expected) < 1e-12


def test_corwin_schultz_is_causal():
    rng = np.random.default_rng(0)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 80)))
    b = _bars(c, c * 1.01, c * 0.99, c)
    base = P.corwin_schultz(b, window=5, min_periods=2)
    b2 = b.copy()
    b2.iloc[50:, b2.columns.get_loc("high")] *= 1.5
    assert base.iloc[:50].equals(P.corwin_schultz(b2, window=5, min_periods=2).iloc[:50])


def test_labels_up_is_strict_and_aligned():
    close = pd.Series([1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0],
                      index=pd.bdate_range("2021-01-04", periods=7))
    y = make_labels(close)
    assert y["up1"].tolist()[:6] == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]  # unchanged is not up
    assert np.isnan(y["up1"].iloc[-1]) and y["up5"].isna().sum() == 5
    assert np.isclose(y["rv1"].iloc[1], np.log(2) ** 2)


def test_scaler_uses_only_fit_rows_and_clips():
    x = pd.DataFrame({"a": [1.0, 2, 3, 4, 5], "b": [0.0, 0, 0, 0, 1]})
    s = RobustScaler.fit(x.iloc[:4])
    assert s.center["a"] == 2.5 and s.scale["a"] == 1.5
    z = s.transform(pd.DataFrame({"a": [1000.0], "b": [np.nan]}))
    assert z.iloc[0].tolist() == [5.0, 0.0]


def test_real_feature_matrix_is_finite_after_warmup(features):
    X = features.pooled().loc[features.warmup_end():]
    assert np.isfinite(X.to_numpy(dtype=float)[~np.isnan(X.to_numpy(dtype=float))]).all()
    assert X.shape[1] == 95
