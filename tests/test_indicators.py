"""Tests for the optional technical-indicator block: MACD and Bollinger Bands (D-39)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from xaglab.features import price as P


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    """Textbook recursive EMA, written out: e_t = e_{t-1} + α (x_t − e_{t-1}), α = 2 / (span + 1)."""
    a, out = 2.0 / (span + 1), np.empty(len(x))
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = out[t - 1] + a * (x[t] - out[t - 1])
    return out


def test_macd_matches_the_textbook_recursion():
    rng = np.random.default_rng(0)
    close = pd.Series(30 * np.exp(np.cumsum(rng.normal(0, 0.01, 120))))
    m = P.macd(close)
    c = close.to_numpy()
    line = _ema(c, 12) - _ema(c, 26)
    sig = np.full(len(c), np.nan)
    sig[25:] = _ema(line[25:], 9)                     # the signal starts at the line's first value
    assert m["macd"].iloc[:25].isna().all() and m["macd_signal"].iloc[:33].isna().all()
    assert np.allclose(m["macd"].iloc[25:], (line / c)[25:])
    assert np.allclose(m["macd_signal"].iloc[33:], (sig / c)[33:])
    assert np.allclose(m["macd_hist"].iloc[33:], ((line - sig) / c)[33:])


def test_bollinger_by_hand():
    b = P.bollinger(pd.Series([1.0, 2.0, 3.0]), window=3, k=2.0)
    sd = np.sqrt(2.0 / 3.0)                           # population sd of 1, 2, 3
    assert b.iloc[:2].isna().all().all()
    assert np.isclose(b["bb_pctb"].iloc[2], (3 - (2 - 2 * sd)) / (4 * sd))     # 0.806
    assert np.isclose(b["bb_width"].iloc[2], 4 * sd / 2.0)                     # 1.633
    flat = P.bollinger(pd.Series([5.0] * 4), window=3)
    assert flat["bb_pctb"].isna().all() and (flat["bb_width"].iloc[2:] == 0).all()
    # the close on the middle band gives %B = 0.5
    assert np.isclose(P.bollinger(pd.Series([1.0, 3.0, 2.0]), window=3)["bb_pctb"].iloc[2], 0.5)


def test_indicators_never_look_ahead():
    rng = np.random.default_rng(1)
    close = pd.Series(20 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))))
    later = close.copy()
    later.iloc[200:] *= np.exp(rng.normal(0, 0.05, 100))          # a different future
    for f in (P.macd, P.bollinger):
        a, b = f(close), f(later)
        assert a.iloc[:200].equals(b.iloc[:200]) and not a.iloc[200:].equals(b.iloc[200:])


def test_the_block_is_optional_and_the_frozen_features_are_untouched(features):
    base = features.pooled()
    assert base.shape[1] == 95                                    # the frozen set
    silver = features.pooled(extras="silver")
    added = [c for c in silver.columns if c not in base.columns]
    assert added == ["silver__macd", "silver__macd_signal", "silver__macd_hist", "silver__bb_pctb", "silver__bb_width"]
    assert silver[base.columns].equals(base)
    every = features.pooled(extras="all")
    assert every.shape[1] == 95 + 5 * 5 and "vix__bb_width" in every.columns
    assert "real_yield_10y__macd" not in every.columns            # a yield level has no chart bands


def test_flat_models_opt_in_and_run(features):
    from xaglab.eval.folds import make_calendar
    from xaglab.eval.harness import ModelSpec, run
    from xaglab.models import registry
    from xaglab.models.baselines import PooledLinear

    assert {"linear_ta", "gbm_ta", "linear_ta_all", "gbm_ta_all"} <= set(registry.names())
    assert registry.get("linear").build().name == "linear" and registry.get("gbm_ta_all").build().name == "gbm_ta_all"
    cal = make_calendar(features.index, features.labels, train_start=features.warmup_end())
    res = run(ModelSpec(PooledLinear, {"extras": "silver", "n_trials": 2}, "linear_ta"), features, cal,
              n_jobs=4, save=False, folds=[1])
    assert res.manifest["model_config"]["extras"] == "silver"
    assert res.predictions[["p_up1", "p_up5", "var1", "var5"]].notna().all().all()
