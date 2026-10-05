"""Per-node feature primitives. Every function is causal: the value at t uses bars <= t.

Definitions follow the core set and the price-action and volume set (D-09), with two
deliberate corrections (D-19):
  * Corwin-Schultz is computed on bars (t-1, t), not (t, t+1) as in the paper's
    two-day definition, which would read tomorrow's bar.
  * Range-based quantities use only bars with real range information
    (``bar_ok``); flat or filled bars contribute nothing rather than a zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LN2 = np.log(2.0)
CS_K = 3.0 - 2.0 * np.sqrt(2.0)


def log_return(close: pd.Series, k: int = 1) -> pd.Series:
    return np.log(close / close.shift(k))


def realised_vol(close: pd.Series, window: int = 21, min_periods: int = 15) -> pd.Series:
    return log_return(close).rolling(window, min_periods=min_periods).std()


def garman_klass_vol(bars: pd.DataFrame, window: int = 21, min_periods: int = 10) -> pd.Series:
    """log of sqrt(mean GK variance) over valid bars in the window."""
    hl = np.log(bars["high"] / bars["low"])
    co = np.log(bars["close"] / bars["open"])
    var = (0.5 * hl**2 - (2 * LN2 - 1) * co**2).clip(lower=0)
    var = var.where(bars["bar_ok"])
    return np.log(np.sqrt(var.rolling(window, min_periods=min_periods).mean()) + 1e-8)


def atr_norm(bars: pd.DataFrame, window: int = 14) -> pd.Series:
    """Wilder ATR over true range, divided by close. True range includes the gap."""
    prev = bars["close"].shift(1)
    tr = pd.concat([bars["high"] - bars["low"], (bars["high"] - prev).abs(),
                    (bars["low"] - prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    return atr / bars["close"]


def close_location(bars: pd.DataFrame) -> pd.Series:
    rng = bars["high"] - bars["low"]
    clv = (2 * bars["close"] - bars["high"] - bars["low"]) / rng
    return clv.where(bars["bar_ok"], 0.0)


def body_fraction(bars: pd.DataFrame) -> pd.Series:
    rng = bars["high"] - bars["low"]
    body = (bars["close"] - bars["open"]) / rng
    return body.where(bars["bar_ok"], 0.0)


def overnight_gap(bars: pd.DataFrame) -> pd.Series:
    """ln(open_t / close_{t-1}); zero on bars whose open is not a real open."""
    gap = np.log(bars["open"] / bars["close"].shift(1))
    return gap.where(bars["bar_ok"], 0.0)


def ma_distance(close: pd.Series, window: int) -> pd.Series:
    return np.log(close / close.rolling(window, min_periods=window).mean())


def ts_momentum(close: pd.Series, lookback: int = 252, skip: int = 21) -> pd.Series:
    return np.log(close.shift(skip) / close.shift(lookback))


def donchian_position(bars: pd.DataFrame, window: int = 55) -> pd.Series:
    lo = bars["low"].rolling(window, min_periods=window).min()
    hi = bars["high"].rolling(window, min_periods=window).max()
    return ((bars["close"] - lo) / (hi - lo)).where(hi > lo, 0.5)


def corwin_schultz(bars: pd.DataFrame, window: int = 63, min_periods: int = 20) -> pd.Series:
    """Causal Corwin-Schultz (2012) spread proxy on bars (t-1, t), averaged over `window`."""
    h, l, ok = bars["high"], bars["low"], bars["bar_ok"]
    b = np.log(h / l) ** 2
    beta = b + b.shift(1)
    gamma = np.log(pd.concat([h, h.shift(1)], axis=1).max(axis=1)
                   / pd.concat([l, l.shift(1)], axis=1).min(axis=1)) ** 2
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / CS_K - np.sqrt(gamma / CS_K)
    spread = (2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))).clip(lower=0)
    spread = spread.where(ok & ok.shift(1, fill_value=False))
    return spread.rolling(window, min_periods=min_periods).mean()


# ------------------------------------------------------------- volume family

def volume_z(volume: pd.Series, window: int = 60, min_periods: int = 30) -> pd.Series:
    x = np.log1p(volume)
    med = x.rolling(window, min_periods=min_periods).median()
    mad = (x - med).abs().rolling(window, min_periods=min_periods).median()
    return ((x - med) / (1.4826 * mad.replace(0, np.nan))).clip(-4, 4)


def _slope(y: np.ndarray) -> float:
    ok = ~np.isnan(y)
    if ok.sum() < len(y) // 2:
        return np.nan
    t = np.arange(len(y))[ok]
    t = t - t.mean()
    return float((t * (y[ok] - y[ok].mean())).sum() / (t * t).sum())


def obv_slope(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    signed = np.sign(log_return(close)).fillna(0) * volume.fillna(0)
    obv = signed.cumsum().where(volume.notna())
    slope = obv.rolling(window, min_periods=window // 2).apply(_slope, raw=True)
    return slope / volume.rolling(window, min_periods=window // 2).mean()


def amihud(close: pd.Series, volume: pd.Series, window: int = 21, min_periods: int = 10) -> pd.Series:
    illiq = log_return(close).abs() / (close * volume)
    return np.log(illiq.rolling(window, min_periods=min_periods).mean() + 1e-20)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 40:
        return np.nan
    ra = pd.Series(a[ok]).rank().to_numpy()
    rb = pd.Series(b[ok]).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def volume_vol_coupling(close: pd.Series, volume: pd.Series, window: int = 60) -> pd.Series:
    a = log_return(close).abs().to_numpy()
    b = np.log1p(volume).to_numpy()
    out = np.full(len(a), np.nan)
    for i in range(window - 1, len(a)):
        out[i] = _spearman(a[i - window + 1: i + 1], b[i - window + 1: i + 1])
    return pd.Series(out, index=close.index)


# ------------------------------------------------------------- macro

def level_change(level: pd.Series, k: int) -> pd.Series:
    return level - level.shift(k)


# ------------------------------------------------------------------ technical indicators
# Optional extras (D-39): classic chart indicators with their textbook parameters,
# none tuned. They are not part of the frozen 95 features; models opt in.

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD(12, 26, 9), as fractions of the close so that it is comparable across price levels.

    line = EMA_fast − EMA_slow; signal = EMA_signal of the line; histogram = line − signal.
    The EMAs are recursive (adjust=False), so each value uses closes up to t only. The first
    ``slow + signal − 2`` rows have no value.
    """
    ema_f = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_s = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line / close, "macd_signal": sig / close, "macd_hist": (line - sig) / close})


def bollinger(close: pd.Series, window: int = 20, k: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands (20, 2σ): %B = (close − lower) / (upper − lower), 0 at the lower band and
    1 at the upper; bandwidth = (upper − lower) / middle, the classic "squeeze" gauge.
    Bands use the population standard deviation of the last ``window`` closes (Bollinger's
    definition). A flat window (zero width) gives no %B."""
    mid = close.rolling(window, min_periods=window).mean()
    sd = close.rolling(window, min_periods=window).std(ddof=0)
    width = 2 * k * sd
    pctb = ((close - (mid - k * sd)) / width).where(width > 0)
    return pd.DataFrame({"bb_pctb": pctb, "bb_width": width / mid})
