"""Source adapters. Each returns a tidy daily DataFrame indexed by date.

Storage policy: capture the FULLEST data each source offers, once, into the
Parquet cache. Market sources (yfinance, stooq) provide OHLCV: columns
open/high/low/close/volume (plus adj_close where offered). Macro sources (FRED)
publish a single daily value, stored as a one-column frame ("close") so every
consumer can rely on a close column existing.

Rationale: the dissertation's features are close-based, but the app's signal
layer (ATR fallback for TP/SL) needs high/low, and the snapshot is meant to be
pulled once and frozen; discarding columns at ingest would force a re-pull later.

Design rules:
  * Every fetch cached to Parquet on first call.
  * FRED via the official keyed API (FRED_API_KEY in .env); the keyless CSV
    endpoint proved unreliable (timeouts, Jul 2026) and is fallback only.
  * Failures return None rather than raising, so the availability report can
    show a complete picture.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

from xaglab.paths import CACHE_DIR

_UA = {"User-Agent": "Mozilla/5.0 (compatible; xaglab research pipeline)"}
_TIMEOUT = 30

OHLCV_COLS = ("open", "high", "low", "close", "adj_close", "volume")


def cache_filename(kind: str, symbol: str) -> str:
    """Stable file name for one series; shared by the live cache and every snapshot."""
    safe = symbol.replace("/", "_").replace("^", "idx_").replace("=", "_")
    return f"{kind}__{safe}.parquet"


def _cache_path(kind: str, symbol: str) -> Path:
    return CACHE_DIR / cache_filename(kind, symbol)


def _read_cache(kind: str, symbol: str) -> pd.DataFrame | None:
    p = _cache_path(kind, symbol)
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        # Legacy close-only caches stored a single "value" column.
        if list(df.columns) == ["value"]:
            df = df.rename(columns={"value": "close"})
        return df
    except Exception:
        return None


def _write_cache(kind: str, symbol: str, df: pd.DataFrame) -> None:
    p = _cache_path(kind, symbol)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(p)
    except Exception:
        pass  # caching is best-effort, never fatal


def _tidy_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    df = df[~df.index.isna()]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = "date"
    return df


def _normalise_frame(df: pd.DataFrame) -> pd.DataFrame | None:
    """Lowercase to the OHLCV schema, coerce numerics, require a close column."""
    rename = {}
    for c in df.columns:
        key = str(c).strip().lower().replace(" ", "_")
        if key in ("adj_close", "adjclose"):
            key = "adj_close"
        if key in OHLCV_COLS:
            rename[c] = key
    df = df.rename(columns=rename)
    keep = [c for c in OHLCV_COLS if c in df.columns]
    if "close" not in keep:
        return None
    df = df[keep].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["close"])
    return _tidy_index(df)


def fetch_yfinance(symbol: str, start: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf

        raw = yf.download(
            symbol, start=start, auto_adjust=False, progress=False, threads=False
        )
        if raw is None or len(raw) == 0:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw.xs(symbol, axis=1, level=1) if symbol in raw.columns.get_level_values(1) \
                else raw.droplevel(1, axis=1)
        return _normalise_frame(raw)
    except Exception:
        return None


def fetch_stooq(symbol: str, start: str) -> pd.DataFrame | None:
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    try:
        r = requests.get(url, headers=_UA, timeout=_TIMEOUT)
        if r.status_code != 200 or not r.text.strip().lower().startswith("date"):
            return None
        df = pd.read_csv(io.StringIO(r.text)).set_index("Date")
        df = _normalise_frame(df)
        return None if df is None else df[df.index >= pd.Timestamp(start)]
    except Exception:
        return None


def _load_env_key(name: str) -> str | None:
    """Read a key from the environment, falling back to the repo's .env file."""
    import os

    if os.environ.get(name):
        return os.environ[name]
    from xaglab.paths import ENV_FILE as env_file
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith(f"{name}=") and len(line) > len(name) + 1:
                return line.split("=", 1)[1].strip()
    return None


def fetch_fred_api(series_id: str, start: str) -> pd.DataFrame | None:
    """Official keyed FRED API. Single daily value, stored as 'close'."""
    key = _load_env_key("FRED_API_KEY")
    if not key:
        return None
    try:
        from fredapi import Fred

        s = Fred(api_key=key).get_series(series_id)
        if s is None or len(s) == 0:
            return None
        df = _tidy_index(s.to_frame(name="close"))
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna()
        return df[df.index >= pd.Timestamp(start)]
    except Exception:
        return None


def fetch_fred_csv(series_id: str, start: str) -> pd.DataFrame | None:
    """Keyless FRED CSV endpoint (unreliable; fallback only)."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        r = requests.get(url, headers=_UA, timeout=_TIMEOUT)
        if r.status_code != 200:
            return None
        raw = pd.read_csv(io.StringIO(r.text))
        date_col = raw.columns[0]
        val_col = [c for c in raw.columns if c != date_col][0]
        df = raw.set_index(date_col)[[val_col]].rename(columns={val_col: "close"})
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = _tidy_index(df.dropna())
        return df[df.index >= pd.Timestamp(start)]
    except Exception:
        return None


def fetch(kind: str, symbol: str, start: str, use_cache: bool = True) -> pd.DataFrame | None:
    """Dispatch to the right adapter, with Parquet caching. Returns OHLCV frame."""
    if use_cache:
        cached = _read_cache(kind, symbol)
        if cached is not None and len(cached) > 0:
            return cached

    if kind == "yfinance":
        df = fetch_yfinance(symbol, start)
    elif kind == "stooq":
        df = fetch_stooq(symbol, start)
    elif kind == "fred_api":
        df = fetch_fred_api(symbol, start)
    elif kind == "fred_csv":
        df = fetch_fred_csv(symbol, start)
    elif kind == "alphavantage":
        return None  # sentiment node is built in Phase A-late
    else:
        raise ValueError(f"unknown source kind: {kind!r}")

    if df is not None and len(df) > 0:
        _write_cache(kind, symbol, df)
    return df


def fetch_close(kind: str, symbol: str, start: str, use_cache: bool = True) -> pd.Series | None:
    """Convenience: the close series alone, named after the symbol."""
    df = fetch(kind, symbol, start, use_cache=use_cache)
    if df is None or len(df) == 0:
        return None
    s = df["close"].copy()
    s.name = symbol
    return s


#: Public name for other adapters (event calendars) that need the same key lookup.
load_env_key = _load_env_key
