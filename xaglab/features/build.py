"""Feature and label construction from an aligned :class:`~xaglab.data.panel.Panel`.

Output is a :class:`FeatureSet`: one feature frame per node (the graph model's node
inputs), the event frame, and the labels. Flat models use :meth:`FeatureSet.pooled`.
Everything here is causal. Scaling is fitted later, inside each training window
(see :mod:`xaglab.features.scaling`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from xaglab.data.nodes import TARGET
from xaglab.data.panel import Panel, is_macro
from xaglab.features import price as P

HORIZONS = (1, 5)


@dataclass(frozen=True)
class FeatureSet:
    index: pd.DatetimeIndex
    nodes: dict[str, pd.DataFrame]
    events: pd.DataFrame | None
    labels: pd.DataFrame
    #: optional per-node indicator blocks (MACD, Bollinger; D-39). Not part of the frozen 95
    #: features: they enter a model only when it asks for them via ``pooled(extras=...)``.
    extras: dict[str, pd.DataFrame] | None = None

    def pooled(self, include_events: bool = True, extras: str | None = None) -> pd.DataFrame:
        """Flat feature matrix: columns '<node>__<feature>' (+ 'event__<name>').

        ``extras``: None (the frozen 95 features), "silver" (add the target's indicator block)
        or "all" (add every market node's block)."""
        parts = [f.add_prefix(f"{k}__") for k, f in self.nodes.items()]
        if include_events and self.events is not None:
            parts.append(self.events.add_prefix("event__"))
        if extras is not None:
            if extras not in ("silver", "all"):
                raise ValueError(f"extras must be None, 'silver' or 'all', not {extras!r}")
            keep = [TARGET] if extras == "silver" else list(self.extras or {})
            parts += [self.extras[k].add_prefix(f"{k}__") for k in keep]
        return pd.concat(parts, axis=1)

    @property
    def feature_names(self) -> dict[str, list[str]]:
        return {k: list(v.columns) for k, v in self.nodes.items()}

    def warmup_end(self) -> pd.Timestamp:
        """First date on which every node has all its features (after rolling warm-up)."""
        ok = np.ones(len(self.index), dtype=bool)
        for f in self.nodes.values():
            ok &= f.notna().to_numpy().all(axis=1) | _allowed_nan_rows(f)
        return self.index[np.argmax(ok)]


def _allowed_nan_rows(f: pd.DataFrame) -> np.ndarray:
    # Volume features may be legitimately missing on individual days (masked volume).
    vol_cols = [c for c in f.columns if c in VOLUME_FEATURES]
    core = f.drop(columns=vol_cols)
    return core.notna().to_numpy().all(axis=1)


VOLUME_FEATURES = ("vol_z60", "obv_slope20", "amihud21", "corr_rv60")


def market_features(bars: pd.DataFrame, has_volume: bool, level_feature: bool = False) -> pd.DataFrame:
    c = bars["close"]
    f = {
        "r1": P.log_return(c, 1),
        "r5": P.log_return(c, 5),
        "r21": P.log_return(c, 21),
        "rv21": np.log(P.realised_vol(c) + 1e-8),
        "gk21": P.garman_klass_vol(bars),
        "atr14": P.atr_norm(bars),
        "clv": P.close_location(bars),
        "body": P.body_fraction(bars),
        "gap": P.overnight_gap(bars),
        "ma21": P.ma_distance(c, 21),
        "ma200": P.ma_distance(c, 200),
        "tsmom252": P.ts_momentum(c),
        "donch55": P.donchian_position(bars),
        "cs63": P.corwin_schultz(bars),
    }
    if level_feature:  # a level that is itself meaningful (VIX)
        f["log_level"] = np.log(c)
    if has_volume:
        v = bars["volume"]
        f.update({
            "vol_z60": P.volume_z(v),
            "obv_slope20": P.obv_slope(c, v),
            "amihud21": P.amihud(c, v),
            "corr_rv60": P.volume_vol_coupling(c, v),
        })
    return pd.DataFrame(f, index=bars.index)


def macro_features(bars: pd.DataFrame) -> pd.DataFrame:
    lvl = bars["close"]
    d1 = P.level_change(lvl, 1)
    return pd.DataFrame({
        "level": lvl,
        "d1": d1,
        "d5": P.level_change(lvl, 5),
        "d21": P.level_change(lvl, 21),
        "rv21": d1.rolling(21, min_periods=15).std(),
    }, index=bars.index)


def gold_silver_ect(gold: pd.Series, silver: pd.Series, window: int = 252) -> pd.Series:
    """Error-correction term: z-score of ln(gold/silver) against its trailing equilibrium."""
    lr = np.log(gold / silver)
    mu = lr.rolling(window, min_periods=window).mean()
    sd = lr.rolling(window, min_periods=window).std()
    return (lr - mu) / sd


def make_labels(close: pd.Series) -> pd.DataFrame:
    """Targets for a forecast made at the close of t. 'Up' means strictly positive."""
    r = P.log_return(close, 1)
    out = {}
    for h in HORIZONS:
        fwd = np.log(close.shift(-h) / close)
        out[f"ret{h}"] = fwd
        out[f"up{h}"] = (fwd > 0).astype(float).where(fwd.notna())
        rv = sum(r.shift(-i) ** 2 for i in range(1, h + 1))
        out[f"rv{h}"] = rv
    return pd.DataFrame(out, index=close.index)


def build_features(panel: Panel) -> FeatureSet:
    nodes = {}
    for key in panel.node_keys:
        bars = panel.nodes[key]
        if is_macro(key):
            nodes[key] = macro_features(bars)
        else:
            has_vol = bool(panel.quality.loc[key, "has_volume"])
            nodes[key] = market_features(bars, has_vol, level_feature=(key == "vix"))
    if "gold" in nodes:
        nodes[TARGET]["gsr_ect"] = gold_silver_ect(panel.close("gold"), panel.close(TARGET))
    labels = make_labels(panel.close(TARGET))
    extras = {k: technical_extras(panel.close(k)) for k in panel.node_keys if not is_macro(k)}
    return FeatureSet(panel.index, nodes, panel.events, labels, extras)


def technical_extras(close: pd.Series) -> pd.DataFrame:
    """The optional indicator block of one market: MACD(12, 26, 9) and Bollinger(20, 2σ)."""
    out = pd.concat([P.macd(close), P.bollinger(close)], axis=1)
    out.index = close.index
    return out
