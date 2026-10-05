"""Align every node onto the target's trading calendar under the pre-registered rules.

Rules (Chapter 3, Stage P2; D-18):
  1. The calendar is the target's (silver's) trading days.
  2. Market nodes keep same-day bars. A missing day is forward-filled from the node's
     last close for at most ``max_ffill`` trading days, as a flat bar with no volume and
     ``bar_ok = False``; beyond that the value is missing, never invented.
  3. Macro (FRED) values dated d become usable only from the first trading day
     strictly after d (``macro_lag = 1``): FRED publishes daily yields the next business
     day. The same staleness limit applies.
  4. Bar repair: high/low are widened to contain open and close (Yahoo futures bars
     before 2012 often close outside their own range). Bars with high == low carry no
     range information and are flagged ``bar_ok = False``. Zero volume is missing
     volume. Nodes that never report volume (DXY, VIX) get an all-missing volume column.

The output is the single input to feature construction, identical for research and app.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from xaglab.data.events import EventCalendar
from xaglab.data.nodes import CORE_KEYS, NODES_BY_KEY, TARGET
from xaglab.data.store import Snapshot

PRICE_COLS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class Panel:
    index: pd.DatetimeIndex
    nodes: dict[str, pd.DataFrame]  # per node: open high low close volume bar_ok filled
    events: pd.DataFrame | None
    quality: pd.DataFrame = field(repr=False)  # per-node data-quality counts (Ch3 table)

    @property
    def node_keys(self) -> list[str]:
        return list(self.nodes)

    def close(self, key: str) -> pd.Series:
        return self.nodes[key]["close"]

    def truncate(self, end: pd.Timestamp) -> Panel:
        """Panel as it would have looked with data only up to and including `end`."""
        idx = self.index[self.index <= end]
        return Panel(idx, {k: v.loc[idx] for k, v in self.nodes.items()},
                     None if self.events is None else self.events.loc[idx], self.quality)


def is_macro(key: str) -> bool:
    return NODES_BY_KEY[key].primary.kind.startswith("fred")


def _align_market(raw: pd.DataFrame, index: pd.DatetimeIndex, max_ffill: int) -> tuple[pd.DataFrame, dict]:
    raw = raw.copy()
    has_volume = "volume" in raw and (raw["volume"].fillna(0) > 0).mean() > 0.5
    if "open" not in raw:  # close-only source: treat as flat bars
        for c in ("open", "high", "low"):
            raw[c] = raw["close"]
    hi = raw[["high", "open", "close"]].max(axis=1)
    lo = raw[["low", "open", "close"]].min(axis=1)
    repaired = int(((hi != raw["high"]) | (lo != raw["low"])).sum())
    raw["high"], raw["low"] = hi, lo
    raw["volume"] = raw["volume"].where(raw["volume"] > 0) if has_volume else np.nan

    df = raw.reindex(index)
    present = df["close"].notna()
    close = df["close"].ffill(limit=max_ffill)
    filled = ~present & close.notna()
    for c in ("open", "high", "low"):
        df[c] = df[c].where(present, close)
    df["close"] = close
    df["volume"] = df["volume"].where(present)
    df["filled"] = filled
    df["bar_ok"] = present & (df["high"] > df["low"])
    q = {"rows_raw": len(raw), "repaired_range": repaired,
         "flat_bars": int((present & (df["high"] <= df["low"])).sum()),
         "filled": int(filled.sum()), "missing_after_fill": int(close.isna().sum()),
         "volume_missing": int(df["volume"].isna().sum()), "has_volume": bool(has_volume)}
    return df[[*PRICE_COLS, "bar_ok", "filled"]], q


def _align_macro(raw: pd.DataFrame, index: pd.DatetimeIndex, max_ffill: int, lag: int) -> tuple[pd.DataFrame, dict]:
    s = raw["close"].dropna()
    # Trading-day position at which each observation becomes usable.
    pos = index.searchsorted(s.index, side="right") + (lag - 1)
    keep = pos < len(index)
    by_pos = pd.Series(s.to_numpy()[keep], index=pos[keep]).groupby(level=0).last()
    val = by_pos.reindex(np.arange(len(index)))
    present = val.notna().to_numpy()
    val = val.ffill(limit=max_ffill)
    df = pd.DataFrame(index=index)
    for c in ("open", "high", "low", "close"):
        df[c] = val.to_numpy()
    df["volume"] = np.nan
    df["filled"] = ~present & val.notna().to_numpy()
    df["bar_ok"] = False  # a single daily value has no range
    q = {"rows_raw": len(s), "repaired_range": 0, "flat_bars": 0,
         "filled": int(df["filled"].sum()), "missing_after_fill": int(val.isna().sum()),
         "volume_missing": len(index), "has_volume": False}
    return df, q


def build_panel_from_frames(
    raw: dict[str, pd.DataFrame],
    event_table: pd.DataFrame | None = None,
    max_ffill: int = 5,
    macro_lag: int = 1,
    event_cap: int = 5,
) -> Panel:
    """Core alignment on raw per-node frames (keys = node keys). Pure: no I/O."""
    if TARGET not in raw:
        raise ValueError("the target node must be part of the panel")
    target = raw[TARGET]
    index = pd.DatetimeIndex(target.index[target["close"].notna()], name="date")
    nodes, quality = {}, {}
    for key, frame in raw.items():
        if is_macro(key):
            nodes[key], quality[key] = _align_macro(frame, index, max_ffill, macro_lag)
        else:
            nodes[key], quality[key] = _align_market(frame, index, max_ffill)
    events = None
    if event_table is not None:
        events = EventCalendar.from_table(event_table).features(index, cap=event_cap)
    return Panel(index, nodes, events, pd.DataFrame(quality).T)


def load_raw(prices: Snapshot, node_keys: tuple[str, ...] = CORE_KEYS) -> dict[str, pd.DataFrame]:
    raw = {}
    for key in node_keys:
        frame = prices.read_node(NODES_BY_KEY[key])
        if frame is None:
            raise KeyError(f"node {key!r} is not in snapshot {prices.name!r}")
        raw[key] = frame
    return raw


def build_panel(
    prices: Snapshot,
    calendar: Snapshot | None = None,
    node_keys: tuple[str, ...] = CORE_KEYS,
    **kwargs,
) -> Panel:
    events = calendar.read("events.csv") if calendar is not None else None
    return build_panel_from_frames(load_raw(prices, node_keys), events, **kwargs)


def load_default_panel(verify: bool = True) -> Panel:
    """The dissertation panel: snapshot dissertation_v1 plus calendar_v1."""
    return build_panel(Snapshot.open("dissertation_v1", verify=verify),
                       Snapshot.open("calendar_v1", verify=verify))
