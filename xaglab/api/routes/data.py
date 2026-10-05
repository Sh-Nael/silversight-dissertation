"""Data: nodes, aligned prices, features, events, regime windows, data quality."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from xaglab.api.deps import columns, feature_set, panel, records
from xaglab.data.nodes import NODES_BY_KEY
from xaglab.data.regimes import REGIMES

router = APIRouter(prefix="/data", tags=["data"])

Freq = Literal["D", "W-FRI", "ME"]


class NodeInfo(BaseModel):
    key: str
    label: str
    role: str
    cluster: str
    source: str
    has_volume: bool
    features: list[str]


def _node(key: str) -> str:
    if key not in panel().nodes:
        raise HTTPException(404, f"no node {key!r}; available: {sorted(panel().nodes)}")
    return key


def _resample(df, freq: Freq):
    return df if freq == "D" else df.resample(freq).last()


@router.get("/nodes", response_model=list[NodeInfo])
def nodes() -> list[NodeInfo]:
    """The markets in the panel, with their source and feature names."""
    p, fs = panel(), feature_set()
    out = []
    for key in p.node_keys:
        n = NODES_BY_KEY[key]
        out.append(NodeInfo(key=key, label=n.label, role=n.role, cluster=n.cluster,
                            source=f"{n.primary.kind}:{n.primary.symbol}",
                            has_volume=bool(p.quality.loc[key, "has_volume"]),
                            features=list(fs.nodes[key].columns)))
    return out


@router.get("/panel")
def panel_series(node: str = Query(...), freq: Freq = Query("D")) -> dict[str, list[Any]]:
    """Aligned OHLCV plus the bar_ok / filled flags for one node."""
    df = panel().nodes[_node(node)].copy()
    df[["bar_ok", "filled"]] = df[["bar_ok", "filled"]].astype(bool)
    return columns(_resample(df, freq))


@router.get("/features")
def features(node: str = Query(...), names: str | None = Query(None, description="comma-separated; default all"),
             freq: Freq = Query("D")) -> dict[str, list[Any]]:
    """Feature values for one node (``node=events`` gives the event features)."""
    fs = feature_set()
    if node == "events":
        df = fs.events
    else:
        df = fs.nodes[_node(node)]
    if names:
        wanted = [n.strip() for n in names.split(",") if n.strip()]
        missing = [n for n in wanted if n not in df.columns]
        if missing:
            raise HTTPException(404, f"unknown features {missing}")
        df = df[wanted]
    return columns(_resample(df, freq))


@router.get("/events")
def events() -> list[dict[str, Any]]:
    """The frozen macro event calendar (CPI, NFP, FOMC) with the scheduled flag."""
    from xaglab.data.store import Snapshot

    t = Snapshot.open("calendar_v1").read("events.csv")
    t["scheduled"] = t["scheduled"].astype(str).str.lower().isin(("true", "1"))
    return records(t)


@router.get("/regimes")
def regimes() -> list[dict[str, str]]:
    """Documented regime windows used for chart shading and the regime analysis."""
    return [{"label": a, "start": b, "end": c} for a, b, c in REGIMES]


@router.get("/quality")
def quality() -> list[dict[str, Any]]:
    """The data-quality audit per node (repaired bars, flat bars, fills, missing volume)."""
    q = panel().quality.reset_index().rename(columns={"index": "node"})
    return records(q)
