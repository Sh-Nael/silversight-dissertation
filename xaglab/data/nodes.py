"""The node universe: silver and its candidate inter-market drivers.

This is the single source of truth for both the dissertation experiments and the
app's inference path. Adding or removing a driver happens here and nowhere else.

Cluster labels matter downstream: the signal layer's regime-agreement filter asks
whether the *monetary* cluster agrees in sign, and the attribution analysis reports
per-cluster contributions.

Every source is public and aggregate. No social media, no data about identifiable
people. See README scope boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    """Where one series comes from.

    kind: "yfinance" | "fred_api" | "fred_csv" | "alphavantage"
    (Stooq was dropped as a fallback on 7 Aug 2026: its CSV endpoint now serves a
    JavaScript proof-of-work wall. See D-10.)
    symbol: provider-specific identifier
    """

    kind: str
    symbol: str


@dataclass(frozen=True)
class Node:
    key: str
    label: str
    role: str  # "target" | "driver" | "optional"
    cluster: str  # "target" | "monetary" | "industrial" | "risk" | "sentiment"
    primary: Source
    fallback: Source | None = None
    rationale: str = ""


# Roughly 20 years back, spanning the 2008, 2011, 2020 and 2022 regime breaks that
# an adaptive model exists to handle.
HISTORY_START = "2006-01-01"


NODES: tuple[Node, ...] = (
    Node(
        key="silver",
        label="Silver (XAG/USD)",
        role="target",
        cluster="target",
        primary=Source("yfinance", "SI=F"),
        rationale="Forecast target. Futures front-month as the liquid daily proxy for spot.",
    ),
    Node(
        key="gold",
        label="Gold (XAU/USD)",
        role="driver",
        cluster="monetary",
        primary=Source("yfinance", "GC=F"),
        rationale="Dominant monetary co-mover; the gold-silver ratio is the classic regime tell.",
    ),
    Node(
        key="dxy",
        label="US Dollar Index",
        role="driver",
        cluster="monetary",
        primary=Source("yfinance", "DX-Y.NYB"),
        rationale="Silver is USD-denominated; dollar strength is a first-order inverse driver.",
    ),
    Node(
        key="real_yield_10y",
        label="US 10y real yield (TIPS)",
        role="driver",
        cluster="monetary",
        primary=Source("fred_api", "DFII10"),
        fallback=Source("fred_csv", "DFII10"),
        rationale="Opportunity cost of holding a non-yielding metal. Published with a lag.",
    ),
    Node(
        key="nominal_10y",
        label="US 10y nominal yield",
        # NOT in the committed M2 node set (gold, DXY, real yields, VIX, copper,
        # sentiment). Auxiliary candidate only: must not enter any reported
        # experiment without supervisor sign-off as an enrichment.
        role="optional",
        cluster="monetary",
        primary=Source("fred_api", "DGS10"),
        fallback=Source("fred_csv", "DGS10"),
        rationale="With real yield gives the breakeven inflation channel. Aux only, see above.",
    ),
    Node(
        key="copper",
        label="Copper",
        role="driver",
        cluster="industrial",
        primary=Source("yfinance", "HG=F"),
        rationale="Industrial-demand proxy. This is the half of silver that gold does not explain.",
    ),
    Node(
        key="vix",
        label="VIX",
        role="driver",
        cluster="risk",
        primary=Source("yfinance", "^VIX"),
        rationale="Risk-on/off state; expected to gate when the monetary cluster dominates.",
    ),
    Node(
        key="sentiment",
        label="Published-news sentiment",
        role="optional",
        cluster="sentiment",
        # Built in Phase A-late from Alpha Vantage NEWS_SENTIMENT or RSS + FinBERT.
        # Optional by design: it is one of the ablations, so an outage degrades
        # inference gracefully instead of breaking it.
        primary=Source("alphavantage", "NEWS_SENTIMENT"),
        rationale="PUBLISHED NEWS ONLY. Never social media or influencer content.",
    ),
)


NODES_BY_KEY: dict[str, Node] = {n.key: n for n in NODES}

TARGET = "silver"

#: Nodes required for a runnable Phase-A pipeline (sentiment is optional by design).
CORE_KEYS: tuple[str, ...] = tuple(n.key for n in NODES if n.role in ("target", "driver"))


def clusters() -> dict[str, list[str]]:
    """Map cluster name -> node keys, used by the regime-agreement filter."""
    out: dict[str, list[str]] = {}
    for n in NODES:
        out.setdefault(n.cluster, []).append(n.key)
    return out
