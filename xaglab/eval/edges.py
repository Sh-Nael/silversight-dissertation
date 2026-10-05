"""Edge dynamics: which driver edges AIM-DG kept, when, and whether that follows market
regimes (Phase 6.3, SQ3; the rules were fixed in D-40 before any by-date result).

From a run whose diagnostics carry a monthly ``selection`` (AIM-DG and its replays):

  timeline()          one row per monthly refit: each edge on/off, its reliability and
                      contribution, the number of active edges, the clock
  cluster_activity()  per month and driver cluster (monetary: gold, dollar, real yield;
                      industrial: copper; risk: VIX): the share of the cluster's candidate
                      edges that is active, and the cluster's mean reliability
  regime_tests()      stress months (2011, 2013, 2020, 2022) against the rest, per cluster and
                      measure, with a one-sided circular-shift permutation test
  persistence()       per edge: share of months active, mean run length, switches; and the
                      month-to-month Jaccard similarity of the active sets

The circular shift moves the regime labels along the 200 months as a block, so the
persistence of both the selections and the regimes is kept under the null; with n months
there are n − 1 shifts, all of which are used (an exact test, resolution 1 / n).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from xaglab.data.nodes import NODES
from xaglab.data.regimes import REGIMES

CLUSTER_OF = {n.key: n.cluster for n in NODES}
#: Pre-stated direction of each cluster's hypothesis in stress months (D-40).
HYPOTHESIS = {"monetary": "greater", "risk": "greater", "industrial": "less"}
TEST_START = "2010-01-01"


def _driver(edge: str) -> str:
    return edge.rsplit("@", 1)[0]


def timeline(run: Any) -> pd.DataFrame:
    """One row per refit (indexed by its first forecast date). Columns: ``on:<edge>`` (bool),
    ``r:<edge>`` and ``c:<edge>`` where recorded (reliability, contribution in mnats),
    ``n_edges``, ``rate``, ``tau``, ``shift``."""
    first = run.predictions.reset_index().groupby(["fold", "refit"])["date"].min()
    rows = []
    edges: list[str] | None = None
    for d in run.diagnostics:
        sel = d.get("selection")
        if sel is None:
            raise ValueError(f"{run.run_id} has no edge selections in its diagnostics")
        fold, refit = d.get("fold", sel.get("fold")), d.get("refit", sel.get("refit"))
        if edges is None and "reliability" in sel:
            edges = list(sel["reliability"])
        clock = sel.get("clock", sel)
        row: dict[str, Any] = {"date": first.loc[(fold, refit)], "fold": fold, "refit": refit,
                               "active": list(sel["active"]), "rate": clock.get("rate"), "tau": clock.get("tau"),
                               "shift": clock.get("shift")}
        for e, v in sel.get("reliability", {}).items():
            row[f"r:{e}"] = v
        for e, v in sel.get("contribution_mnats", {}).items():
            row[f"c:{e}"] = v
        rows.append(row)
    if edges is None:  # replays record the active sets only: the 15 candidate edges in the usual order
        drivers = [n.key for n in NODES if n.role == "driver"]
        edges = [f"{d}@{lag}" for d in ("gold", "copper", "dxy", "real_yield_10y", "vix") if d in drivers
                 for lag in (1, 5, 21)]
    out = pd.DataFrame(rows).set_index("date").sort_index()
    for e in edges:
        out[f"on:{e}"] = out["active"].map(lambda a, e=e: e in a)
    out["n_edges"] = out["active"].map(len)
    out.attrs["edges"] = edges
    return out.drop(columns="active")


def edges_of(tl: pd.DataFrame) -> list[str]:
    return [c[3:] for c in tl.columns if c.startswith("on:")]


def cluster_activity(tl: pd.DataFrame) -> pd.DataFrame:
    """Per month: ``act:<cluster>`` = active share of the cluster's candidate edges,
    ``rel:<cluster>`` = mean reliability of its edges (where reliability was recorded)."""
    out = pd.DataFrame(index=tl.index)
    edges = edges_of(tl)
    for cluster in sorted({CLUSTER_OF[_driver(e)] for e in edges}):
        mine = [e for e in edges if CLUSTER_OF[_driver(e)] == cluster]
        out[f"act:{cluster}"] = tl[[f"on:{e}" for e in mine]].mean(axis=1)
        rel = [f"r:{e}" for e in mine if f"r:{e}" in tl.columns]
        if rel:
            out[f"rel:{cluster}"] = tl[rel].mean(axis=1)
    return out


def regime_flags(dates: pd.DatetimeIndex, regimes: tuple[tuple[str, str, str], ...] = REGIMES,
                 start: str = TEST_START) -> pd.DataFrame:
    """Per date: membership of each regime that overlaps the test period, and ``stress`` (any)."""
    out = pd.DataFrame(index=dates)
    for label, a, b in regimes:
        if pd.Timestamp(b) < pd.Timestamp(start):
            continue  # before the test period (2008): no refit falls in it
        out[label] = (dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b))
    out["stress"] = out.any(axis=1)
    return out


def circular_shift_test(x: np.ndarray, inside: np.ndarray, alternative: str = "greater") -> dict[str, float]:
    """mean(x inside) − mean(x outside), with the p-value from shifting the labels along the
    series by every possible offset (the observed placement counts as one of them)."""
    x, inside = np.asarray(x, float), np.asarray(inside, bool)
    ok = ~np.isnan(x)
    x, inside = x[ok], inside[ok]
    n = len(x)
    if inside.sum() == 0 or inside.sum() == n:
        return {"inside": np.nan, "outside": np.nan, "diff": np.nan, "p": np.nan, "n_inside": int(inside.sum())}

    def stat(g: np.ndarray) -> float:
        return float(x[g].mean() - x[~g].mean())

    obs = stat(inside)
    null = np.array([stat(np.roll(inside, s)) for s in range(1, n)])
    more = null >= obs - 1e-12 if alternative == "greater" else null <= obs + 1e-12
    return {"inside": float(x[inside].mean()), "outside": float(x[~inside].mean()), "diff": obs,
            "p": float((1 + more.sum()) / n), "n_inside": int(inside.sum())}


def regime_tests(tl: pd.DataFrame, regimes: tuple[tuple[str, str, str], ...] = REGIMES) -> pd.DataFrame:
    """Every cluster and measure against every regime and against all stress months.

    The six ``stress`` rows (3 clusters × activity, reliability) are the pre-stated tests; the
    per-regime rows are descriptive."""
    act = cluster_activity(tl)
    flags = regime_flags(tl.index, regimes)
    rows = []
    for col in act.columns:
        measure, cluster = col.split(":")
        for regime in flags.columns:
            res = circular_shift_test(act[col].to_numpy(), flags[regime].to_numpy(), HYPOTHESIS[cluster])
            rows.append({"cluster": cluster, "measure": "activity" if measure == "act" else "reliability",
                         "regime": regime, "hypothesis": HYPOTHESIS[cluster], **res,
                         "primary": regime == "stress"})
    return pd.DataFrame(rows)


def persistence(tl: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Per edge: share of months active, number of switches, mean length of an active spell.
    Overall: mean Jaccard similarity of consecutive months' active sets, mean edges changed."""
    rows = []
    on = tl[[f"on:{e}" for e in edges_of(tl)]].to_numpy(bool)
    for k, e in enumerate(edges_of(tl)):
        a = on[:, k]
        switches = int(np.sum(a[1:] != a[:-1]))
        starts = int(a[0]) + int(np.sum(a[1:] & ~a[:-1]))
        rows.append({"edge": e, "cluster": CLUSTER_OF[_driver(e)], "share_active": float(a.mean()),
                     "switches": switches, "mean_spell_months": float(a.sum() / starts) if starts else 0.0})
    inter = (on[1:] & on[:-1]).sum(1)
    union = (on[1:] | on[:-1]).sum(1)
    jac = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    overall = {"mean_jaccard": float(jac.mean()), "mean_edges_changed": float((on[1:] != on[:-1]).sum(1).mean()),
               "months_unchanged": int(np.sum((on[1:] == on[:-1]).all(1))), "months": len(on),
               "mean_edges": float(on.sum(1).mean())}
    return pd.DataFrame(rows), overall
