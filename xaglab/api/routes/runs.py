"""Runs: the list, one run's record, its forecasts, calibration and volatility path."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from xaglab.api.deps import clean, columns, get_run, records, run_ids, runs_dir
from xaglab.eval.metrics import summarise
from xaglab.eval.report import calibration_bins, selective, volatility_path
from xaglab.eval.signals import Rules, equity_curve, evaluate, signals

router = APIRouter(tags=["runs"])


class RunSummary(BaseModel):
    id: str
    model: str
    tag: str | None
    started_utc: str
    wall_seconds: float
    fit_seconds_total: float
    n_refits: int
    tuning_points: int | None
    commit: str
    dirty: bool
    headline: dict[str, float | None]  # brier1, brier5, logloss1, accuracy1, qlike1, qlike5


class RunDetail(BaseModel):
    id: str
    manifest: dict[str, Any]
    timings: list[dict[str, Any]]
    tuning: list[dict[str, Any]] | None
    diagnostics: list[dict[str, Any]]


def _summary(run_id: str, base: Path) -> RunSummary:
    r = get_run(base, run_id)
    m = r.manifest
    s = summarise(r.predictions).set_index("h")
    get = lambda h, k: clean(s.loc[h, k]) if k in s.columns else None
    model = m["model"]
    prefix = f"{model}-"
    rest = run_id.removeprefix(prefix)
    tag = rest[: -len("-YYYYMMDD-HHMMSS")] if len(rest) > len("YYYYMMDD-HHMMSS") else None
    return RunSummary(
        id=run_id, model=model, tag=tag or None, started_utc=m["started_utc"],
        wall_seconds=m["wall_seconds"], fit_seconds_total=m.get("fit_seconds_total", 0.0),
        n_refits=m["protocol"]["n_refits"],
        tuning_points=(m.get("tuning") or {}).get("n_points"),
        commit=m["code"]["commit"][:7], dirty=m["code"]["dirty"],
        headline={"brier1": get(1, "brier"), "brier5": get(5, "brier"), "logloss1": get(1, "logloss"),
                  "accuracy1": get(1, "accuracy"), "qlike1": get(1, "vol_qlike"), "qlike5": get(5, "vol_qlike")},
    )


@router.get("/runs", response_model=list[RunSummary])
def list_runs(base: Path = Depends(runs_dir)) -> list[RunSummary]:
    """Every run on disk, newest first, with headline scores."""
    out = [_summary(i, base) for i in run_ids(base)]
    return sorted(out, key=lambda r: r.started_utc, reverse=True)


@router.get("/runs/{run_id}", response_model=RunDetail)
def run_detail(run_id: str, base: Path = Depends(runs_dir)) -> RunDetail:
    """The full record: manifest, per-refit timings, tuning log, diagnostics."""
    r = get_run(base, run_id)
    return RunDetail(id=run_id, manifest=clean(r.manifest), timings=records(r.timings),
                     tuning=clean(r.tuning), diagnostics=clean(r.diagnostics))


@router.get("/runs/{run_id}/predictions")
def run_predictions(run_id: str, base: Path = Depends(runs_dir)) -> dict[str, list[Any]]:
    """Forecasts and true labels per test day, as columns (date, fold, refit, p_up1 …)."""
    return columns(get_run(base, run_id).predictions)


@router.get("/runs/{run_id}/calibration")
def run_calibration(run_id: str, h: int = Query(1, ge=1, le=5), bins: int = Query(10, ge=2, le=50),
                    base: Path = Depends(runs_dir)) -> list[dict[str, Any]]:
    """Reliability-diagram points: mean forecast, observed up-rate and days per bin."""
    return records(calibration_bins(get_run(base, run_id), h, bins))


@router.get("/runs/{run_id}/volatility")
def run_volatility(run_id: str, h: int = Query(5, ge=1, le=5),
                   freq: Literal["D", "W-FRI", "ME"] = Query("W-FRI"),
                   base: Path = Depends(runs_dir)) -> dict[str, list[Any]]:
    """Annualised forecast vs realised volatility, averaged per period (default weekly)."""
    return columns(volatility_path(get_run(base, run_id), h, freq))


@router.get("/runs/{run_id}/selective")
def run_selective(run_id: str, h: int = Query(1, ge=1, le=5),
                  mode: Literal["causal", "ranked"] = Query("causal"),
                  base: Path = Depends(runs_dir)) -> list[dict[str, Any]]:
    """Accuracy against coverage: how often the model is right on its most confident days
    (causal: confident = in the top share of its own last 252 forecasts)."""
    return records(selective(get_run(base, run_id), h, mode=mode))


def _bars():
    from xaglab.api.deps import panel

    return panel().nodes["silver"][["open", "high", "low", "close"]]


@router.get("/runs/{run_id}/signals")
def run_signals(run_id: str, h: int = Query(1, ge=1, le=5), cost_bps: float = Query(10.0, ge=0, le=100),
                base: Path = Depends(runs_dir)) -> list[dict[str, Any]]:
    """Signal and risk layer per coverage level and side: trades, win rate, profit factor, mean
    net return, p (mean > 0), exit mix, and the one-position-at-a-time account's return and drawdown."""
    return records(evaluate(get_run(base, run_id).predictions, _bars(), h, rules=Rules(cost_bps=cost_bps)))


@router.get("/runs/{run_id}/equity")
def run_equity(run_id: str, h: int = Query(1, ge=1, le=5), coverage: float = Query(0.3, gt=0, le=1),
               side: Literal["both", "buy", "sell"] = Query("both"), cost_bps: float = Query(10.0, ge=0, le=100),
               base: Path = Depends(runs_dir)) -> dict[str, list[Any]]:
    """Equity curve (units of starting capital) and drawdown of the account, weekly."""
    run = get_run(base, run_id)
    bars = _bars()
    tr = signals(run.predictions, bars, h, coverage, Rules(cost_bps=cost_bps))
    if side != "both":
        tr = tr[tr["side"] == (1 if side == "buy" else -1)]
    idx = bars.index[bars.index >= run.predictions.index[0]]
    curve = equity_curve(tr, idx, h)[["equity", "drawdown"]].resample("W-FRI").last()
    return columns(curve)


@router.get("/runs/{run_id}/edges")
def run_edges(run_id: str, base: Path = Depends(runs_dir)) -> dict[str, Any]:
    """Edge dynamics of an AIM-DG run (Phase 6.3): the monthly on/off timeline with reliability,
    persistence per edge, and the regime tests (stress months against the rest)."""
    from xaglab.data.regimes import REGIMES
    from xaglab.eval import edges as E

    try:
        tl = E.timeline(get_run(base, run_id))
    except ValueError as exc:  # a run without edge selections
        raise HTTPException(422, str(exc)) from exc
    names = E.edges_of(tl)
    per, overall = E.persistence(tl)
    act = E.cluster_activity(tl)
    has_r = all(f"r:{e}" in tl.columns for e in names)
    return clean({
        "edges": names,
        "clusters": {e: E.CLUSTER_OF[e.rsplit("@", 1)[0]] for e in names},
        "dates": [str(d.date()) for d in tl.index],
        "on": tl[[f"on:{e}" for e in names]].astype(int).to_numpy().tolist(),
        "reliability": tl[[f"r:{e}" for e in names]].to_numpy().tolist() if has_r else None,
        "n_edges": tl["n_edges"].tolist(),
        "rate": tl["rate"].tolist(),
        "activity": {c[4:]: act[c].tolist() for c in act.columns if c.startswith("act:")},
        "persistence": per.to_dict(orient="records"),
        "overall": overall,
        "tests": E.regime_tests(tl).to_dict(orient="records"),
        "regimes": [{"label": a, "start": b, "end": c} for a, b, c in REGIMES],
    })
