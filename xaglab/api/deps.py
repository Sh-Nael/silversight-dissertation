"""Shared helpers for the API routes: cached data, safe run lookup, JSON conversion.

Caching is safe because the inputs are immutable: snapshots never change (checksummed)
and run folders are write-once. The feature set is built once per process (~1.5 s);
a run is re-read only if its folder changes on disk.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import HTTPException, Request

from xaglab.data.panel import Panel, load_default_panel
from xaglab.eval.folds import FoldCalendar, make_calendar
from xaglab.eval.harness import RunResult, load_run
from xaglab.features.build import FeatureSet, build_features

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


@lru_cache(maxsize=1)
def panel() -> Panel:
    return load_default_panel()


@lru_cache(maxsize=1)
def feature_set() -> FeatureSet:
    return build_features(panel())


@lru_cache(maxsize=1)
def calendar() -> FoldCalendar:
    fs = feature_set()
    return make_calendar(fs.index, fs.labels, train_start=fs.warmup_end())


def runs_dir(request: Request) -> Path:
    return request.app.state.runs_dir


def run_ids(base: Path) -> list[str]:
    if not base.exists():
        return []
    return sorted(p.parent.name for p in base.glob("*/run.json"))


@lru_cache(maxsize=64)
def _load(path: str, mtime: float) -> RunResult:  # mtime is part of the cache key
    return load_run(Path(path))


def get_run(base: Path, run_id: str) -> RunResult:
    """Load a run by ID; 404 if it doesn't exist. IDs can't point outside `base`."""
    if not _RUN_ID.match(run_id):
        raise HTTPException(404, f"no run {run_id!r}")
    path = base / run_id
    if not (path / "run.json").is_file():
        raise HTTPException(404, f"no run {run_id!r}")
    return _load(str(path), (path / "run.json").stat().st_mtime)


def clean(v: Any) -> Any:
    """NaN/inf → None and numpy scalars → Python, recursively, for JSON."""
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if not np.isfinite(f) else f
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def columns(df: pd.DataFrame, date_col: str = "date") -> dict[str, list[Any]]:
    """Columnar JSON for charts: {"date": [...], "<col>": [...]} with NaN → null."""
    out: dict[str, list[Any]] = {}
    if isinstance(df.index, pd.DatetimeIndex):
        out[date_col] = [d.strftime("%Y-%m-%d") for d in df.index]
    for c in df.columns:
        vals = df[c].to_numpy()
        if vals.dtype.kind == "f":
            out[str(c)] = [None if not np.isfinite(x) else float(x) for x in vals]
        else:
            out[str(c)] = clean(vals.tolist())
    return out


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return clean(df.to_dict(orient="records"))
