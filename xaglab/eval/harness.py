"""The walk-forward harness: drives any :class:`Forecaster` through every refit.

For each refit (fold k, refit j) the harness
  1. builds a FitView (no row after train_end) and calls ``model.fit``;
  2. builds a PredictView (no labels at all) and calls ``model.predict``;
  3. checks the returned frame (right dates, right columns, probabilities in [0, 1],
     variances positive) and joins the true labels *afterwards*, outside the model;
  4. records wall-clock time for fit and predict separately.

Tunable models (``model.tunable``) first go through a tuning stage: hyperparameters are
searched at yearly tuning points (in parallel), and each refit then receives the
settings of its latest tuning point (D-26). Tuning results, including the
out-of-fold development scores, are saved with the run.

Stateless models (fit() resets everything) run their refits in parallel processes,
one thread each. Stateful models (AIM-DG warm-starts its edge search from the
previous refit) run sequentially in calendar order.

Every run is written to ``experiments/runs/<run_id>/`` with its predictions,
timings, diagnostics and a manifest recording the exact code commit, data
checksums, fold fingerprint, configuration and library versions.

**Development runs** (:func:`develop`) run the tuning stage only and are written to
``experiments/dev/<run_id>/``. They touch no test row: every tuning point sees only its
own training span, and the result is the out-of-fold development score (D-22, D-26).
A later test run can reuse their settings exactly (``run(..., tuning_from=...)``), so
the configuration judged on development scores is the one that meets the test folds.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from xaglab.eval.folds import FoldCalendar, Refit, index_fingerprint
from xaglab.features.build import FeatureSet
from xaglab.models.base import FORECAST_COLUMNS, FitView, Forecaster, PredictView
from xaglab.models.tuning import tuning_points
from xaglab.paths import DEV_DIR, ROOT, RUNS_DIR, SNAPSHOT_DIR

LABEL_COLUMNS = ("up1", "up5", "rv1", "rv5", "ret1", "ret5")


@dataclass(frozen=True)
class ModelSpec:
    """How to construct a model: a class and keyword arguments (picklable for workers)."""

    cls: type[Forecaster]
    kwargs: dict[str, Any]
    name: str

    def build(self) -> Forecaster:
        return self.cls(**self.kwargs)


@dataclass
class RunResult:
    run_id: str
    predictions: pd.DataFrame
    timings: pd.DataFrame
    diagnostics: list[dict]
    manifest: dict
    path: Path | None = None
    tuning: list[dict] | None = None


def _check(pred: pd.DataFrame, dates: pd.DatetimeIndex, name: str) -> pd.DataFrame:
    missing = [c for c in FORECAST_COLUMNS if c not in pred.columns]
    if missing:
        raise ValueError(f"{name}: forecast missing columns {missing}")
    if not pred.index.equals(dates):
        raise ValueError(f"{name}: forecast index does not match the requested origins")
    p = pred[["p_up1", "p_up5"]].to_numpy(float)
    if np.any((p[~np.isnan(p)] < 0) | (p[~np.isnan(p)] > 1)):
        raise ValueError(f"{name}: probabilities outside [0, 1]")
    v = pred[["var1", "var5"]].to_numpy(float)
    if np.any(v[~np.isnan(v)] <= 0):
        raise ValueError(f"{name}: non-positive variance forecast")
    return pred[list(FORECAST_COLUMNS)]


def _one_refit(model: Forecaster, fs: FeatureSet, refit: Refit, seed: int,
               tuned: dict | None = None) -> tuple[pd.DataFrame, dict, dict]:
    if tuned is not None:
        model.set_tuned(tuned)
    t0 = time.perf_counter()
    model.fit(FitView.build(fs, refit, seed))
    t1 = time.perf_counter()
    view = PredictView.build(fs, refit)
    pred = _check(model.predict(view), view.dates, model.name)
    t2 = time.perf_counter()
    pred = pred.assign(fold=refit.fold, refit=refit.j)
    timing = {"fold": refit.fold, "refit": refit.j, "origin": str(view.dates[0].date()),
              "n_train": refit.train_end - refit.train_start + 1, "n_pred": len(pred),
              "fit_s": t1 - t0, "predict_s": t2 - t1}
    diag = {"fold": refit.fold, "refit": refit.j, **model.diagnostics()}
    return pred, timing, diag


def _stateless_task(spec: ModelSpec, fs: FeatureSet, refit: Refit, seed: int, tuned: dict | None):
    return _one_refit(spec.build(), fs, refit, seed, tuned)


def _tune_task(spec: ModelSpec, fs: FeatureSet, refit: Refit, seed: int) -> dict:
    t0 = time.perf_counter()
    result = spec.build().tune(FitView.build(fs, refit, seed))
    return {"results": result, "seconds": time.perf_counter() - t0}


def _tuning_stage(spec: ModelSpec, fs: FeatureSet, refits: list[Refit], seed: int, n_jobs: int,
                  every: int, only: list[int] | None = None) -> tuple[list[dict], list[dict]]:
    """Run every tuning point (or the ``only`` given ones, by number: development screens);
    return per-refit settings and the tuning log."""
    idx = tuning_points([r.origin for r in refits], every)
    points = sorted(set(idx))
    if only is not None:
        points = [points[k] for k in only]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        outs = Parallel(n_jobs=n_jobs)(
            delayed(_tune_task)(spec, fs, refits[i], seed * 1000 + k) for k, i in enumerate(points))
    by_point = dict(zip(points, outs, strict=True))
    numbers = dict(zip(points, only, strict=True)) if only is not None else {i: k for k, i in enumerate(points)}
    log = [{"tuning_point": numbers[i], "fold": refits[i].fold, "refit": refits[i].j,
            "origin": str(fs.index[refits[i].origin].date()),
            "train_end": str(fs.index[refits[i].train_end].date()),
            "seconds": round(by_point[i]["seconds"], 2), "results": by_point[i]["results"]}
           for i in points]
    return [by_point[i]["results"] if i in by_point else None for i in idx], log


def _reuse_tuning(source: Path | str, spec: ModelSpec, fs: FeatureSet, refits: list[Refit], seed: int,
                  every: int, fingerprint: str) -> tuple[list[dict], list[dict]]:
    """Per-refit settings from a development run, after checking it matches this run."""
    dev = load_dev(source)
    m = dev.manifest
    # model_config catches changed defaults that the keyword arguments alone would not show
    want = {"model": spec.name, "model_kwargs": spec.kwargs, "model_config": spec.build().config(),
            "seed": seed, "fold_fingerprint": fingerprint}
    for k, v in want.items():
        if json.loads(json.dumps(m[k])) != json.loads(json.dumps(v)):
            raise ValueError(f"tuning_from {dev.run_id}: {k} differs ({m[k]!r} vs {v!r})")
    if m["tuning"]["every"] != every:
        raise ValueError(f"tuning_from {dev.run_id}: tuning interval differs")
    idx = tuning_points([r.origin for r in refits], every)
    origin = lambda i: str(fs.index[refits[i].origin].date())
    have = {t["origin"]: t for t in dev.tuning}
    points = sorted(set(idx))
    missing = [origin(i) for i in points if origin(i) not in have]
    if missing:
        raise ValueError(f"tuning_from {dev.run_id}: no tuning point at {missing}")
    return [have[origin(i)]["results"] for i in idx], [have[origin(i)] for i in points]


def run(
    spec: ModelSpec,
    fs: FeatureSet,
    cal: FoldCalendar,
    every: int | None = None,
    seed: int = 0,
    n_jobs: int = 1,
    save: bool = True,
    tag: str = "",
    folds: list[int] | None = None,
    tuning_from: Path | str | None = None,
) -> RunResult:
    refits = [r for r in cal.all_refits(every) if folds is None or r.fold in folds]
    started = datetime.now(UTC)
    t0 = time.perf_counter()
    probe = spec.build()
    tuned, tuning_log = [None] * len(refits), None
    if getattr(probe, "tunable", False):
        if tuning_from is not None:
            tuned, tuning_log = _reuse_tuning(tuning_from, spec, fs, refits, seed, probe.tune_every,
                                              index_fingerprint(cal.index))
        else:
            tuned, tuning_log = _tuning_stage(spec, fs, refits, seed, n_jobs, probe.tune_every)
    elif tuning_from is not None:
        raise ValueError(f"{spec.name} has no tuning stage; tuning_from does not apply")
    t_tuned = time.perf_counter()
    if probe.stateless and n_jobs != 1:
        with parallel_config(backend="loky", inner_max_num_threads=1):
            outs = Parallel(n_jobs=n_jobs)(
                delayed(_stateless_task)(spec, fs, r, seed, t) for r, t in zip(refits, tuned, strict=True))
    else:
        outs = [_one_refit(probe, fs, r, seed, t) for r, t in zip(refits, tuned, strict=True)]
    wall = time.perf_counter() - t0

    preds = pd.concat([o[0] for o in outs]).sort_index()
    labels = fs.labels.loc[preds.index, list(LABEL_COLUMNS)]
    predictions = pd.concat([preds, labels], axis=1)
    timings = pd.DataFrame([o[1] for o in outs])
    diagnostics = [o[2] for o in outs]

    run_id = _run_id(spec, tag, started)
    manifest = _manifest(run_id, spec, probe, cal, refits, every, seed, n_jobs, started, wall, folds)
    manifest["tuning"] = None if tuning_log is None else {
        "n_points": len(tuning_log), "every": probe.tune_every,
        "wall_seconds": round(t_tuned - t0, 2),
        "cpu_seconds": round(sum(t["seconds"] for t in tuning_log), 2),
        "from": None if tuning_from is None else load_dev(tuning_from).run_id}
    manifest["fit_seconds_total"] = round(float(timings["fit_s"].sum()), 2)
    result = RunResult(run_id, predictions, timings, diagnostics, manifest, tuning=tuning_log)
    if save:
        result.path = _save(result)
    return result


@dataclass
class DevResult:
    """A development run: the tuning stage only, no test row touched."""

    run_id: str
    manifest: dict
    tuning: list[dict]
    path: Path | None = None


def develop(spec: ModelSpec, fs: FeatureSet, cal: FoldCalendar, seed: int = 0, n_jobs: int = 1,
            save: bool = True, tag: str = "", folds: list[int] | None = None,
            base: Path = DEV_DIR, points: list[int] | None = None) -> DevResult:
    """Run only the tuning stage of a tunable model and record its development scores.
    ``points``: tuning-point numbers to run (a screen); None = all."""
    refits = [r for r in cal.all_refits(None) if folds is None or r.fold in folds]
    probe = spec.build()
    if not getattr(probe, "tunable", False):
        raise ValueError(f"{spec.name} has no tuning stage, so no development run")
    started = datetime.now(UTC)
    t0 = time.perf_counter()
    _, log = _tuning_stage(spec, fs, refits, seed, n_jobs, probe.tune_every, only=points)
    wall = time.perf_counter() - t0
    run_id = _run_id(spec, "dev" + ("-" + tag if tag else ""), started)
    manifest = _manifest(run_id, spec, probe, cal, refits, None, seed, n_jobs, started, wall, folds)
    manifest["stage"] = "development"
    manifest["points"] = points
    manifest["tuning"] = {"n_points": len(log), "every": probe.tune_every, "wall_seconds": round(wall, 2),
                          "cpu_seconds": round(sum(t["seconds"] for t in log), 2)}
    res = DevResult(run_id, manifest, log)
    if save:
        out = base / run_id
        out.mkdir(parents=True, exist_ok=False)
        (out / "run.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        (out / "tuning.json").write_text(json.dumps(log, indent=1, default=str) + "\n")
        res.path = out
    return res


def load_dev(path: Path | str, base: Path = DEV_DIR) -> DevResult:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = base / path
    manifest = json.loads((path / "run.json").read_text())
    return DevResult(manifest["run_id"], manifest, json.loads((path / "tuning.json").read_text()), path)


def _run_id(spec: ModelSpec, tag: str, started: datetime) -> str:
    return f"{spec.name}{'-' + tag if tag else ''}-{started.strftime('%Y%m%d-%H%M%S')}"


def _manifest(run_id: str, spec: ModelSpec, probe: Forecaster, cal: FoldCalendar, refits: list[Refit],
              every: int | None, seed: int, n_jobs: int, started: datetime, wall: float,
              folds: list[int] | None) -> dict:
    return {
        "run_id": run_id,
        "model": spec.name,
        "model_class": f"{spec.cls.__module__}.{spec.cls.__name__}",
        "model_kwargs": spec.kwargs,
        "model_config": probe.config(),
        "protocol": {**cal.protocol(), "refit_every": every or cal.refit_every,
                     "n_refits": len(refits)},
        "seed": seed,
        "n_jobs": n_jobs,
        "started_utc": started.isoformat(timespec="seconds"),
        "wall_seconds": round(wall, 2),
        "folds": folds or list(range(len(cal.folds))),
        "fold_fingerprint": index_fingerprint(cal.index),
        "data": _snapshot_checksums(),
        "code": _git_state(),
        "env": _versions(),
    }


def _save(res: RunResult, base: Path = RUNS_DIR) -> Path:
    out = base / res.run_id
    out.mkdir(parents=True, exist_ok=False)
    res.predictions.to_parquet(out / "predictions.parquet")
    res.timings.to_csv(out / "timings.csv", index=False)
    (out / "diagnostics.json").write_text(json.dumps(res.diagnostics, indent=1, default=str))
    (out / "run.json").write_text(json.dumps(res.manifest, indent=2, default=str) + "\n")
    if res.tuning is not None:
        (out / "tuning.json").write_text(json.dumps(res.tuning, indent=1, default=str) + "\n")
    return out


def save_run(res: RunResult, base: Path = RUNS_DIR) -> Path:
    """Write a run folder under `base` (write-once). Used by `run(..., save=True)`."""
    res.path = _save(res, base)
    return res.path


def load_run(path: Path | str, base: Path = RUNS_DIR) -> RunResult:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = base / path
    manifest = json.loads((path / "run.json").read_text())
    return RunResult(
        run_id=manifest["run_id"],
        predictions=pd.read_parquet(path / "predictions.parquet"),
        timings=pd.read_csv(path / "timings.csv"),
        diagnostics=json.loads((path / "diagnostics.json").read_text()),
        manifest=manifest,
        path=path,
        tuning=json.loads((path / "tuning.json").read_text()) if (path / "tuning.json").exists() else None,
    )


def _snapshot_checksums() -> dict:
    out = {}
    for m in sorted(SNAPSHOT_DIR.glob("*/MANIFEST.json")):
        man = json.loads(m.read_text())
        out[man["snapshot"]] = {f: meta["sha256"][:16] for f, meta in man["files"].items()}
    return out


def _git_state() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=False).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def _versions() -> dict:
    import importlib.metadata as md

    pkgs = ["numpy", "pandas", "scikit-learn", "statsmodels", "arch", "torch",
            "torch-geometric", "deap"]
    out = {"python": platform.python_version(), "machine": platform.machine()}
    for p in pkgs:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    try:
        import torch

        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return out
