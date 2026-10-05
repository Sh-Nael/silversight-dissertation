"""Equal-weight ensembles of finished runs, saved as runs of their own.

An ensemble averages its members' calibrated probabilities and variance forecasts, day by
day, with equal weights: there is nothing to tune, so evaluating it on the test period once
is as legitimate as evaluating any single model. Which models to combine is decided from
development scores, never from test results (D-22).

The result is written as a run folder (predictions, an empty timings table, a manifest that
lists the member run ids and their commits), so every report, the signal layer and the app
work on it unchanged.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from xaglab.eval.harness import LABEL_COLUMNS, RunResult, _git_state, _versions
from xaglab.models.base import FORECAST_COLUMNS


def ensemble(members: list[RunResult], name: str = "ensemble") -> RunResult:
    """Average the members' forecasts (equal weights); labels and fold/refit ids from the first."""
    if len(members) < 2:
        raise ValueError("an ensemble needs at least two member runs")
    idx = members[0].predictions.index
    for m in members[1:]:
        if not m.predictions.index.equals(idx):
            raise ValueError(f"{m.run_id} was evaluated on different origins than {members[0].run_id}")
    stack = np.stack([m.predictions[list(FORECAST_COLUMNS)].to_numpy(float) for m in members])
    pred = pd.DataFrame(stack.mean(axis=0), index=idx, columns=list(FORECAST_COLUMNS))
    keep = [c for c in ("fold", "refit", *LABEL_COLUMNS) if c in members[0].predictions.columns]
    pred = pd.concat([pred, members[0].predictions[keep]], axis=1)
    started = datetime.now(UTC)
    run_id = f"{name}-{started.strftime('%Y%m%d-%H%M%S')}"
    manifest = {
        "run_id": run_id,
        "model": name,
        "model_class": "xaglab.eval.ensemble.ensemble",
        "model_kwargs": {"members": [m.run_id for m in members]},
        "model_config": {"weights": "equal", "combined": "p_up (probabilities) and var (variances), per day"},
        "protocol": {**members[0].manifest["protocol"], "n_refits": members[0].manifest["protocol"]["n_refits"]},
        "seed": None,
        "n_jobs": 1,
        "started_utc": started.isoformat(timespec="seconds"),
        "wall_seconds": 0.0,
        "tuning": None,
        "folds": members[0].manifest["folds"],
        "fit_seconds_total": 0.0,
        "fold_fingerprint": members[0].manifest["fold_fingerprint"],
        "members": [{"run_id": m.run_id, "model": m.manifest["model"], "commit": m.manifest["code"]["commit"]}
                    for m in members],
        "data": members[0].manifest["data"],
        "code": _git_state(),
        "env": _versions(),
    }
    timings = pd.DataFrame(columns=["fold", "refit", "origin", "n_train", "n_pred", "fit_s", "predict_s"])
    return RunResult(run_id, pred, timings, [], manifest)
