"""Compare runs: scoreboard, development scores, significance, per-fold and cumulative losses.

All numbers come from xaglab.eval.report, the same functions the CLI `report` uses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from xaglab.api.deps import clean, columns, get_run, records, runs_dir
from xaglab.eval.report import (
    cumulative_difference,
    dev_scoreboard,
    per_fold,
    scoreboard,
    significance,
)

router = APIRouter(tags=["compare"])

Loss = Literal["brier", "logloss", "qlike"]


class CompareRequest(BaseModel):
    runs: list[str] = Field(min_length=2, description="run IDs, one per model")
    reference: str = Field("climatology", description="model every other model is tested against")
    losses: list[Loss] = ["brier", "logloss", "qlike"]
    horizons: list[Literal[1, 5]] = [1, 5]


class CompareResponse(BaseModel):
    models: list[str]
    reference: str
    scoreboard: list[dict[str, Any]]
    dev_scoreboard: list[dict[str, Any]]
    significance: list[dict[str, Any]]
    per_fold: dict[str, dict[str, list[float | None]]]        # "brier_h1" -> model -> 8 values
    cumulative: dict[str, dict[str, list[Any]]]               # "brier_h1" -> columns (weekly)


@router.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest, base: Path = Depends(runs_dir)) -> CompareResponse:
    """Compare runs against a reference model on the same test days."""
    runs = [get_run(base, i) for i in req.runs]
    models = [r.manifest["model"] for r in runs]
    if len(set(models)) != len(models):
        raise HTTPException(422, "pick one run per model")
    if req.reference not in models:
        raise HTTPException(422, f"reference {req.reference!r} is not among the selected runs' models")
    try:
        board = scoreboard(runs)
    except ValueError as exc:  # runs evaluated on different test days
        raise HTTPException(422, str(exc)) from exc

    sig, pf, cum = [], {}, {}
    for loss in req.losses:
        for h in req.horizons:
            key = f"{loss}_h{h}"
            sig.append(significance(runs, req.reference, loss, h))
            f = per_fold(runs, req.reference, loss, h)
            pf[key] = {m: clean(list(f[m])) for m in f.columns}
            cum[key] = columns(cumulative_difference(runs, req.reference, loss, h))
    dev = dev_scoreboard(runs)
    return CompareResponse(
        models=models, reference=req.reference, scoreboard=records(board),
        dev_scoreboard=records(dev) if len(dev) else [],
        significance=records(pd.concat(sig, ignore_index=True)) if sig else [],
        per_fold=pf, cumulative=cum,
    )
