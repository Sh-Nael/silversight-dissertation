"""The frozen walk-forward calendar: folds, refits and tuning points."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from xaglab.api.deps import calendar
from xaglab.models.tuning import tuning_points

router = APIRouter(tags=["folds"])


class FoldsResponse(BaseModel):
    calendar: dict[str, Any]           # the frozen description (== experiments/folds_v1.json)
    protocol: dict[str, Any]           # validation rule, purge, refit cadence
    refits: list[dict[str, Any]]       # fold, j, origin, train_end, predict_end (dates)
    tuning_points: list[str]           # origin dates of the 17 yearly tuning points


@router.get("/folds", response_model=FoldsResponse)
def folds() -> FoldsResponse:
    cal = calendar()
    d = lambda p: str(cal.index[p].date())
    refits = cal.all_refits()
    idx = tuning_points([r.origin for r in refits], 252)
    return FoldsResponse(
        calendar=cal.describe(),
        protocol=cal.protocol(),
        refits=[{"fold": r.fold, "j": r.j, "origin": d(r.origin), "train_end": d(r.train_end),
                 "predict_end": d(r.predict_end)} for r in refits],
        tuning_points=[d(refits[i].origin) for i in sorted(set(idx))],
    )
