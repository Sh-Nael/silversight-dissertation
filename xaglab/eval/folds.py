"""Walk-forward fold calendar (frozen 29 Jul 2026, D-03) and refit schedule (D-13).

Terminology. A *forecast origin* t is a trading day; the forecast is made at the
close of t for the label window (t, t+h]. Row t's label is fully known at the close
of t + h.

Folds. Test origins run from the first trading day of 2010 to the last origin
whose longest-horizon label exists, split into ``n_folds`` contiguous blocks of
equal size (~520 origins each). Training spans expand: everything before the block.

Refits. Inside each fold the model is refitted every ``refit_every`` origins
(accuracy-first protocol, D-13). At a refit origin r the model may train on rows t
with t + H <= r, where H is the longest label horizon: the *purge* removes exactly
the rows whose labels are not yet known at r (Lopez de Prado, 2018).

Embargo. The frozen design lists a 21-day embargo after each test block. In a
forward-only walk-forward no training row ever comes after a test block, so the
embargo has nothing to remove. It is recorded here and reported as inactive,
rather than silently dropped.

Validation. Each refit sets aside its last training rows for early stopping,
calibration, tuning and mask selection, separated from the inner training rows by a
further purge of H. The window length depends on the horizon (D-24): the
1-day horizon uses ``val_size`` (252, one year); longer horizons use
``val_size_long`` (756, three years), because overlapping 5-day outcomes carry about
one fifth of the information per row. Every window is capped at half the training
rows, so the earliest refits (fold 0 has ~750 training rows) still keep most of
their data for fitting. The validation rule is part of the protocol recorded in
every run manifest; the frozen fold boundaries (folds_v1.json) do not depend on it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    k: int
    test_start: int  # position in the panel index, inclusive
    test_end: int  # inclusive


@dataclass(frozen=True)
class Refit:
    fold: int
    j: int  # refit number inside the fold
    origin: int  # position of the first origin predicted by this fit
    train_start: int
    train_end: int  # inclusive; = origin - purge
    val_start: int  # 1-day inner split: validation rows [val_start, train_end]
    inner_end: int  # 1-day inner training rows [train_start, inner_end]
    predict_end: int  # inclusive: last origin predicted by this fit
    val_start_long: int = -1  # the same split for horizons > 1 (D-24)
    inner_end_long: int = -1

    @property
    def train(self) -> slice:
        return slice(self.train_start, self.train_end + 1)

    @property
    def inner(self) -> slice:
        return slice(self.train_start, self.inner_end + 1)

    @property
    def val(self) -> slice:
        return slice(self.val_start, self.train_end + 1)

    def inner_for(self, h: int) -> slice:
        """Inner training rows for horizon h (fit candidate settings here)."""
        return self.inner if h == 1 else slice(self.train_start, self.inner_end_long + 1)

    def val_for(self, h: int) -> slice:
        """Validation rows for horizon h (choose settings and calibrate here)."""
        return self.val if h == 1 else slice(self.val_start_long, self.train_end + 1)

    @property
    def predict(self) -> slice:
        return slice(self.origin, self.predict_end + 1)


@dataclass(frozen=True)
class FoldCalendar:
    index: pd.DatetimeIndex
    folds: tuple[Fold, ...]
    train_start: int
    purge: int
    embargo: int
    refit_every: int
    val_size: int
    val_size_long: int = 756
    val_cap: float = 0.5  # validation never exceeds this share of the training rows

    def val_length(self, n_train: int, h: int) -> int:
        target = self.val_size if h == 1 else self.val_size_long
        return min(target, int(n_train * self.val_cap))

    def refits(self, fold: Fold, every: int | None = None) -> list[Refit]:
        every = every or self.refit_every
        out = []
        for j, origin in enumerate(range(fold.test_start, fold.test_end + 1, every)):
            train_end = origin - self.purge
            n_train = train_end - self.train_start + 1
            val_start = train_end - self.val_length(n_train, 1) + 1
            val_start_long = train_end - self.val_length(n_train, 5) + 1
            out.append(Refit(
                fold=fold.k, j=j, origin=origin, train_start=self.train_start,
                train_end=train_end, val_start=val_start,
                inner_end=val_start - self.purge - 1,
                predict_end=min(origin + every - 1, fold.test_end),
                val_start_long=val_start_long,
                inner_end_long=val_start_long - self.purge - 1,
            ))
        return out

    def protocol(self) -> dict:
        """The within-training validation rule, recorded in every run manifest."""
        return {"val_size_h1": self.val_size, "val_size_long": self.val_size_long,
                "val_cap_share_of_train": self.val_cap, "purge": self.purge,
                "refit_every": self.refit_every}

    def all_refits(self, every: int | None = None) -> list[Refit]:
        return [r for f in self.folds for r in self.refits(f, every)]

    def date(self, pos: int) -> pd.Timestamp:
        return self.index[pos]

    # ------------------------------------------------------------ persistence

    def describe(self) -> dict:
        d = lambda p: str(self.index[p].date())
        return {
            "train_start": d(self.train_start),
            "purge": self.purge,
            "embargo": self.embargo,
            "embargo_note": "inactive: forward-only walk-forward, no training row follows a test block",
            "refit_every": self.refit_every,
            "val_size": self.val_size,
            "index_fingerprint": index_fingerprint(self.index),
            "folds": [
                {"k": f.k, "test_start": d(f.test_start), "test_end": d(f.test_end),
                 "n_test": f.test_end - f.test_start + 1,
                 "n_refits": len(self.refits(f)),
                 "first_train_end": d(f.test_start - self.purge)}
                for f in self.folds
            ],
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.describe(), indent=2) + "\n")


def index_fingerprint(index: pd.DatetimeIndex) -> str:
    return hashlib.sha256("\n".join(str(d.date()) for d in index).encode()).hexdigest()[:16]


def make_calendar(
    index: pd.DatetimeIndex,
    labels: pd.DataFrame,
    train_start: str | pd.Timestamp,
    test_start: str = "2010-01-01",
    n_folds: int = 8,
    purge: int = 5,
    embargo: int = 21,
    refit_every: int = 21,
    val_size: int = 252,
) -> FoldCalendar:
    """Build the calendar. `labels` defines the last usable origin (all labels present)."""
    usable = np.flatnonzero(labels.notna().all(axis=1).to_numpy())
    first = int(index.searchsorted(pd.Timestamp(test_start)))
    last = int(usable.max())
    blocks = np.array_split(np.arange(first, last + 1), n_folds)
    folds = tuple(Fold(k, int(b[0]), int(b[-1])) for k, b in enumerate(blocks))
    return FoldCalendar(index, folds, int(index.searchsorted(pd.Timestamp(train_start))),
                        purge, embargo, refit_every, val_size)
