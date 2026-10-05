from __future__ import annotations

import json

import numpy as np

from xaglab.eval.folds import make_calendar
from xaglab.paths import EXPERIMENTS_DIR


def _cal(features):
    return make_calendar(features.index, features.labels, train_start=features.warmup_end())


def test_calendar_matches_frozen_file(features):
    """The folds were frozen before any model ran; regeneration must reproduce them."""
    frozen = json.loads((EXPERIMENTS_DIR / "folds_v1.json").read_text())
    assert _cal(features).describe() == frozen


def test_folds_are_contiguous_equal_and_cover_the_test_period(features):
    cal = _cal(features)
    assert len(cal.folds) == 8
    sizes = [f.test_end - f.test_start + 1 for f in cal.folds]
    assert max(sizes) - min(sizes) <= 1 and 500 <= min(sizes) <= 540
    for a, b in zip(cal.folds, cal.folds[1:]):
        assert b.test_start == a.test_end + 1
    assert str(cal.date(cal.folds[0].test_start).date()) == "2010-01-04"
    assert features.labels.iloc[cal.folds[-1].test_end].notna().all()


def test_refits_partition_each_fold_exactly_once(features):
    cal = _cal(features)
    for f in cal.folds:
        covered = np.concatenate([np.arange(r.origin, r.predict_end + 1) for r in cal.refits(f)])
        assert np.array_equal(covered, np.arange(f.test_start, f.test_end + 1))
    # the static protocol (one fit per fold) partitions too
    for f in cal.folds:
        (only,) = cal.refits(f, every=10_000)
        assert (only.origin, only.predict_end) == (f.test_start, f.test_end)


def test_validation_windows_per_horizon(features):
    """D-24: 1 year for h=1, 3 years for h=5, both capped at half the training rows and
    purged from inner training; the first refits of fold 0 are the capped case."""
    cal = _cal(features)
    capped = 0
    for r in cal.all_refits():
        n_train = r.train_end - r.train_start + 1
        for h, target in ((1, 252), (5, 756)):
            val, inner = r.val_for(h), r.inner_for(h)
            n_val = val.stop - val.start
            assert n_val == min(target, n_train // 2)
            assert val.stop - 1 == r.train_end                 # validation ends the training span
            assert inner.stop - 1 + cal.purge < val.start       # purge between inner and val
            assert inner.start == r.train_start
        capped += (r.val_for(5).stop - r.val_for(5).start) < 756
    first = cal.refits(cal.folds[0])[0]
    assert first.val_for(5).stop - first.val_for(5).start == (first.train_end - first.train_start + 1) // 2
    assert 0 < capped < 50  # only the earliest refits are capped
