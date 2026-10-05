"""Tests for recency-weighted training (Phase 5 A4, D-37) and tuning-point screens."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from xaglab.models.neural import trainer
from xaglab.models.neural.losses import multitask_loss


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setenv("XAGLAB_DEVICE", "cpu")


def test_recency_weights_halve_every_half_life_and_average_one():
    rows = np.arange(100, 605)                                 # 505 training rows
    w = trainer.recency_weights(rows, half_life=504.0, n=700, device=torch.device("cpu")).numpy()
    assert np.isclose(w[100] / w[604], 0.5) and np.isclose(w[rows].mean(), 1.0)
    assert w[:100].sum() == 0 and w[605:].sum() == 0            # rows outside the training set


def test_weighted_loss_is_a_weighted_mean():
    out = {"logit1": torch.tensor([2.0, -2.0]), "logit5": torch.tensor([0.0, 0.0]),
           "z1": torch.tensor([0.0, 1.0]), "z5": torch.tensor([0.0, 0.0])}
    y = {"up1": torch.tensor([1.0, 1.0]), "up5": torch.tensor([1.0, 0.0]),
         "r1": torch.tensor([1.0, 1.0]), "r5": torch.tensor([1.0, 1.0])}
    plain, _ = multitask_loss(out, y, lam=1.0)
    first, _ = multitask_loss(out, y, lam=1.0, w=torch.tensor([1.0, 0.0]))
    second, _ = multitask_loss(out, y, lam=1.0, w=torch.tensor([0.0, 1.0]))
    assert torch.isclose(plain, (first + second) / 2)
    assert first < second                                       # row 0 is the confident, correct one


def test_recency_variants_are_registered_and_named():
    from xaglab.models import registry
    from xaglab.models.aimdg.forecaster import AimDG

    assert {"static_gnn_rw", "aimdg_rw"} <= set(registry.names())
    m = registry.get("aimdg_rw").build()
    assert isinstance(m, AimDG) and m.name == "aimdg_rw" and m.config()["half_life_rows"] == 504.0
    assert m._learner(1).half_life == 504.0
    assert registry.get("aimdg").build().name == "aimdg"


def test_development_screen_runs_only_the_given_tuning_points(features, tmp_path):
    from xaglab.eval.folds import make_calendar
    from xaglab.eval.harness import ModelSpec, develop
    from xaglab.models.baselines import PooledLinear

    cal = make_calendar(features.index, features.labels, train_start=features.warmup_end())
    dev = develop(ModelSpec(PooledLinear, {"n_trials": 2}, "linear"), features, cal, n_jobs=2, folds=[1],
                  base=tmp_path, points=[0, 2])
    assert [t["tuning_point"] for t in dev.tuning] == [0, 2]   # fold 1 alone has points 0, 1, 2
    assert dev.manifest["points"] == [0, 2] and dev.manifest["tuning"]["n_points"] == 2
