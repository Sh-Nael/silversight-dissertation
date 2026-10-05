"""Tests for the Phase 5 selection lab (fixed-network ablations and sweep). CPU only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from xaglab.models.aimdg import lab
from xaglab.models.aimdg.evolution import MaskScorer
from xaglab.models.aimdg.reliability import contributions
from xaglab.models.neural import trainer
from xaglab.models.neural.data import prepare
from xaglab.models.neural.learner import _targets_for
from xaglab.models.neural.nets import StaticGraphNet, column_groups

COLS = ["silver__r1", "silver__rv21", "gold__r1", "copper__r1", "dxy__r1", "real_yield_10y__d1", "vix__r1"]


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setenv("XAGLAB_DEVICE", "cpu")


def _context(seed=0):
    """A block context on random data with tiny, randomly initialised supernets."""
    rng = np.random.default_rng(seed)
    n = 400
    X = pd.DataFrame(rng.normal(size=(n, len(COLS))), columns=COLS)
    rows = np.arange(30, 340)
    val = np.arange(345, 400)
    up = (rng.uniform(size=n) < 0.5).astype(float)
    y = {"up1": up, "up5": up, "vol1": rng.normal(0, 0.3, n), "vol5": rng.normal(0, 0.3, n)}
    prep = prepare(X, rows, 25, torch.device("cpu"))
    tgt = _targets_for({k: v[rows] for k, v in y.items()}, rows, n, torch.device("cpu"))
    groups = column_groups(COLS)
    torch.manual_seed(seed)
    nets = [StaticGraphNet(groups, 8, 1, 1, 0.0, "lstm", 5, edge_dropout=(0.2, 1.0)).eval() for _ in range(2)]
    _, tail = trainer.early_stop_split(rows)
    bn = lab.BlockNets(nets, nets, prep.scaler, prep.missing_cols, 5, {}, 0.0)
    return lab.BlockContext(bn, prep, tgt, tail, contributions(nets, prep, tgt, tail),
                            MaskScorer(nets, prep, tgt, tail), val)


def test_every_setting_yields_a_valid_mask_and_forecasts():
    ctx = _context()
    small = [lab.with_name(lab.Setting("d", pop_size=10, generations=3), "d"),
             lab.Setting("a1", mode="topk", k=3), lab.Setting("all", mode="all_on"),
             lab.Setting("ex", mode="exhaustive"), lab.Setting("b0", beta=0.0, pop_size=10, generations=3),
             lab.Setting("sw", subwindow=10, pop_size=10, generations=3)]
    seen = {}
    for s in small:
        mask, info = lab.select_mask(s, ctx, lam2_default=0.001, lam3_default=0.001)
        assert mask.shape == (15,) and mask.dtype == bool and "seconds" in info
        seen[s.name] = mask
        pred = lab.block_predictions(ctx, mask)
        assert set(pred) == {"up1", "up5", "vol1", "vol5"} and len(pred["up1"]) == len(ctx.val_rows)
        assert all(net.fixed_mask is None for net in ctx.nets.final)          # mask cleared afterwards
    assert seen["all"].all() and seen["a1"].sum() == 3
    # the exhaustive optimum is never worse than the search's winner (same scorer, same fitness)
    from xaglab.models.aimdg.evolution import fitness
    base = float(ctx.scorer(np.ones((1, 15), bool))[0])
    f = lambda m: fitness(ctx.scorer(m[None]), base, m[None], None, 0.001, 0.001)[0]
    assert f(seen["ex"]) <= f(seen["d"]) + 1e-9


def test_summary_table_compares_settings_to_the_reference():
    def point(k, d_loss, a_loss, d_mask, a_mask):
        return {"point": k, "settings": {
            "default": {"cv_loss": d_loss, "components": {"direction_h5": 0.69}, "masks": [d_mask] * 3, "n_edges": sum(d_mask)},
            "a1": {"cv_loss": a_loss, "components": {"direction_h5": 0.68}, "masks": [a_mask] * 3, "n_edges": sum(a_mask)}}}
    m0, m1 = [1, 0, 1] + [0] * 12, [1, 1, 1] + [0] * 12
    t = lab.summarise([point(0, -4.7, -4.71, m0, m1), point(1, -4.6, -4.59, m0, m0)]).set_index("setting")
    assert t.loc["default", "diff_vs_ref"] == 0 and t.loc["default", "points_better"] == 0
    assert np.isclose(t.loc["a1", "diff_vs_ref"], (-0.01 + 0.01) / 2) and t.loc["a1", "points_better"] == 1
    assert np.isclose(t.loc["a1", "diff_direction_h5"], -0.01) and t.loc["a1", "edges_moved_vs_ref"] == 0.5
    assert t.loc["a1", "edges"] == 2.5
