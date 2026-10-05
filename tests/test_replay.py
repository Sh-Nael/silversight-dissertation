"""Tests for replaying AIM-DG's pipeline on cached bundles (Phase 5). CPU only."""

from __future__ import annotations

import numpy as np
import pytest

from xaglab.models.aimdg import lab
from xaglab.models.aimdg.replay import Sequential, joint_loss, replay, train_bundle
from xaglab.models.aimdg.synthetic import monthly_refits, planted_break_market

PARAMS = {"hidden": 8, "layers": 1, "gnn_layers": 1, "dropout": 0.0, "lr": 3e-3, "weight_decay": 1e-5,
          "lam": 0.5, "window": 5, "lam2": 0.001, "lam3": 0.001}


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setenv("XAGLAB_DEVICE", "cpu")


def test_replay_chains_refits_and_reproduces_the_forecast_contract(tmp_path):
    fs = planted_break_market(n=800, brk=10_000, noise=0.3, seed=2)      # no break: one regime
    refits = monthly_refits(first_origin=600, n_refits=2, train_start=30)
    bundles = [train_bundle(fs, r, PARAMS, seed=0, n_seeds=1, path=tmp_path / f"r{j}") for j, r in enumerate(refits)]
    assert (tmp_path / "r0" / "final.pt").exists() and (tmp_path / "r1" / "block2.pt").exists()
    reloaded = train_bundle(fs, refits[0], PARAMS, seed=0, n_seeds=1, path=tmp_path / "r0")   # from the cache
    assert reloaded.seconds == bundles[0].seconds

    small = lab.Setting("default", pop_size=10, generations=5)
    pred, diags = replay(fs, bundles, Sequential("default", base=small))
    want_dates = fs.index[refits[0].origin: refits[1].predict_end + 1]
    assert pred.index.equals(want_dates) and list(pred.columns) == ["p_up1", "var1", "p_up5", "var5"]
    probs = pred[["p_up1", "p_up5"]]
    assert ((probs >= 0) & (probs <= 1)).all().all() and (pred[["var1", "var5"]] > 0).all().all()
    assert [d["warm_start"] for d in diags] == [False, True]             # the chain carries the winner
    assert diags[0]["shift"] is None and diags[1]["shift"] is not None   # the clock ticked once per refit
    loss = joint_loss(pred, fs.labels)
    assert len(loss) == len(pred) and np.isfinite(loss).all()

    memoryless = replay(fs, bundles, Sequential("no_warm_start", warm_start=False, base=small))[1]
    assert [d["warm_start"] for d in memoryless] == [False, False] and all(d["rate"] == 0.3 for d in memoryless)
    all_on = replay(fs, bundles, Sequential("all_on", base=lab.Setting("all_on", mode="all_on")))[1]
    assert all(d["n_edges"] == 15 for d in all_on)
