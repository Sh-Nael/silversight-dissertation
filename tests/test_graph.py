"""Tests for the static-graph GNN (step 3.4, D-33). CPU only, so seeds reproduce exactly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from xaglab.eval.folds import make_calendar
from xaglab.eval.harness import _check
from xaglab.models.base import FitView, PredictView
from xaglab.models.neural.forecaster import TunedNeural
from xaglab.models.neural.learner import NeuralLearner
from xaglab.models.neural.nets import LAGS, StaticGraphNet, column_groups

COLS = ["silver__r1", "silver__rv21", "gold__r1", "gold__vol_z60__missing", "copper__r1", "dxy__r1",
        "real_yield_10y__d1", "vix__r1", "event__cpi_days_to"]


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setenv("XAGLAB_DEVICE", "cpu")


def _net(gnn_layers=1, hidden=8, window=6):
    torch.manual_seed(0)
    return StaticGraphNet(column_groups(COLS), hidden=hidden, layers=1, gnn_layers=gnn_layers,
                          dropout=0.0, window=window).eval()


def test_columns_are_grouped_per_node_and_events_go_to_silver():
    g = column_groups(COLS)
    assert g["silver"] == [0, 1, 8] and g["gold"] == [2, 3] and g["real_yield_10y"] == [6]
    with pytest.raises(ValueError, match="no node type"):
        column_groups(["oil__r1"])


@pytest.mark.filterwarnings("ignore::UserWarning")         # PyG: sources are never destinations here
@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # PyG's typing internals on Python 3.13
def test_message_passing_equals_pyg_heteroconv():
    from torch_geometric.nn import GraphConv, HeteroConv

    net, B, H = _net(), 4, 8
    silver, sources = torch.randn(B, H), torch.randn(B, len(net.edges), H)
    names = [f"{d}_{lag}" for d, lag in net.edges]
    for mask in (torch.ones(len(names), dtype=torch.bool), torch.tensor([k % 3 == 0 for k in range(len(names))])):
        convs = {}
        for e, name in enumerate(names):
            if not mask[e]:
                continue  # a masked edge is simply absent from the PyG graph
            c = GraphConv((H, H), H, aggr="mean")
            with torch.no_grad():
                c.lin_rel.weight.copy_(net.rel[0].weight[e])
                c.lin_rel.bias.copy_(net.rel[0].bias[e])
                c.lin_root.weight.copy_(net.root[0].weight[e])
            convs[(name, "to", "silver")] = c
        hetero = HeteroConv(convs, aggr="mean")
        x = {"silver": silver, **{n: sources[:, e] for e, n in enumerate(names)}}
        link = torch.stack([torch.arange(B), torch.arange(B)])  # source i -> silver i in graph i
        edges = {k: link for k in convs}
        want = hetero(x, edges)["silver"]
        with torch.no_grad():
            got = net.conv(0, silver, sources, mask)
        assert torch.allclose(got, want, atol=1e-5)


def test_a_masked_edge_has_no_influence_and_the_scale_is_a_mean():
    net, B, H = _net(), 3, 8
    silver, sources = torch.randn(B, H), torch.randn(B, len(net.edges), H)
    mask = torch.zeros(len(net.edges), dtype=torch.bool)
    mask[[0, 4]] = True
    with torch.no_grad():
        a = net.conv(0, silver, sources, mask)
        changed = sources.clone()
        changed[:, 1] += 100.0                                   # an inactive edge's source
        b = net.conv(0, silver, changed, mask)
        one = net.rel[0](sources)[:, 0] + net.root[0](silver)[:, 0]
        four = net.rel[0](sources)[:, 4] + net.root[0](silver)[:, 4]
    assert torch.allclose(a, b)
    assert torch.allclose(a, (one + four) / 2, atol=1e-6)       # mean, not sum


def test_lagged_states_are_causal():
    net = _net(window=6)
    T = 6 + max(LAGS) - 1
    x = torch.randn(2, T, len(COLS))
    later = x.clone()
    later[:, -1] += 5.0                                           # change only the origin day
    with torch.no_grad():
        s1, src1 = net.encode(x)
        s2, src2 = net.encode(later)
    lag = {lag: [k for k, (_, g) in enumerate(net.edges) if g == lag] for lag in LAGS}
    assert not torch.allclose(src1[:, lag[1]], src2[:, lag[1]])  # today's states see the origin day
    assert torch.equal(src1[:, lag[5]], src2[:, lag[5]])          # states 4 and 20 days back do not
    assert torch.equal(src1[:, lag[21]], src2[:, lag[21]])
    assert not torch.allclose(s1, s2)


@pytest.mark.parametrize("gnn_layers", [1, 2])
def test_learns_a_planted_cross_market_signal(gnn_layers):
    """Silver goes up tomorrow when gold rose 4 days ago: only the gold lag-5 edge carries it."""
    rng = np.random.default_rng(0)
    n = 1400
    X = pd.DataFrame(rng.normal(size=(n, len(COLS))), columns=COLS)
    g = X["gold__r1"].to_numpy()
    up = np.r_[np.full(4, np.nan), (g[:-4] > 0).astype(float)]
    y = {"up1": up, "up5": up, "vol1": np.zeros(n), "vol5": np.zeros(n)}
    params = {"hidden": 16, "layers": 1, "gnn_layers": gnn_layers, "dropout": 0.0, "lr": 3e-3,
              "weight_decay": 1e-5, "lam": 0.5}
    learner = NeuralLearner(net="static_gnn", window=5, max_epochs=200, patience=20)
    train, test = np.arange(30, 1150), np.arange(1150, 1400)
    fitted = learner.fit_rows(X, train, {k: v[train] for k, v in y.items()}, params, seed=0, n_seeds=2)
    pred = learner.predict_rows(fitted, X, test)
    acc = np.mean((pred["up1"] > 0.5) == (up[test] > 0.5))
    assert acc > 0.9, acc


def test_static_gnn_through_one_real_refit(features):
    cal = make_calendar(features.index, features.labels, train_start=features.warmup_end())
    refit = cal.all_refits(None)[0]
    m = TunedNeural(net="static_gnn", window=10, windows=(10,), n_seeds=2, tune_seeds=1, n_trials=2,
                    max_epochs=2, patience=2)
    assert m.name == "static_gnn"
    tuned = m.tune(FitView.build(features, refit, 0))
    assert "gnn_layers" in tuned["joint"]["params"]
    m.set_tuned(tuned)
    m.fit(FitView.build(features, refit, 0))
    view = PredictView.build(features, refit)
    pred = _check(m.predict(view), view.dates, m.name)
    assert pred.notna().all().all()
