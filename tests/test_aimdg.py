"""Tests for AIM-DG (Phase 4). CPU only, so seeds reproduce exactly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from xaglab.models.aimdg.reliability import contributions, reliability, subwindows
from xaglab.models.neural.data import prepare
from xaglab.models.neural.learner import NeuralLearner, _targets_for

COLS = ["silver__r1", "silver__rv21", "gold__r1", "copper__r1", "dxy__r1", "real_yield_10y__d1", "vix__r1"]


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setenv("XAGLAB_DEVICE", "cpu")


# ------------------------------------------------------------------ 4.2 reliability

def test_subwindows_are_full_months_ending_at_the_last_row():
    w = subwindows(100, 21)
    assert [(s.start, s.stop) for s in w] == [(16, 37), (37, 58), (58, 79), (79, 100)]
    with pytest.raises(ValueError):
        subwindows(41, 21)


def test_reliability_formula_by_hand():
    # two edges, two sub-windows of 2 rows: phi = [[1, 3], [0, 0]] -> pooled sd of {1,3,0,0}
    contrib = np.array([[1.0, 0.0], [1.0, 0.0], [3.0, 0.0], [3.0, 0.0]])
    rel = reliability(contrib, size=2, alpha=0.5, beta=0.5)
    sd = np.std([1.0, 3.0, 0.0, 0.0])
    z = np.array([1.0, 3.0]) / sd
    want = 1 / (1 + np.exp(-(0.5 * z.mean() - 0.5 * z.var())))
    assert np.allclose(rel.phi, [[1, 3], [0, 0]])
    assert np.isclose(rel.r[0], want) and np.isclose(rel.r[1], 0.5)


def test_steady_edge_beats_erratic_edge_and_harmful_edge_scores_lowest():
    rng = np.random.default_rng(0)
    months = 12
    steady = np.full(months, 0.010)                        # moderate but steady help
    erratic = np.tile([0.050, -0.020], months // 2)        # larger on average (0.015), unstable
    useless = np.zeros(months)
    harmful = np.full(months, -0.010)                      # removing it helps
    phi = np.stack([steady, erratic, useless, harmful], axis=1)          # [months, E]
    contrib = np.repeat(phi, 21, axis=0) + rng.normal(0, 1e-4, (months * 21, 4))
    r = reliability(contrib).r
    assert r[0] > r[1]                                     # the PG-GFE property: stability wins
    assert r[0] > r[2] > r[3]
    assert np.all((r > 0) & (r < 1))


def test_the_planted_edge_is_the_most_reliable_in_a_trained_graph():
    """Silver goes up tomorrow when gold rose 4 days ago: only the gold lag-5 edge carries it."""
    rng = np.random.default_rng(0)
    n = 1400
    X = pd.DataFrame(rng.normal(size=(n, len(COLS))), columns=COLS)
    g = X["gold__r1"].to_numpy()
    up = np.r_[np.full(4, np.nan), (g[:-4] > 0).astype(float)]
    y = {"up1": up, "up5": up, "vol1": np.zeros(n), "vol5": np.zeros(n)}
    params = {"hidden": 16, "layers": 1, "gnn_layers": 1, "dropout": 0.0, "lr": 3e-3,
              "weight_decay": 1e-5, "lam": 0.5}
    learner = NeuralLearner(net="static_gnn", window=5, max_epochs=150, patience=15, refit_all=False)
    train, held = np.arange(30, 1150), np.arange(1150, 1400)
    fitted = learner.fit_rows(X, train, {k: v[train] for k, v in y.items()}, params, seed=0, n_seeds=2)
    dev = torch.device("cpu")
    prep = prepare(X, held, learner._lookback(5), dev, fitted.scaler, fitted.missing_cols)
    tgt = _targets_for({k: v[held] for k, v in y.items()}, held, n, dev)
    rel = reliability(contributions(fitted.nets, prep, tgt, held))
    edges = fitted.nets[0].edges
    assert edges[int(np.argmax(rel.r))] == ("gold", 5)
    assert rel.phi[edges.index(("gold", 5))].mean() > 0.05  # removing it costs real loss


# ------------------------------------------------------------------ 4.3 edge-dropout supernet

def _graph(edge_dropout=(0.2, 1.0)):
    from xaglab.models.neural.nets import StaticGraphNet, column_groups

    torch.manual_seed(0)
    return StaticGraphNet(column_groups(COLS), hidden=8, layers=1, gnn_layers=1, dropout=0.0, window=5,
                          edge_dropout=edge_dropout)


def test_edge_dropout_samples_per_sample_masks_in_training_only():
    net = _graph()
    torch.manual_seed(1)
    m = net.sample_masks(20000, torch.device("cpu")).float()
    assert abs(m.mean().item() - 0.6) < 0.01                   # E[q] = (0.2 + 1) / 2
    per_sample = m.mean(1)
    assert per_sample.min() < 0.15 and per_sample.max() == 1.0  # sparse and dense masks alike
    x = torch.randn(4, 25, len(COLS))
    net.eval()
    with torch.no_grad():
        a, b = net(x)["logit1"], net(x, torch.ones(15, dtype=torch.bool))["logit1"]
    assert torch.equal(a, b)                                    # evaluation: all edges on
    net.train()
    with torch.no_grad():
        c, d = net(x)["logit1"], net(x)["logit1"]
    assert not torch.equal(c, d)                                # training: random masks


def test_a_fixed_mask_overrides_dropout_and_silences_its_edges():
    net = _graph()
    mask = torch.zeros(15, dtype=torch.bool)
    mask[[0, 7]] = True
    net.set_mask(mask)
    net.train()                                                 # even in training mode
    x = torch.randn(3, 25, len(COLS))
    with torch.no_grad():
        silver, sources = net.encode(x)
        a = net.propagate(silver, sources)["logit1"]
        changed = sources.clone()
        changed[:, 3] += 50.0                                   # an inactive edge's source
        b = net.propagate(silver, changed)["logit1"]
        c = net.propagate(silver, sources)["logit1"]
    assert torch.equal(a, b) and torch.equal(a, c)              # deterministic, edge 3 silent


def test_supernet_ranks_masks_like_separately_trained_networks():
    """D-14 stage 1: scoring a mask by a forward pass of one edge-dropout network must rank
    masks the way training a network per mask would (planted signal on the gold lag-5 edge)."""
    from scipy.stats import spearmanr

    from xaglab.models.neural import trainer
    from xaglab.models.neural.losses import joint_nll_rows
    from xaglab.models.neural.nets import StaticGraphNet, column_groups

    rng = np.random.default_rng(0)
    n = 1400
    X = pd.DataFrame(rng.normal(size=(n, len(COLS))), columns=COLS)
    g, c = X["gold__r1"].to_numpy(), X["copper__r1"].to_numpy()
    # silver: mostly gold 4 days ago, a little copper today
    score = np.r_[np.full(4, np.nan), 1.0 * g[:-4] + 0.4 * c[4:]]
    up = (score > 0).astype(float)
    up[:4] = np.nan
    rows, held = np.arange(30, 1150), np.arange(1150, 1400)
    dev = torch.device("cpu")
    prep = prepare(X, rows, 25, dev)
    y = {"up1": up, "up5": up, "vol1": np.zeros(n), "vol5": np.zeros(n)}
    tgt = _targets_for({k: v[rows] for k, v in y.items()}, rows, n, dev)
    tgt_held = _targets_for({k: v[held] for k, v in y.items()}, held, n, dev)
    fit_r, val_r = trainer.early_stop_split(rows)
    cfg = trainer.TrainConfig(lr=3e-3, weight_decay=1e-5, lam=0.5, max_epochs=150, patience=15)
    groups = column_groups(COLS)
    edges = [(d, lag) for d in ("gold", "copper", "dxy", "real_yield_10y", "vix") for lag in (1, 5, 21)]
    E = len(edges)

    def mask_of(on):
        m = torch.zeros(E, dtype=torch.bool)
        m[[edges.index(e) for e in on]] = True
        return m

    masks = [mask_of(edges), mask_of([("gold", 5), ("copper", 1)]), mask_of([("gold", 5)]),
             mask_of([("copper", 1)]), mask_of([("dxy", 1), ("vix", 21)]), mask_of([e for e in edges if e != ("gold", 5)])]

    def build(mask=None, drop=None):
        def f():
            net = StaticGraphNet(groups, 16, 1, 1, 0.0, "lstm", 5, edge_dropout=drop)
            if mask is not None:
                net.set_mask(mask)
            return net
        return f

    def held_loss(net, mask):
        with torch.no_grad():
            b = torch.as_tensor(held)
            out = net(prep.windows(b), mask)
            return float(joint_nll_rows(out, {k: v[b] for k, v in tgt_held.items()}).mean())

    supernet = trainer.train(build(drop=(0.2, 1.0)), prep, tgt, fit_r, val_r, cfg, seed=0).net
    screened = [held_loss(supernet, m) for m in masks]
    trained = [held_loss(trainer.train(build(mask=m), prep, tgt, fit_r, val_r, cfg, seed=0).net, None) for m in masks]
    rho = spearmanr(screened, trained).statistic
    assert rho >= 0.8, (rho, screened, trained)
    assert screened[5] > screened[2] and screened[3] > screened[1]  # dropping gold@5 hurts most


# ------------------------------------------------------------------ 4.4 evolutionary selection

def _toy_loss(target_on, weight=0.01):
    """Tail loss that falls by `weight` per target edge on and rises by `weight` per other edge on."""
    target = np.zeros(15, bool)
    target[list(target_on)] = True

    def loss(masks):
        masks = np.atleast_2d(masks)
        return 2.0 + weight * (masks & ~target).sum(1) - weight * (masks & target).sum(1)
    return loss, target


def test_fitness_and_clock_by_hand():
    from xaglab.models.aimdg.evolution import MarketClock, all_masks, fitness, mask_key

    m = np.array([[1, 0, 1, 0], [0, 0, 0, 0]], bool)
    f = fitness(np.array([1.0, 1.2]), 1.1, m, prev=np.array([1, 1, 0, 0], bool), lam2=0.4, lam3=0.2)
    assert np.allclose(f, [-0.1 + 0.4 * 2 / 4 + 0.2 * 2 / 4, 0.1 + 0.4 * 2 / 4 + 0.0])
    assert mask_key(np.array([1, 0, 1], bool)) == 5 and all_masks(3).shape == (8, 3)
    clock = MarketClock(T=4, eps=0.1)
    r = np.full(15, 0.5)
    assert np.isclose(clock.tick(r), 0.3)                      # first refit: explore
    rates = [clock.tick(r + 0.01) for _ in range(5)]           # stable reliabilities: settle
    assert np.allclose(rates, [0.25, 0.2, 0.15, 0.1, 0.1])
    assert np.isclose(clock.tick(r + 0.3), 0.3) and clock.tau == 0  # a shift: explore again


def test_the_clock_threshold_is_relative_to_recent_shifts():
    from xaglab.models.aimdg.evolution import MarketClock

    clock = MarketClock(T=6, eps=0.10, k=3.0, floor=0.01)
    base = np.full(15, 0.4)
    clock.tick(base)
    rng = np.random.default_rng(0)
    for _ in range(6):                                         # a stable regime: shifts ~0.004
        clock.tick(base + rng.normal(0, 0.005, 15))
    assert clock.tau == 6                                      # no shift registered
    assert np.isclose(clock.threshold(), max(3 * np.median(clock.history), 0.01))
    clock.tick(base + 0.057)                                   # the planted break's size of jump
    assert clock.tau == 0 and np.isclose(clock.rate(), 0.3)    # ... registers (0.10 would not)
    quiet = MarketClock(eps=0.10)
    quiet.tick(base)
    quiet.tick(base + 0.003)
    quiet.tick(base + 0.006)
    assert quiet.threshold() == 0.10 and quiet.tau == 2        # fewer than 3 shifts: fixed ε_c


def test_evolution_finds_the_optimum_and_matches_the_exhaustive_check():
    from xaglab.models.aimdg.evolution import evolve, exhaustive

    loss, target = _toy_loss([0, 4, 7, 12])
    sel = evolve(loss, r=np.full(15, 0.5), rate=0.2, seed=1)
    best, fbest = exhaustive(loss, 15)
    assert np.array_equal(sel.mask, target) and np.array_equal(best, target)
    assert np.isclose(sel.fitness, fbest)
    assert sel.evaluations < 2 ** 15                           # far fewer masks than exhaustive
    assert all(a >= b for a, b in zip(sel.history, sel.history[1:], strict=False))  # elitism
    assert len(sel.finalists) == 3 and np.array_equal(sel.finalists[0], sel.mask)


def test_warm_start_never_ends_worse_and_selection_is_reversible():
    from xaglab.models.aimdg.evolution import evolve, fitness

    loss, _ = _toy_loss([2, 3])
    prev = np.zeros(15, bool)
    prev[[3, 9]] = True                                        # edge 2 was off last month
    sel = evolve(loss, r=np.full(15, 0.3), prev=prev, lam2=0.001, generations=30, seed=2)
    f_prev = fitness(loss(prev[None]), float(loss(np.ones((1, 15), bool))[0]), prev[None], prev, 0.001, 0.0)[0]
    assert sel.fitness <= f_prev
    assert sel.mask[2] and not sel.mask[9]                     # edge 2 came back, edge 9 went


def test_stability_and_sparsity_weights_do_their_job():
    from xaglab.models.aimdg.evolution import evolve

    flat = lambda masks: np.full(len(np.atleast_2d(masks)), 2.0)  # edges don't matter
    prev = np.zeros(15, bool)
    prev[[1, 5, 11]] = True
    keep = evolve(flat, r=np.full(15, 0.5), prev=prev, lam2=0.01, generations=40, seed=3)
    assert np.array_equal(keep.mask, prev)                     # stability: no reason to move
    sparse = evolve(flat, r=np.full(15, 0.5), lam3=0.01, generations=40, seed=3)
    assert not sparse.mask.any()                               # sparsity: nothing earns its place


def test_mask_scorer_equals_the_deployed_ensemble_forecast():
    from xaglab.models.aimdg.evolution import MaskScorer
    from xaglab.models.neural import trainer
    from xaglab.models.neural.losses import qlike_rel

    rng = np.random.default_rng(4)
    n = 300
    X = pd.DataFrame(rng.normal(size=(n, len(COLS))), columns=COLS)
    up = (rng.uniform(size=n) < 0.5).astype(float)
    y = {"up1": up, "up5": up, "vol1": rng.normal(0, 0.3, n), "vol5": rng.normal(0, 0.3, n)}
    rows = np.arange(40, 300)
    dev = torch.device("cpu")
    prep = prepare(X, rows, 25, dev)
    tgt = _targets_for({k: v[rows] for k, v in y.items()}, rows, n, dev)
    nets = [_graph(None) for _ in range(3)]
    for k, net in enumerate(nets):
        torch.manual_seed(k)
        for p in net.parameters():
            torch.nn.init.normal_(p, 0, 0.1)
    scorer = MaskScorer(nets, prep, tgt, rows, budget=10_000)  # a tiny budget forces chunking
    masks = np.array([np.ones(15, bool), np.eye(15, dtype=bool)[3], np.zeros(15, bool)])
    got = scorer(masks)
    for i, m in enumerate(masks):
        for net in nets:
            net.set_mask(torch.as_tensor(m))
        pred = trainer.predict(nets, prep, rows)               # the deployed ensemble forecast
        want = 0.0
        for h in (1, 5):
            p = np.clip(pred[f"up{h}"], 1e-6, 1 - 1e-6)
            yy = up[rows]
            want = want - (yy * np.log(p) + (1 - yy) * np.log(1 - p))
            r = torch.as_tensor(np.exp(y[f"vol{h}"][rows]), dtype=torch.float64)
            want = want + 0.5 * qlike_rel(torch.as_tensor(pred[f"vol{h}"]), r).numpy()
        assert np.isclose(got[i], want.mean(), atol=1e-4), (i, got[i], want.mean())
    for net in nets:
        net.set_mask(None)
    assert np.allclose(scorer(masks), got) and scorer.evaluations == 3   # cached, not re-scored


# ------------------------------------------------------------------ 4.5 finalists

def test_finalists_keep_their_mask_never_worsen_the_tail_and_the_best_wins():
    from xaglab.models.aimdg.evolution import MaskScorer
    from xaglab.models.aimdg.finalists import choose, fine_tune
    from xaglab.models.neural import trainer
    from xaglab.models.neural.nets import StaticGraphNet, column_groups

    rng = np.random.default_rng(0)
    n = 900
    X = pd.DataFrame(rng.normal(size=(n, len(COLS))), columns=COLS)
    g = X["gold__r1"].to_numpy()
    up = np.r_[np.full(4, np.nan), (g[:-4] > 0).astype(float)]
    rows = np.arange(30, 900)
    dev = torch.device("cpu")
    prep = prepare(X, rows, 25, dev)
    y = {"up1": up, "up5": up, "vol1": np.zeros(n), "vol5": np.zeros(n)}
    tgt = _targets_for({k: v[rows] for k, v in y.items()}, rows, n, dev)
    fit_r, tail = trainer.early_stop_split(rows)
    cfg = trainer.TrainConfig(lr=3e-3, weight_decay=1e-5, lam=0.5, max_epochs=40, patience=8)
    groups = column_groups(COLS)
    supernets = [trainer.train(lambda: StaticGraphNet(groups, 16, 1, 1, 0.0, "lstm", 5, edge_dropout=(0.2, 1.0)),
                               prep, tgt, fit_r, tail, cfg, seed=k).net for k in range(2)]
    good = np.zeros(15, bool)
    good[1] = True                                             # gold@5 only
    bad = np.zeros(15, bool)
    bad[4] = True                                              # copper@5 only: no signal
    tuned = fine_tune(supernets, good, prep, tgt, fit_r, tail, cfg)
    assert all(torch.equal(t.fixed_mask, torch.as_tensor(good)) for t in tuned)
    before = MaskScorer(supernets, prep, tgt, tail)(good[None])[0]
    after = MaskScorer(tuned, prep, tgt, tail)(good[None])[0]
    assert after <= before + 0.02                              # joint loss on the tail: no real harm
    res = choose([bad, good], supernets, prep, tgt, fit_r, tail, cfg)
    assert res.winner == 1 and np.array_equal(res.mask, good) and res.tail_losses[1] < res.tail_losses[0]


# ------------------------------------------------------------------ 4.6 the AIM-DG forecaster

def test_aimdg_two_real_refits_warm_start_clock_and_attribution(features):
    import json

    from xaglab.eval.folds import make_calendar
    from xaglab.eval.harness import _check
    from xaglab.models.aimdg.forecaster import AimDG
    from xaglab.models.base import FitView, PredictView

    cal = make_calendar(features.index, features.labels, train_start=features.warmup_end())
    r0, r1 = cal.all_refits(None)[:2]
    m = AimDG(window=10, windows=(10,), n_seeds=2, tune_seeds=1, n_trials=1, max_epochs=2, patience=2,
              pop_size=10, generations=5, exhaustive_check=False)
    assert m.name == "aimdg" and m.stateless is False
    tuned = m.tune(FitView.build(features, r0, 0))
    params = tuned["joint"]["params"]
    assert 0.0 <= params["lam2"] <= 0.005 and 0.0 <= params["lam3"] <= 0.005
    m.set_tuned(tuned)

    diags = []
    for refit in (r0, r1):
        m.fit(FitView.build(features, refit, 0))
        view = PredictView.build(features, refit)
        pred = _check(m.predict(view), view.dates, m.name)
        assert pred.notna().all().all()
        d = m.diagnostics()
        json.dumps(d)                                          # saved with the run
        diags.append(d)
        sel, attr = d["selection"], d["attribution"]
        assert sorted(attr) == [str(x.date()) for x in view.dates]
        for day in attr.values():
            assert set(day) == set(sel["active"])              # exactly the active edges
            assert all(len(v) == 2 and np.all(np.isfinite(v)) for v in day.values())  # 1 and 5 days
    assert diags[0]["selection"]["warm_start"] is False and diags[1]["selection"]["warm_start"] is True
    assert diags[0]["selection"]["clock"]["shift"] is None     # the first tick: nothing to compare
    assert diags[1]["selection"]["clock"]["shift"] is not None
    assert np.array_equal(m.prev_mask, m.fitted.extra["mask"])


# ------------------------------------------------------------------ 4.7 planted-break market

def test_planted_break_market_follows_the_planted_rule():
    from xaglab.models.aimdg.synthetic import monthly_refits, planted_break_market

    fs = planted_break_market(n=1300, brk=900, noise=0.0, seed=1)
    up = fs.labels["up1"].to_numpy()
    gold, copper = fs.nodes["gold"]["r1"].to_numpy(), fs.nodes["copper"]["r1"].to_numpy()
    t = np.arange(4, 900)
    assert np.array_equal(up[t], (gold[t - 4] > 0).astype(float))      # before: gold four days ago
    t = np.arange(900, 1300)
    assert np.array_equal(up[t], (copper[t] > 0).astype(float))        # after: copper today
    assert set(fs.pooled().columns) >= {"silver__r1", "gold__r1", "copper__r1", "vix__r5"}
    r = monthly_refits(first_origin=816, n_refits=3)
    assert [x.origin for x in r] == [816, 837, 858]
    assert all(x.train_end == x.origin - 5 and x.predict_end == x.origin + 20 for x in r)
