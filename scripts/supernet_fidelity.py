"""Supernet checks on real development data (step 4.3, D-34): at the last tuning point, on the
held-out tail of the training span only (no test row).

  (a) does edge dropout cost accuracy with all edges on?  (static_gnn vs supernet, 5 networks)
  (c) reliability from the static network vs from the supernet
  (b) fidelity: 8 masks scored by one supernet forward pass vs networks trained with each mask
      fixed (two independent 5-network sets A and B, so the noise of the "truth" is measured too)

Run:  python scripts/supernet_fidelity.py   (GPU: ~4 min)
Result on 28 Sep 2026: (a) all-on tail NLL static 2.4716, supernet 2.4677; (c) rank corr 0.82;
(b) mask spread 0.011 vs seed noise |A-B| 0.013; rank corr A vs B 0.45; screened vs mean(A, B) 0.64.
"""
import time

import numpy as np
import torch
from scipy.stats import spearmanr

from xaglab.cli import _data
from xaglab.eval.harness import load_dev
from xaglab.models.aimdg.reliability import contributions, reliability
from xaglab.models.base import FitView
from xaglab.models.neural import trainer
from xaglab.models.neural.data import prepare, rows_array
from xaglab.models.neural.forecaster import joint_task
from xaglab.models.neural.learner import NeuralLearner, _targets_for
from xaglab.models.neural.losses import joint_nll_rows

fs, cal = _data()
view = FitView.build(fs, cal.all_refits(None)[-1], 0)
params = load_dev("static_gnn-dev-20260927-164105").tuning[-1]["results"]["joint"]["params"]
X, task = view.data.pooled(), joint_task(view.data)
rows = rows_array(view.train)
y = task.target(view.train)
dev = trainer.device()
_, tail = trainer.early_stop_split(rows)
nets = {}
for kind in ("static_gnn", "supernet"):
    L = NeuralLearner(net=kind, refit_all=False)
    t = time.perf_counter()
    f = L.fit_rows(X, view.train, y, params, seed=0, n_seeds=5)
    nets[kind] = (L, f)
    print(kind, f"{time.perf_counter()-t:.1f}s", "epochs", [i["best_epoch"] for i in f.info])
L, f = nets["supernet"]
prep = prepare(X, tail, L._lookback(f.window), dev, f.scaler, f.missing_cols)
tgt = _targets_for(y, rows, len(X), dev)
edges = f.nets[0].edges

def tail_loss(net_list, mask):
    b = torch.as_tensor(tail, device=dev)
    with torch.no_grad():
        return float(np.mean([joint_nll_rows(n(prep.windows(b), mask), {k: v[b] for k, v in tgt.items()}).mean().item() for n in net_list]))

allon = torch.ones(15, dtype=torch.bool, device=dev)
print("(a) all-on tail joint NLL: static", round(tail_loss(nets["static_gnn"][1].nets, allon), 4), " supernet", round(tail_loss(f.nets, allon), 4))
rel_s = reliability(contributions(nets["static_gnn"][1].nets, prep, tgt, tail))
rel_u = reliability(contributions(f.nets, prep, tgt, tail))
print("(c) reliability static vs supernet, rank corr", round(spearmanr(rel_s.r, rel_u.r).statistic, 2))
for e, a, b, p in sorted(zip(edges, rel_s.r, rel_u.r, rel_u.phi.mean(1)), key=lambda t: -t[2]):
    print(f"   {e[0]:>15}@{e[1]:<2} static {a:.3f}  supernet {b:.3f}  contribution {p*1000:+.2f} mnats")
# (b) fidelity: masks screened by the supernet vs networks trained with that fixed mask (2 seeds)
rng = np.random.default_rng(0)
order = np.argsort(-rel_u.r)
cands = [allon.clone()]
for k in (3, 6, 10):
    m = torch.zeros(15, dtype=torch.bool, device=dev); m[torch.as_tensor(order[:k].copy())] = True; cands.append(m)
for _ in range(4):
    cands.append(torch.as_tensor(rng.uniform(size=15) < 0.5, device=dev))
screened = [tail_loss(f.nets, m) for m in cands]
fit_r, val_r = trainer.early_stop_split(rows)
from xaglab.models.neural.nets import StaticGraphNet, column_groups

full_prep = prepare(X, rows, L._lookback(f.window), dev)
groups = column_groups(full_prep.columns)
cfg = trainer.TrainConfig(lr=params["lr"], weight_decay=params["weight_decay"], lam=params["lam"])
def trained_losses(seeds):
    out = []
    for m in cands:
        ns = []
        for k in seeds:
            def build(m=m):
                net = StaticGraphNet(groups, params["hidden"], params["layers"], params["gnn_layers"], params["dropout"], "lstm", f.window)
                net.set_mask(m); return net
            ns.append(trainer.train(build, full_prep, tgt, fit_r, val_r, cfg, seed=k).net)
        out.append(tail_loss(ns, None))
    return out
A, B = trained_losses(range(5)), trained_losses(range(5, 10))
print("(b) masks: n_on", [int(m.sum()) for m in cands])
print("    screened     ", [round(v, 4) for v in screened])
print("    trained A    ", [round(v, 4) for v in A])
print("    trained B    ", [round(v, 4) for v in B])
print("    rank corr A vs B (is the truth itself stable?)", round(spearmanr(A, B).statistic, 2))
print("    rank corr screened vs mean(A,B)", round(spearmanr(screened, np.add(A, B) / 2).statistic, 2))
print("    spread of masks (max-min) trained", round(max(np.add(A, B) / 2) - min(np.add(A, B) / 2), 4),
      " |A-B| mean", round(float(np.mean(np.abs(np.subtract(A, B)))), 4))
