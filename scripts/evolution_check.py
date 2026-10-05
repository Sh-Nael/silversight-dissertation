"""Evolutionary edge selection on real development data (step 4.4, D-34): at the last tuning
point, on the held-out tail of the training span only (no test row).

Trains the 5-network edge-dropout supernet, scores reliability, runs the DEAP search with the
paper's settings (population 50, 100 generations, first-refit mutation rate 0.3), and checks it
against the exhaustive optimum over all 2^15 masks, for a few sparsity weights λ3.

Run:  python scripts/evolution_check.py   (GPU: ~2 min)
"""

from __future__ import annotations

import time

import numpy as np

from xaglab.cli import _data
from xaglab.eval.harness import load_dev
from xaglab.models.aimdg.evolution import MarketClock, MaskScorer, evolve, exhaustive
from xaglab.models.aimdg.reliability import contributions, reliability
from xaglab.models.base import FitView
from xaglab.models.neural import trainer
from xaglab.models.neural.data import prepare, rows_array
from xaglab.models.neural.forecaster import joint_task
from xaglab.models.neural.learner import NeuralLearner, _targets_for


def main() -> None:
    fs, cal = _data()
    view = FitView.build(fs, cal.all_refits(None)[-1], 0)
    params = load_dev("static_gnn-dev-20260927-164105").tuning[-1]["results"]["joint"]["params"]
    X, task = view.data.pooled(), joint_task(view.data)
    rows, y = rows_array(view.train), task.target(view.train)
    learner = NeuralLearner(net="supernet", refit_all=False)
    fitted = learner.fit_rows(X, view.train, y, params, seed=0, n_seeds=5)
    _, tail = trainer.early_stop_split(rows)
    dev = trainer.device()
    prep = prepare(X, tail, learner._lookback(fitted.window), dev, fitted.scaler, fitted.missing_cols)
    tgt = _targets_for(y, rows, len(X), dev)
    rel = reliability(contributions(fitted.nets, prep, tgt, tail))
    edges = fitted.nets[0].edges
    rate = MarketClock().tick(rel.r)
    for lam3 in (0.0, 0.001, 0.003):
        scorer = MaskScorer(fitted.nets, prep, tgt, tail)
        sel = evolve(scorer, rel.r, rate=rate, lam3=lam3, seed=0)
        t = time.perf_counter()
        best, fbest = exhaustive(scorer, len(edges), lam3=lam3)
        t_ex = time.perf_counter() - t
        on = [f"{d}@{lag}" for (d, lag), b in zip(edges, sel.mask, strict=True) if b]
        print(f"λ3={lam3}: search {sel.seconds:.1f}s, {sel.evaluations} masks; exhaustive {t_ex:.1f}s, 32768 masks")
        print(f"   search winner  F={sel.fitness:+.5f}  ΔL={sel.delta_loss * 1000:+.2f} mnats  {len(on)} edges: {on}")
        print(f"   exhaustive opt F={fbest:+.5f}  same mask: {np.array_equal(best, sel.mask)}  "
              f"gap {1000 * (sel.fitness - fbest):.3f} mnats")


if __name__ == "__main__":
    main()
