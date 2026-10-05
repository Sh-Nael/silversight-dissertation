"""Seed noise versus setting differences in a neural model's development score (D-29).

At the last tuning point (the largest training span), 8 random settings from the
NeuralLearner search space are each trained with 4 seeds (single networks), and once
as a 4-network ensemble, on the 3 purged validation blocks. If single-network scores
vary between seeds about as much as between settings, a search that scores each trial
with one network is mostly choosing seeds.

Run:  python scripts/seed_noise.py [--cell lstm] [--settings 8] [--seeds 4]   (GPU: ~2 min)
Result on 27 Sep 2026 (lstm): between settings sd 0.0146, between seeds sd 0.0135,
ensembles sd 0.0103; rank correlation single-seed means vs ensembles 0.64.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import optuna

from xaglab.cli import _data
from xaglab.models.base import FitView
from xaglab.models.neural.forecaster import TunedNeural, joint_task
from xaglab.models.neural.learner import NeuralLearner
from xaglab.models.tuning import cv_predict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="lstm")
    ap.add_argument("--settings", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()

    fs, cal = _data()
    view = FitView.build(fs, cal.all_refits(None)[-1], 0)
    X, task, blocks = view.data.pooled(), joint_task(view.data), TunedNeural(args.cell)._blocks(view)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=3))
    single = NeuralLearner(cell=args.cell, n_seeds=1)
    ensemble = NeuralLearner(cell=args.cell, n_seeds=args.seeds)
    t0 = time.perf_counter()
    means, sds, ens = [], [], []
    for c in range(args.settings):
        trial = study.ask()
        params = single.space(trial)
        study.tell(trial, 0.0)
        scores = [task.score(*cv_predict(single, task, X, blocks, params, seed=100 + s)) for s in range(args.seeds)]
        ens.append(task.score(*cv_predict(ensemble, task, X, blocks, params, seed=100)))
        means.append(np.mean(scores))
        sds.append(np.std(scores, ddof=1))
        print(f"setting {c}: {params}\n  single mean {means[-1]:.4f} sd {sds[-1]:.4f}  ensemble {ens[-1]:.4f}",
              flush=True)
    rank = lambda v: np.argsort(np.argsort(v))
    print(f"\nbetween settings sd {np.std(means, ddof=1):.4f} | between seeds sd "
          f"{np.sqrt(np.mean(np.square(sds))):.4f} | ensembles sd {np.std(ens, ddof=1):.4f}")
    print(f"rank correlation single-seed means vs ensembles {np.corrcoef(rank(means), rank(ens))[0, 1]:.2f}")
    print(f"{time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
