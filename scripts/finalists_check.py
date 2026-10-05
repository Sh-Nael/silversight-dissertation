"""Does the finalist stage (D-14 stage 2) earn its cost? Step 4.5, on development data only.

At 3 tuning points (≈2014, 2020, 2026) × their 3 purged validation blocks (the D-26 development
blocks, all inside the training span), the whole AIM-DG selection runs on each block's
training rows (5 edge-dropout supernets, reliability on the held-out tail, evolution with the
paper's settings), and four forecasts are scored on the block's unseen validation rows with
the development score (joint NLL, D-28):

  screened   supernets with the evolution's winning mask (no fine-tuning)
  finetuned  the finalist stage: top-3 masks fine-tuned, best on the tail wins (D-14)
  all_on     supernets with every edge on
  static     static_gnn (no edge dropout, all edges on): the baseline AIM-DG must beat

Rule fixed before running (28 Sep): keep the finalist stage only if
`finetuned` beats `screened` in at least 6 of the 9 blocks; otherwise deploy the screened
winner directly. Networks here are trained on the fit rows only (no all-rows refit), the
same for every variant, so the comparison is like for like.

Run:  python scripts/finalists_check.py   (GPU: ~15–25 min)
"""

from __future__ import annotations

import json
import time

import numpy as np

from xaglab.cli import _data
from xaglab.eval.harness import load_dev
from xaglab.models.aimdg.evolution import MarketClock, MaskScorer, evolve
from xaglab.models.aimdg.finalists import choose, masked_copy
from xaglab.models.aimdg.reliability import contributions, reliability
from xaglab.models.base import FitView
from xaglab.models.neural import trainer
from xaglab.models.neural.data import prepare, rows_array
from xaglab.models.neural.forecaster import joint_task
from xaglab.models.neural.learner import NeuralLearner, _targets_for
from xaglab.models.tuning import cv_blocks, tuning_points
from xaglab.paths import EXPERIMENTS_DIR

POINTS = (4, 10, 16)   # tuning points ≈ 2014, 2020, 2026


def main() -> None:
    fs, cal = _data()
    refits = cal.all_refits(None)
    idx = sorted(set(tuning_points([r.origin for r in refits], 252)))
    tuning = load_dev("static_gnn-dev-20260927-164105").tuning
    dev = trainer.device()
    rows_out = []
    for k in POINTS:
        refit = refits[idx[k]]
        params = tuning[k]["results"]["joint"]["params"]
        view = FitView.build(fs, refit, 0)
        X, task = view.data.pooled(), joint_task(view.data)
        cfg = trainer.TrainConfig(lr=params["lr"], weight_decay=params["weight_decay"], lam=params["lam"])
        blocks = cv_blocks(refit.train_start, refit.train_end, purge=refit.origin - refit.train_end)
        for b_i, b in enumerate(blocks):
            t0 = time.perf_counter()
            y = task.target(b.train)
            rows = rows_array(b.train)
            val = rows_array(b.val)
            sup = NeuralLearner(net="supernet", refit_all=False)
            fit = sup.fit_rows(X, b.train, y, params, seed=0, n_seeds=5)
            stat = NeuralLearner(net="static_gnn", refit_all=False).fit_rows(X, b.train, y, params, seed=0, n_seeds=5)
            prep = prepare(X, rows, sup._lookback(fit.window), dev, fit.scaler, fit.missing_cols)
            tgt = _targets_for(y, rows, len(X), dev)
            fit_r, tail = trainer.early_stop_split(rows)
            rel = reliability(contributions(fit.nets, prep, tgt, tail))
            sel = evolve(MaskScorer(fit.nets, prep, tgt, tail), rel.r, rate=MarketClock().tick(rel.r), seed=0)
            fin = choose(sel.finalists, fit.nets, prep, tgt, fit_r, tail, cfg)
            variants = {
                "screened": [masked_copy(n, sel.mask) for n in fit.nets],
                "finetuned": fin.nets,
                "all_on": fit.nets,
            }
            score = {name: task.score(trainer.predict(nets, prep, val), val) for name, nets in variants.items()}
            stat_prep = prepare(X, rows, sup._lookback(stat.window), dev, stat.scaler, stat.missing_cols)
            score["static"] = task.score(trainer.predict(stat.nets, stat_prep, val), val)
            edges = fit.nets[0].edges
            row = {"point": k, "origin": str(fs.index[refit.origin].date()), "block": b_i, **score,
                   "n_edges": int(sel.mask.sum()), "finalist_won": fin.winner,
                   "edges": [f"{d}@{lag}" for (d, lag), on in zip(edges, sel.mask, strict=True) if on],
                   "seconds": round(time.perf_counter() - t0, 1)}
            rows_out.append(row)
            print(json.dumps({k2: (round(v, 5) if isinstance(v, float) else v) for k2, v in row.items()}), flush=True)

    wins = sum(r["finetuned"] < r["screened"] for r in rows_out)
    print(f"\nfinetuned beats screened in {wins} of {len(rows_out)} blocks "
          f"(rule: keep the finalist stage if >= 6)")
    for a, b in (("screened", "all_on"), ("screened", "static"), ("all_on", "static"), ("finetuned", "static")):
        d = [r[a] - r[b] for r in rows_out]
        print(f"{a} vs {b}: mean diff {np.mean(d) * 1000:+.2f} mnats, better in {sum(x < 0 for x in d)}/{len(d)}")
    out = EXPERIMENTS_DIR / "reports" / "finalists-check-4.5"
    out.mkdir(parents=True, exist_ok=True)
    (out / "blocks.json").write_text(json.dumps(rows_out, indent=1) + "\n")


if __name__ == "__main__":
    main()
