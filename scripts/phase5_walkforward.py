"""Phase 5 development walk-forward (D-38): the sequential settings of AIM-DG measured on
development data. One year of refits **inside the last tuning point's training span**
(origins from train_end − 252 to train_end, nothing after the training data), the networks
of every refit trained once and cached, and each sequential setting replayed through them:

  default            warm start, λ2 tuned, clock k = 3, T = 6, re-selection every 21 days
  no_warm_start      memoryless selection (no previous winner, exploring rate)
  clock_k_1.5/6      the clock's relative threshold
  clock_T_3/12       how fast the mutation rate settles
  lam2_0/0.02        the stability weight
  cadence_42/63      re-selection every 42 / 63 days (their own refits and bundles)

Scored by the mean joint loss of the calibrated forecasts over the year's 252 forecast days
(the same days for every cadence). Bundles: experiments/lab/phase5-wf/. Tables:
experiments/reports/phase5-walkforward/.

Run:  python scripts/phase5_walkforward.py [--jobs 6] [--point 16]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from itertools import pairwise

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from xaglab.cli import _data
from xaglab.eval.folds import Refit
from xaglab.eval.harness import load_dev
from xaglab.models.aimdg import lab
from xaglab.models.aimdg.replay import DEFAULT_SEQ, Sequential, joint_loss, replay, train_bundle
from xaglab.paths import EXPERIMENTS_DIR

WF_DIR = EXPERIMENTS_DIR / "lab" / "phase5-wf"
REPORT_DIR = EXPERIMENTS_DIR / "reports" / "phase5-walkforward"
YEAR = 252
PURGE = 5


def year_refits(base: Refit, step: int) -> list[Refit]:
    """Monthly (or 42/63-day) refits covering the last year of the training span."""
    start = base.train_end - YEAR
    out, j = [], 0
    while start + step * j + step - 1 <= base.train_end:
        o = start + step * j
        out.append(Refit(fold=base.fold, j=j, origin=o, train_start=base.train_start, train_end=o - PURGE,
                         val_start=o - PURGE - YEAR, inner_end=o - PURGE - YEAR - 1, predict_end=o + step - 1))
        j += 1
    return out


def _train(step: int, j: int, refit: Refit, params: dict) -> float:
    fs, _ = _data()
    b = train_bundle(fs, refit, params, seed=0, path=WF_DIR / f"step{step}" / f"refit{j:02d}")
    return b.seconds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--point", type=int, default=16)
    ap.add_argument("--dev", default="aimdg-dev-20260928-090733")
    args = ap.parse_args()
    fs, cal = _data()
    params = load_dev(args.dev).tuning[args.point]["results"]["joint"]["params"]
    base = lab.tuning_refit(fs, cal, args.point)
    refits = {step: year_refits(base, step) for step in (21, 42, 63)}
    t0 = time.perf_counter()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        secs = Parallel(n_jobs=args.jobs)(delayed(_train)(step, j, r, params)
                                          for step, rs in refits.items() for j, r in enumerate(rs))
    print(f"bundles trained/loaded: {sum(len(v) for v in refits.values())} in {time.perf_counter() - t0:.0f} s "
          f"(training {sum(secs):.0f} s)", flush=True)
    bundles = {step: [train_bundle(fs, r, params, seed=0, path=WF_DIR / f"step{step}" / f"refit{j:02d}")
                      for j, r in enumerate(rs)] for step, rs in refits.items()}  # 22 bundles: fits on the GPU
    settings = [
        (21, DEFAULT_SEQ),
        (21, Sequential("no_warm_start", warm_start=False)),
        (21, Sequential("clock_k_1.5", clock_k=1.5)), (21, Sequential("clock_k_6", clock_k=6.0)),
        (21, Sequential("clock_T_3", clock_T=3)), (21, Sequential("clock_T_12", clock_T=12)),
        (21, Sequential("lam2_0", lam2=0.0)), (21, Sequential("lam2_0.02", lam2=0.02)),
        (21, replace(DEFAULT_SEQ, name="all_edges_on", base=lab.Setting("all_on", mode="all_on"))),
        (42, replace(DEFAULT_SEQ, name="cadence_42")), (63, replace(DEFAULT_SEQ, name="cadence_63")),
    ]
    rows, diag_all = [], {}
    for step, seq in settings:
        t1 = time.perf_counter()
        pred, diags = replay(fs, bundles[step], seq)
        loss = joint_loss(pred, fs.labels)
        edges = [d["n_edges"] for d in diags]
        changed = [len(set(a["active"]) ^ set(b["active"])) for a, b in pairwise(diags)]
        rows.append({"setting": seq.name, "cadence": step, "refits": len(diags), "joint_loss": float(loss.mean()),
                     "n_days": len(loss), "edges": float(np.mean(edges)),
                     "edges_changed_per_refit": float(np.mean(changed)) if changed else 0.0,
                     "clock_resets": int(sum(d["tau"] == 0 for d in diags[1:])),
                     "seconds": round(time.perf_counter() - t1, 1)})
        diag_all[seq.name] = diags
        print(json.dumps(rows[-1]), flush=True)
    table = pd.DataFrame(rows)
    table["diff_vs_default"] = table["joint_loss"] - table.loc[table.setting == "default", "joint_loss"].iloc[0]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORT_DIR / "walkforward.csv", index=False)
    (REPORT_DIR / "selections.json").write_text(json.dumps(diag_all, indent=1, default=float) + "\n")
    pd.set_option("display.width", 200)
    print(f"\nyear {fs.index[refits[21][0].origin].date()} → {fs.index[refits[21][-1].predict_end].date()} "
          f"(training span of tuning point {args.point}); {time.perf_counter() - t0:.0f} s in total")
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
