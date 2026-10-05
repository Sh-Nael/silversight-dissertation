"""Phase 5 test-fold ablations A1 and A2 (D-38): AIM-DG's 200 test refits, networks trained
once per refit (in parallel, cached under experiments/lab/phase5-test/), then the monthly
pipeline replayed through the same networks under three settings, each saved as a run:

  aimdg_replay   the approved settings (a like-for-like reference for the ablations; it should
                 sit close to the official run aimdg-20260928-120236)
  aimdg_A1       no evolution: the top-3 edges by reliability, every month
  aimdg_A2       no variance penalty: β = 0 in the reliability score

These are the two ablation test runs pre-declared in Ch3 and the Phase 5 note; no other
setting is evaluated on the test folds.

Run:  python scripts/phase5_test_bundles.py --train [--jobs 6]     (≈ 25 min)
      python scripts/phase5_test_bundles.py --replay                (≈ 1 h; runs saved)
"""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime

import pandas as pd
from joblib import Parallel, delayed, parallel_config

from xaglab.cli import _data
from xaglab.eval.harness import (
    LABEL_COLUMNS,
    RunResult,
    _git_state,
    _versions,
    load_dev,
    load_run,
    save_run,
)
from xaglab.models.aimdg import lab
from xaglab.models.aimdg.replay import Sequential, replay, train_bundle
from xaglab.models.tuning import tuning_points
from xaglab.paths import EXPERIMENTS_DIR

TEST_DIR = EXPERIMENTS_DIR / "lab" / "phase5-test"
OFFICIAL = "aimdg-20260928-120236"
DEV = "aimdg-dev-20260928-090733"

SETTINGS = (
    Sequential("aimdg_replay"),
    Sequential("aimdg_A1", base=lab.Setting("A1_no_evolution_top3", mode="topk", k=3)),
    Sequential("aimdg_A2", base=lab.Setting("A2_no_variance_penalty", beta=0.0)),
)


def refits_and_params():
    fs, cal = _data()
    refits = cal.all_refits(None)
    idx = tuning_points([r.origin for r in refits], 252)
    points = sorted(set(idx))
    tuning = load_dev(DEV).tuning
    params = [tuning[points.index(i)]["results"]["joint"]["params"] for i in idx]
    return fs, cal, refits, params


def _train(i: int, refit, params: dict) -> float:
    fs, _ = _data()
    return train_bundle(fs, refit, params, seed=0, path=TEST_DIR / f"refit{i:03d}").seconds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--settings", default="", help="comma-separated subset of aimdg_replay,aimdg_A1,aimdg_A2")
    args = ap.parse_args()
    fs, _cal, refits, params = refits_and_params()
    if args.train:
        t0 = time.perf_counter()
        with parallel_config(backend="loky", inner_max_num_threads=1):
            secs = Parallel(n_jobs=args.jobs)(delayed(_train)(i, r, p) for i, (r, p) in enumerate(zip(refits, params, strict=True)))
        print(f"{len(refits)} bundles in {time.perf_counter() - t0:.0f} s wall (training {sum(secs):.0f} s)")
    if args.replay:
        official = load_run(OFFICIAL)
        # loaders, so only one refit's networks sit on the GPU at a time
        bundles = [(lambda i=i, r=r, p=p: train_bundle(fs, r, p, seed=0, path=TEST_DIR / f"refit{i:03d}"))
                   for i, (r, p) in enumerate(zip(refits, params, strict=True))]
        wanted = set(args.settings.split(",")) if args.settings else {s.name for s in SETTINGS}
        for seq in (s for s in SETTINGS if s.name in wanted):
            t0 = time.perf_counter()
            pred, diags = replay(fs, bundles, seq, seed=0)
            labels = fs.labels.loc[pred.index, list(LABEL_COLUMNS)]
            fold_refit = official.predictions.loc[pred.index, ["fold", "refit"]]
            frame = pd.concat([pred, fold_refit, labels], axis=1)
            started = datetime.now(UTC)
            run_id = f"{seq.name}-{started.strftime('%Y%m%d-%H%M%S')}"
            manifest = {**official.manifest, "run_id": run_id, "model": seq.name,
                        "model_class": "xaglab.models.aimdg.replay.replay",
                        "model_kwargs": {"replay_of": OFFICIAL, "setting": seq.base.name, "warm_start": seq.warm_start},
                        "model_config": {**official.manifest["model_config"], "replay": seq.name},
                        "started_utc": started.isoformat(timespec="seconds"),
                        "wall_seconds": round(time.perf_counter() - t0, 2), "tuning": None,
                        "fit_seconds_total": 0.0, "code": _git_state(), "env": _versions()}
            res = RunResult(run_id, frame, official.timings.iloc[:0], [{"selection": d} for d in diags], manifest)
            path = save_run(res)
            print(f"{run_id} -> {path}  ({time.perf_counter() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
