"""Phase 5 selection lab (D-38): ablations A1–A3 and the F1 sensitivity sweep on fixed
networks, at all 17 tuning points, development data only.

Per point: the 5 supernets are trained once per development block with the network
settings the AIM-DG development run chose there (cached under experiments/lab/phase5/),
then every setting in `lab.ALL_SETTINGS` is run through the same networks and scored
out of fold. Results: experiments/lab/phase5/point??.json (one per point, resumable) and
the tables in experiments/reports/phase5-lab/.

Run:  python scripts/phase5_lab.py [--jobs 6] [--points 0,1,...] [--dev aimdg-dev-20260928-090733]
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd
from joblib import Parallel, delayed, parallel_config

from xaglab.cli import _data
from xaglab.eval.harness import load_dev
from xaglab.models.aimdg import lab
from xaglab.paths import EXPERIMENTS_DIR

LAB_DIR = EXPERIMENTS_DIR / "lab" / "phase5"
REPORT_DIR = EXPERIMENTS_DIR / "reports" / "phase5-lab"


def _one(point: int, params: dict) -> dict:
    fs, cal = _data()
    out = LAB_DIR / f"point{point:02d}.json"
    if out.exists():
        return json.loads(out.read_text())
    res = lab.evaluate_point(fs, cal, point, params, cache_dir=LAB_DIR)
    out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--points", default="")
    ap.add_argument("--dev", default="aimdg-dev-20260928-090733")
    args = ap.parse_args()
    tuning = load_dev(args.dev).tuning
    points = [int(p) for p in args.points.split(",")] if args.points else list(range(len(tuning)))
    params = {k: tuning[k]["results"]["joint"]["params"] for k in points}
    LAB_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=args.jobs)(delayed(_one)(k, params[k]) for k in points)
    results = sorted(results, key=lambda r: r["point"])
    table = lab.summarise(results)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORT_DIR / "lab_summary.csv", index=False)
    per_point = pd.DataFrame([{"point": r["point"], "origin": r["origin"], "setting": n, "cv_loss": s["cv_loss"],
                               **{f"c_{k}": v for k, v in s["components"].items()}, "n_edges": s["n_edges"]}
                              for r in results for n, s in r["settings"].items()])
    per_point.to_csv(REPORT_DIR / "lab_per_point.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"{len(results)} points, {time.perf_counter() - t0:.0f} s wall; "
          f"training {sum(r['train_seconds'] for r in results):.0f} s in total")
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
