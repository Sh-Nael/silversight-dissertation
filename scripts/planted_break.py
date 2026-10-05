"""Planted regime break (step 4.7, Ch3 Table 3.1 integration test): does AIM-DG re-select?

A synthetic market (``aimdg/synthetic.py``): silver follows gold@5 until day 900, then
copper@1. The real ``AimDG`` forecaster and ``static_gnn`` are driven month by month across
the break with identical settings. Per refit we record AIM-DG's active edges and each model's
hit rate on that month's forecasts, which measures how fast each one adapts.

Run:  python scripts/planted_break.py [--seed 0]   (GPU: ~5 min)
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from xaglab.models.aimdg.forecaster import AimDG
from xaglab.models.aimdg.synthetic import monthly_refits, planted_break_market
from xaglab.models.base import FitView, PredictView
from xaglab.models.neural.forecaster import TunedNeural
from xaglab.paths import EXPERIMENTS_DIR

BREAK = 900
PARAMS = {"hidden": 16, "layers": 1, "gnn_layers": 1, "dropout": 0.0, "lr": 3e-3, "weight_decay": 1e-5,
          "lam": 0.5, "window": 5, "lam2": 0.001, "lam3": 0.001}
COMMON = {"window": 5, "windows": (5,), "n_seeds": 3, "n_blocks": 2, "block": 63, "max_epochs": 60, "patience": 8}


def run(model, fs, refits) -> list[dict]:
    model.set_tuned({"joint": {"params": PARAMS}})
    rows = []
    for refit in refits:
        model.fit(FitView.build(fs, refit, 0))
        view = PredictView.build(fs, refit)
        p = model.predict(view)["p_up1"].to_numpy()
        y = fs.labels["up1"].to_numpy()[view.rows]
        row = {"origin": refit.origin, "months_after_break": (refit.origin - BREAK) / 21,
               "hit": float(np.mean((p > 0.5) == (y > 0.5)))}
        if isinstance(model, AimDG):
            sel = model.diagnostics()["selection"]
            row.update(active=sel["active"], gold5="gold@5" in sel["active"], copper1="copper@1" in sel["active"],
                       tau=sel["clock"]["tau"], rate=round(sel["clock"]["rate"], 3),
                       shift=sel["clock"]["shift"], drift=sel["drift"])
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    fs = planted_break_market(seed=args.seed, brk=BREAK)
    refits = monthly_refits(first_origin=BREAK - 4 * 21, n_refits=20)
    t0 = time.perf_counter()
    aim = run(AimDG(pop_size=30, generations=30, exhaustive_check=False, **COMMON), fs, refits)
    stat = run(TunedNeural(net="static_gnn", **COMMON), fs, refits)
    print(f"{time.perf_counter() - t0:.0f} s; break at row {BREAK}; tail = max(126, 15% of training rows)")
    print(" months  AIM-DG  static  gold@5 copper@1  τ  rate   shift  drift(mnats)  active")
    f = lambda v, k=1: "   -  " if v is None else f"{v * k:6.3f}"
    for a, s in zip(aim, stat, strict=True):
        print(f" {a['months_after_break']:+5.1f}   {a['hit']:.2f}    {s['hit']:.2f}    "
              f"{'on ' if a['gold5'] else 'off'}    {'on ' if a['copper1'] else 'off'}    "
              f"{a['tau']:2d}  {a['rate']:.3f} {f(a['shift'])}  {f(a['drift'], 1000)}   {a['active']}")
    # Integration criteria, fixed before this run (28 Sep, after the first run):
    before = [a for a in aim if a["months_after_break"] <= 0]
    after = [a for a in aim if a["months_after_break"] > 0]
    c1 = all(a["gold5"] for a in before)
    c2 = any(a["tau"] == 0 for a in after if a["months_after_break"] <= 2)
    settled = [m for m in range(1, 13) if all(not a["gold5"] and a["copper1"] for a in after if a["months_after_break"] >= m)]
    c3 = bool(settled)
    print(f"\n1. before the break, gold@5 on at every refit:             {'PASS' if c1 else 'FAIL'}")
    print(f"2. the clock fires within 2 refits after the break:         {'PASS' if c2 else 'FAIL'}")
    print(f"3. within 12 refits gold@5 off and copper@1 on, and stay:   {'PASS' if c3 else 'FAIL'}"
          + (f" (settled from +{settled[0]} months)" if c3 else ""))
    out = EXPERIMENTS_DIR / "reports" / "planted-break-4.7"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"seed{args.seed}.json").write_text(json.dumps(
        {"aimdg": aim, "static_gnn": stat, "criteria": {"gold5_before": c1, "clock_fires": c2, "settles": c3}},
        indent=1) + "\n")


if __name__ == "__main__":
    main()
