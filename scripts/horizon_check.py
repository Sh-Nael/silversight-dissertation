"""Is direction more predictable at longer horizons? A development-data check before any
protocol change (28 Sep).

At 5 yearly tuning points, for horizons 1, 5, 10, 21 trading days: the pooled features, a
label "silver up over the next h days", 3 purged rolling-origin validation blocks *inside the
training span* (purge = h, and labels only where they are known by the span's end), a small
Optuna search per learner, and the out-of-fold log-loss and hit rate. Skill is measured
against the block's own base rate (always-up), because longer horizons drift up more.

Run:  python scripts/horizon_check.py   (~10 min, CPU)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xaglab.cli import _data
from xaglab.data.panel import load_default_panel
from xaglab.eval.metrics import logloss
from xaglab.models.baselines import GBMClassifierLearner, LogisticLearner
from xaglab.models.tuning import DirectionTask, cv_blocks, cv_predict, tune, tuning_points

HORIZONS = (1, 5, 10, 21)
POINTS = (2, 5, 8, 11, 14)


def main() -> None:
    fs, cal = _data()
    close = load_default_panel().close("silver")
    X = fs.pooled()
    refits = cal.all_refits(None)
    idx = sorted(set(tuning_points([r.origin for r in refits], 252)))
    rows = []
    for h in HORIZONS:
        fwd = np.log(close.shift(-h) / close)
        y = (fwd > 0).astype(float).where(fwd.notna()).to_numpy()
        task = DirectionTask(y)
        for k in POINTS:
            r = refits[idx[k]]
            end = r.train_end - h                       # labels known by train_end only
            blocks = cv_blocks(r.train_start, end, purge=h, n_blocks=3, block=252)
            for name, learner, trials in (("linear", LogisticLearner(), 12), ("gbm", GBMClassifierLearner(), 8)):
                res = tune(learner, task, X, blocks, n_trials=trials, seed=7)
                pred, vrows = cv_predict(learner, task, X, blocks, res.params, seed=7)
                yy = y[vrows]
                base = yy.mean()
                rows.append({"h": h, "point": str(fs.index[r.origin].date())[:4], "model": name,
                             "logloss": float(logloss(pred, yy).mean()),
                             "logloss_base": float(logloss(np.full(len(yy), base), yy).mean()),
                             "hit": float(np.mean((pred > 0.5) == (yy > 0.5))), "always_up": float(base),
                             "n": len(yy)})
                print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df["skill_mnats"] = 1000 * (df.logloss_base - df.logloss)
    df["hit_over_base"] = df.hit - df.always_up
    pd.set_option("display.width", 200)
    print("\nMean over 5 tuning points (skill = base-rate log-loss minus model log-loss; positive = information):")
    print(df.groupby(["h", "model"])[["skill_mnats", "hit", "always_up", "hit_over_base"]].mean().round(4))
    from xaglab.paths import EXPERIMENTS_DIR
    out = EXPERIMENTS_DIR / "reports" / "horizon-check"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "horizon_check.csv", index=False)


if __name__ == "__main__":
    main()
