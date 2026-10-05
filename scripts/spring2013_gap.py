"""How much of each 1-day QLIKE gap to AIM-DG falls in spring 2013 (Table 4.A.2).

For the ablations A1 and A2 and the static graph, the cumulative difference in daily 1-day
QLIKE against AIM-DG over the 4,141 test days, and the share of it that falls in March to
June 2013 and on 12 April 2013 (the gold crash); also the total without each comparison's
five largest days.

Usage:  python scripts/spring2013_gap.py
"""

from __future__ import annotations

import pandas as pd

from xaglab.eval.harness import load_run
from xaglab.eval.metrics import qlike
from xaglab.paths import RUNS_DIR

AIMDG = "aimdg-20260928-120236"
OTHERS = {"A1 (top three by reliability)": "aimdg_A1-20260928-182213",
          "A2 (no variance penalty)": "aimdg_A2-20260928-183419",
          "Static graph": "static_gnn-20260927-183956"}


def daily_qlike(run_id: str) -> pd.Series:
    p = load_run(RUNS_DIR / run_id).predictions
    return pd.Series(qlike(p["var1"].to_numpy(float), p["rv1"].to_numpy(float)), index=p.index)


def main() -> None:
    base = daily_qlike(AIMDG)
    rows = []
    for name, run_id in OTHERS.items():
        d = daily_qlike(run_id) - base
        total = d.sum()
        spell = d["2013-03-01":"2013-06-30"].sum()
        crash = d["2013-04-12":"2013-04-12"].sum()
        top5 = d.abs().sort_values(ascending=False).index[:5]
        rows.append({"compared with AIM-DG": name, "total": round(total, 2),
                     "share Mar-Jun 2013": f"{spell / total:.0%}", "share 12 Apr 2013": f"{crash / total:.0%}",
                     "total without 5 largest days": round(d.drop(top5).sum(), 2)})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
