"""Data availability gate.

Answers one question before any modelling begins: can we actually obtain ~20 years
of aligned daily history for silver and every candidate driver from free public
sources? If not, the design changes now rather than in Week 11.

Reports per node: source used, row count, date span, gaps, and coverage against the
target's trading calendar. Then reports the intersected calendar across all core
nodes, which is what the graph model will actually train on.

Run:  python scripts/check_data_availability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xaglab.data.nodes import CORE_KEYS, HISTORY_START, NODES, TARGET
from xaglab.data.regimes import REGIMES
from xaglab.data.sources import fetch_close


def describe(s: pd.Series) -> dict:
    span_days = (s.index[-1] - s.index[0]).days
    years = span_days / 365.25
    # Business days we would expect versus what we actually have.
    expected = len(pd.bdate_range(s.index[0], s.index[-1]))
    gaps = s.index.to_series().diff().dt.days.dropna()
    return {
        "rows": len(s),
        "start": s.index[0].date().isoformat(),
        "end": s.index[-1].date().isoformat(),
        "years": round(years, 1),
        "bday_coverage": round(100 * len(s) / max(expected, 1), 1),
        "max_gap_days": int(gaps.max()) if len(gaps) else 0,
    }


def main() -> int:
    print(f"Fetching from {HISTORY_START}. First run hits the network; later runs use cache.\n")

    series: dict[str, pd.Series] = {}
    rows: list[dict] = []

    for node in NODES:
        attempts = [("primary", node.primary)]
        if node.fallback:
            attempts.append(("fallback", node.fallback))

        got: pd.Series | None = None
        used = "-"
        for label, src in attempts:
            got = fetch_close(src.kind, src.symbol, HISTORY_START)
            if got is not None and len(got) > 0:
                used = f"{src.kind}:{src.symbol} ({label})"
                break

        if got is None or len(got) == 0:
            status = "OPTIONAL - not built yet" if node.role == "optional" else "*** FAILED ***"
            rows.append({"node": node.key, "role": node.role, "source": used, "status": status})
            print(f"  {node.key:<16} {status}")
            continue

        series[node.key] = got
        info = describe(got)
        rows.append({"node": node.key, "role": node.role, "source": used, "status": "ok", **info})
        print(
            f"  {node.key:<16} {info['rows']:>6} rows  "
            f"{info['start']} -> {info['end']}  ({info['years']:>4.1f}y)  "
            f"bday {info['bday_coverage']:>5.1f}%  max gap {info['max_gap_days']:>3}d"
        )

    print()
    core_present = [k for k in CORE_KEYS if k in series]
    missing = [k for k in CORE_KEYS if k not in series]

    if TARGET not in series:
        print("VERDICT: FAIL - the target series (silver) could not be fetched.")
        return 1

    frame = pd.DataFrame({k: series[k] for k in core_present})

    # The calendar the model can actually train on: every core driver observed.
    complete = frame.dropna(how="any")
    # Forward-filling macro series (published with a lag) is the realistic protocol.
    ffilled = frame.ffill().dropna(how="any")

    print("Joint calendar across core nodes")
    print(f"  core nodes present : {len(core_present)}/{len(CORE_KEYS)}  {core_present}")
    if missing:
        print(f"  MISSING            : {missing}")
    print(f"  union of dates     : {len(frame):>6}")
    print(f"  strict complete    : {len(complete):>6} rows  "
          f"({complete.index[0].date()} -> {complete.index[-1].date()})" if len(complete) else
          "  strict complete    :      0")
    if len(ffilled):
        yrs = (ffilled.index[-1] - ffilled.index[0]).days / 365.25
        print(f"  ffilled complete   : {len(ffilled):>6} rows  "
              f"({ffilled.index[0].date()} -> {ffilled.index[-1].date()}, {yrs:.1f}y)")

    print("\nRegime-break coverage (the model's reason to exist)")
    for label, lo, hi in REGIMES:
        n = len(ffilled.loc[lo:hi]) if len(ffilled) else 0
        mark = "ok" if n > 100 else "THIN"
        print(f"  {label:<20} {n:>5} rows  {mark}")

    ok = len(ffilled) >= 3000 and not missing
    print("\nVERDICT:", "PASS - proceed with the 20-year daily design." if ok else
          "REVIEW - see missing/thin items above before committing to the design.")

    out = Path(__file__).resolve().parents[1] / "data" / "availability_report.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Report written: {out}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
