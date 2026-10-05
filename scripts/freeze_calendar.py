"""Freeze the macro event calendar (CPI, NFP, FOMC) as snapshot `calendar_v1`.

Pulled once from FRED vintage dates and the Federal Reserve's FOMC calendar pages,
then checksummed like the price snapshot. Experiments read the frozen table only.
Re-running verifies instead of rewriting. See D-12 and D-18.

Run:  python scripts/freeze_calendar.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xaglab.data.events import (
    FED_CALENDAR_URL,
    FED_HISTORICAL_URL,
    FRED_VINTAGE_SERIES,
    fetch_event_table,
)
from xaglab.data.store import MANIFEST_NAME, Snapshot, write_snapshot
from xaglab.paths import SNAPSHOT_DIR

NAME = "calendar_v1"


def main() -> int:
    if (SNAPSHOT_DIR / NAME / MANIFEST_NAME).exists():
        snap = Snapshot.open(NAME)  # raises on any mismatch
        print(f"Snapshot '{NAME}' intact: {len(snap.files)} file(s) verified.")
        return 0
    table = fetch_event_table()
    snap = write_snapshot(
        NAME,
        {"events.csv": table},
        snapshot_end=str(table["date"].max()),
        created=date.today().isoformat(),
        note="Scheduled US macro events. CPI/NFP = FRED vintage dates of CPIAUCNS/PAYEMS; "
             "FOMC = last day of each meeting from federalreserve.gov (scheduled flag "
             "separates unscheduled actions). Dates only; no consensus/surprise data.",
        extra={"sources": {"fred_vintage_series": FRED_VINTAGE_SERIES,
                           "fomc_historical": FED_HISTORICAL_URL,
                           "fomc_calendar": FED_CALENDAR_URL}},
    )
    counts = table.groupby("event").size().to_dict()
    print(f"Frozen '{NAME}': {len(table)} events {counts} -> {snap.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
