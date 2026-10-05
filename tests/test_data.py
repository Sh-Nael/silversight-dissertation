"""Unit tests for the data layer: store integrity, alignment rules, event parsing."""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest

from xaglab.data.events import EventCalendar, parse_fomc_historical
from xaglab.data.panel import build_panel_from_frames
from xaglab.data.store import Snapshot, SnapshotIntegrityError


def test_snapshots_verify(prices, calendar):
    assert prices.end == pd.Timestamp("2026-06-30")
    assert len(prices.files) == 7 and calendar.files == ["events.csv"]


def test_tampered_snapshot_fails_loudly(prices, tmp_path):
    shutil.copytree(prices.root, tmp_path / prices.name)
    victim = tmp_path / prices.name / prices.files[0]
    data = bytearray(victim.read_bytes())
    data[-100] ^= 0xFF
    victim.write_bytes(bytes(data))
    with pytest.raises(SnapshotIntegrityError):
        Snapshot.open(prices.name, base=tmp_path)


def _bars(dates, close, **cols):
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                       "volume": 100.0}, index=pd.DatetimeIndex(dates))
    for k, v in cols.items():
        df[k] = v
    return df


def test_alignment_rules():
    days = pd.bdate_range("2020-01-06", periods=12)
    silver = _bars(days, np.linspace(10, 11, 12), high=np.linspace(10, 11, 12) + 0.2,
                   low=np.linspace(10, 11, 12) - 0.2)
    silver.loc[days[3], "close"] = silver.loc[days[3], "high"] + 1  # close outside range
    silver.loc[days[4], "volume"] = 0.0                               # zero volume
    gold = _bars(days.delete([2, 6, 7, 8, 9, 10, 11]), 100.0)          # gap of 7 days at the end
    ry = pd.DataFrame({"close": np.arange(12.0)}, index=days)          # FRED value dated d
    p = build_panel_from_frames({"silver": silver, "gold": gold, "real_yield_10y": ry}, max_ffill=5)

    s = p.nodes["silver"]
    assert s.loc[days[3], "high"] == s.loc[days[3], "close"]           # range repaired
    assert np.isnan(s.loc[days[4], "volume"])                           # zero volume masked
    g = p.nodes["gold"]
    assert g.loc[days[2], "filled"] and not g.loc[days[2], "bar_ok"]    # filled, flagged
    assert g.loc[days[6:11], "close"].notna().all()                     # 5 days filled ...
    assert np.isnan(g.loc[days[11], "close"])                           # ... the 6th is missing
    r = p.nodes["real_yield_10y"]["close"]
    assert np.isnan(r.iloc[0]) and r.iloc[1] == 0.0 and r.iloc[5] == 4.0  # usable next day


def test_event_features_known_answer():
    idx = pd.bdate_range("2021-03-01", periods=10)  # Mon 1 .. Fri 12 March
    table = pd.DataFrame({"date": ["2021-03-03", "2021-03-06", "2021-03-10"],
                          "event": ["cpi", "cpi", "fomc"], "scheduled": [True, True, False]})
    f = EventCalendar.from_table(table).features(idx, cap=5)
    # CPI on Wed 3rd and Sat 6th (-> Mon 8th)
    assert f["cpi_days_to"].tolist()[:6] == [2, 1, 3, 2, 1, 5]
    assert f["cpi_days_since"].tolist()[:6] == [5, 5, 0, 1, 2, 0]
    # an unscheduled FOMC is never anticipated, but is remembered
    assert (f["fomc_days_to"] == 5).all()
    assert f["fomc_days_since"].tolist()[7:] == [0, 1, 2]


@pytest.mark.parametrize("title,date,scheduled", [
    ("January 29-30 Meeting - 2008", "2008-01-30", True),
    ("April/May 30-1 Meeting - 2013", "2013-05-01", True),
    ("July 31-August 1  Meeting - 2012", "2012-08-01", True),
    ("Jan/Feb 31-1 Meeting - 2017", "2017-02-01", True),
    ("March 15 (unscheduled) Meeting - 2020", "2020-03-15", False),
    ("October 16 (unscheduled) - 2013", "2013-10-16", False),
])
def test_fomc_title_parsing(title, date, scheduled):
    (row,) = parse_fomc_historical(f"<h5>{title}</h5>")
    assert (row["date"], row["scheduled"]) == (date, scheduled)


@pytest.mark.parametrize("title", ["March 17-18 (cancelled) Meeting - 2020",
                                   "January 21 Conference Call - 2008",
                                   "March 19 (notation vote) - 2020"])
def test_fomc_non_meetings_are_dropped(title):
    assert parse_fomc_historical(f"<h5>{title}</h5>") == []


def test_real_calendar_has_eight_scheduled_fomc_per_year(calendar):
    t = calendar.read("events.csv")
    f = t[(t.event == "fomc") & t.scheduled.astype(bool)]
    per_year = f.groupby(f.date.str[:4]).size()
    assert (per_year.drop("2020") == 8).all() and per_year["2020"] == 7  # March 2020 cancelled
