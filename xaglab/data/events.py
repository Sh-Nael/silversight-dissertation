"""Scheduled high-impact US macro events: CPI, NFP (Employment Situation), FOMC.

Sources (all public, all aggregate, D-12):
  * CPI and NFP publication days = FRED/ALFRED *vintage dates* of the unadjusted
    series CPIAUCNS and PAYEMS: the days on which new data actually appeared. This
    captures real-world delays (2013 shutdown) and cancellations (Oct/Nov 2025)
    and excludes seasonal-factor-only revisions.
  * FOMC decision days = last day of each meeting, parsed from the Federal
    Reserve's own calendar pages. Scheduled meetings and unscheduled meetings are
    kept apart: only scheduled dates may be used for "days until" features, since
    an unscheduled meeting cannot be anticipated.

Consensus forecasts (and therefore surprise magnitudes) are not freely available
historically, so the dissertation uses date-proximity only. Stated limitation.

Timing convention: all three releases happen before the silver session close
(CPI/NFP 08:30 ET, FOMC statement 14:00 ET), so an event on day t is known at the
close of t, which is when a forecast is made.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

from xaglab.data.sources import load_env_key

EVENT_TYPES = ("cpi", "nfp", "fomc")
FRED_VINTAGE_SERIES = {"cpi": "CPIAUCNS", "nfp": "PAYEMS"}
FED_HISTORICAL_URL = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
FED_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
_UA = {"User-Agent": "Mozilla/5.0 (compatible; xaglab research pipeline)"}
_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}
_MONTH_ABBR = {k[:3]: v for k, v in _MONTHS.items()}


# ---------------------------------------------------------------- fetching

def fetch_vintage_dates(series_id: str, start: str = "2005-01-01") -> list[str]:
    key = load_env_key("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY missing (.env)")
    r = requests.get(
        "https://api.stlouisfed.org/fred/series/vintagedates",
        params={"series_id": series_id, "api_key": key, "file_type": "json",
                "realtime_start": start, "limit": 10000},
        timeout=30,
    )
    r.raise_for_status()
    return list(r.json()["vintage_dates"])


def _get(url: str) -> str:
    r = requests.get(url, headers=_UA, timeout=30)
    r.raise_for_status()
    return r.text


_HIST_RE = re.compile(
    r"^(?P<when>.+?)(?: \((?P<tag>unscheduled|cancelled|notation vote)\))?"
    r"(?: (?P<kind>Meeting|Conference Call))? - (?P<year>\d{4})$"
)


def _last_month_day(when: str) -> tuple[int, int] | None:
    """Decision day of a possibly multi-month span: 'April/May 30-1', 'July 31-August 1'."""
    months = [_MONTH_ABBR[w[:3].lower()] for w in re.findall(r"[A-Za-z]+", when)
              if w[:3].lower() in _MONTH_ABBR]
    days = re.findall(r"\d+", when)
    return (months[-1], int(days[-1])) if months and days else None


def parse_fomc_historical(page: str) -> list[dict]:
    """Parse one fomchistoricalYYYY page.

    Kept: scheduled meetings, and unscheduled meetings/actions (flagged). Dropped:
    conference calls, notation votes and cancelled meetings.
    """
    out = []
    for raw in re.findall(r"<h5[^>]*>(.*?)</h5>", page, re.DOTALL):
        title = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", raw))).strip()
        m = _HIST_RE.match(title)
        if not m:
            continue
        tag, kind = m.group("tag"), m.group("kind")
        if tag in ("cancelled", "notation vote") or kind == "Conference Call":
            continue
        if kind != "Meeting" and tag != "unscheduled":
            continue
        md = _last_month_day(m.group("when"))
        if md is None:
            continue
        year = int(m.group("year"))
        out.append({"date": f"{year:04d}-{md[0]:02d}-{md[1]:02d}", "event": "fomc",
                    "scheduled": tag != "unscheduled", "source": title})
    return out


def parse_fomc_calendar(page: str) -> list[dict]:
    """Parse the current fomccalendars page (recent years, including announced future dates)."""
    out = []
    parts = re.split(r"<h4><a[^>]*>(\d{4}) FOMC Meetings</a></h4>", page)
    for year_s, body in zip(parts[1::2], parts[2::2], strict=True):
        year = int(year_s)
        for month_s, date_s in re.findall(
            r"fomc-meeting__month[^>]*>\s*<strong>(.*?)</strong>.*?fomc-meeting__date[^>]*>(.*?)</div>",
            body, re.DOTALL,
        ):
            month_txt = html.unescape(re.sub(r"<[^>]+>", "", month_s)).strip().lower()
            date_txt = html.unescape(re.sub(r"<[^>]+>", "", date_s)).strip()
            low = date_txt.lower()
            if "notation" in low or "cancel" in low:
                continue
            last_month = month_txt.split("/")[-1][:3]
            days = re.findall(r"\d+", date_txt)
            if last_month not in _MONTH_ABBR or not days:
                continue
            out.append({"date": f"{year:04d}-{_MONTH_ABBR[last_month]:02d}-{int(days[-1]):02d}",
                        "event": "fomc", "scheduled": "unscheduled" not in low,
                        "source": f"{year} {month_txt} {date_txt}"})
    return out


def fetch_event_table(first_year: int = 2005, last_hist_year: int = 2020) -> pd.DataFrame:
    """Download and assemble the full event table (date, event, scheduled, source)."""
    rows: list[dict] = []
    for ev, sid in FRED_VINTAGE_SERIES.items():
        for d in fetch_vintage_dates(sid):
            rows.append({"date": d, "event": ev, "scheduled": True,
                         "source": f"FRED vintage {sid}"})
    for year in range(first_year, last_hist_year + 1):
        rows += parse_fomc_historical(_get(FED_HISTORICAL_URL.format(year=year)))
    rows += [r for r in parse_fomc_calendar(_get(FED_CALENDAR_URL))
             if int(r["date"][:4]) > last_hist_year]
    df = pd.DataFrame(rows).drop_duplicates(subset=["date", "event"]).sort_values(["date", "event"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- features

@dataclass(frozen=True)
class EventCalendar:
    """Event table mapped onto a trading calendar."""

    table: pd.DataFrame  # columns: date, event, scheduled

    @classmethod
    def from_table(cls, df: pd.DataFrame) -> EventCalendar:
        t = df.copy()
        t["date"] = pd.to_datetime(t["date"]).dt.normalize().astype("datetime64[ns]")
        t["scheduled"] = t["scheduled"].astype(str).str.lower().isin(("true", "1"))
        return cls(t[["date", "event", "scheduled"]].sort_values("date").reset_index(drop=True))

    def features(self, index: pd.DatetimeIndex, cap: int = 5) -> pd.DataFrame:
        """Per trading day t: trading days until the next scheduled event and since the
        last event of each type, capped at `cap` (cap means "far away / none").

        days_to uses scheduled events only (anticipation must not use unscheduled
        dates); days_since uses every event (it is past information). Events on a
        non-trading day are attributed to the next trading day.
        """
        index = pd.DatetimeIndex(index).as_unit("ns")
        pos = np.arange(len(index))
        out = {}
        for ev in EVENT_TYPES:
            sub = self.table[self.table.event == ev]
            for kind, rows in (("to", sub[sub.scheduled]), ("since", sub)):
                # Trading-day position of each event (next trading day if on a holiday).
                ev_pos = np.unique(index.searchsorted(rows["date"].to_numpy(), side="left"))
                ev_pos = ev_pos[ev_pos < len(index)]
                val = np.full(len(index), cap, dtype=float)
                if len(ev_pos):
                    if kind == "to":
                        j = np.searchsorted(ev_pos, pos, side="right")  # strictly after t
                        ok = j < len(ev_pos)
                        val[ok] = np.minimum(ev_pos[j[ok]] - pos[ok], cap)
                    else:
                        j = np.searchsorted(ev_pos, pos, side="right") - 1  # at or before t
                        ok = j >= 0
                        val[ok] = np.minimum(pos[ok] - ev_pos[j[ok]], cap)
                out[f"{ev}_days_{kind}"] = val
        return pd.DataFrame(out, index=index)
