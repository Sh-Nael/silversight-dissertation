"""Documented regime windows the adaptive model must handle (Ch1 §1.7, Ch5).

Used for shading charts and for the regime analysis. Defined once, here.
"""

from __future__ import annotations

#: (label, start, end), inclusive calendar dates.
REGIMES: tuple[tuple[str, str, str], ...] = (
    ("2008 GFC", "2008-01-01", "2009-06-30"),
    ("2011 silver spike", "2011-01-01", "2011-12-31"),
    ("2013 taper tantrum", "2013-04-01", "2013-12-31"),  # gold crash (Apr) and taper tantrum; added in D-40
    ("2020 COVID", "2020-02-01", "2020-06-30"),
    ("2022 rate shock", "2022-01-01", "2022-12-31"),
)
