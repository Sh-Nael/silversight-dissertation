"""Export the cached driver series to one aligned CSV and a quick-look chart.

Viewing utility only; the modelling pipeline does its own alignment with the
frozen protocol. Run:  python scripts/export_data.py
Outputs: data/all_series.csv and data/all_series.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xaglab.data.nodes import CORE_KEYS, HISTORY_START, NODES_BY_KEY
from xaglab.data.sources import fetch_close


def main() -> None:
    series = {}
    for key in CORE_KEYS:
        node = NODES_BY_KEY[key]
        for src in filter(None, (node.primary, node.fallback)):
            s = fetch_close(src.kind, src.symbol, HISTORY_START)
            if s is not None and len(s):
                series[key] = s
                break

    df = pd.DataFrame(series).ffill()
    out_dir = Path(__file__).resolve().parents[1] / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "all_series.csv"
    df.to_csv(csv_path)

    # One panel per series: scales differ wildly (silver ~$30, DXY ~100, yields ~1%).
    fig, axes = plt.subplots(len(df.columns), 1, figsize=(12, 2.2 * len(df.columns)), sharex=True)
    for ax, col in zip(axes, df.columns):
        ax.plot(df.index, df[col], linewidth=0.6)
        ax.set_ylabel(col, fontsize=9)
        ax.grid(alpha=0.3)
    for x, label in [("2008-09-15", "2008"), ("2011-04-25", "2011"),
                     ("2020-03-16", "2020"), ("2022-03-16", "2022")]:
        for ax in axes:
            ax.axvline(pd.Timestamp(x), color="red", alpha=0.25, linewidth=0.8)
        axes[0].annotate(label, (pd.Timestamp(x), axes[0].get_ylim()[1]),
                         fontsize=7, color="red", ha="center")
    axes[-1].set_xlabel("date")
    fig.suptitle("AIM-DG driver series, daily closes (regime breaks in red)", fontsize=11)
    fig.tight_layout()
    png_path = out_dir / "all_series.png"
    fig.savefig(png_path, dpi=110)

    print(f"rows: {len(df)}   columns: {list(df.columns)}")
    print(f"csv : {csv_path}")
    print(f"png : {png_path}")


if __name__ == "__main__":
    main()
