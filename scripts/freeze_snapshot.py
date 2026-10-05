"""Freeze the dissertation dataset snapshot.

Cuts every cached series at the frozen end date (2026-06-30, decided and diarised
29 Jul 2026), writes them to data/snapshots/dissertation_v1/, and records a
SHA-256 checksum per file plus a manifest. The snapshot is immutable: experiments
read from here and only here. The live cache keeps appending for the app; the
snapshot never changes.

Run:  python scripts/freeze_snapshot.py
Re-running verifies checksums instead of overwriting (fails loudly on mismatch).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SNAPSHOT_END = "2026-06-30"
SNAP_DIR = Path(__file__).resolve().parents[1] / "data" / "snapshots" / "dissertation_v1"
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"
MANIFEST = SNAP_DIR / "MANIFEST.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def freeze() -> int:
    files = sorted(CACHE_DIR.glob("*.parquet"))
    if not files:
        print("No cached series found; run the availability check first.")
        return 1

    if MANIFEST.exists():
        return verify()

    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    entries = {}
    for src in files:
        df = pd.read_parquet(src)
        df = df[df.index <= pd.Timestamp(SNAPSHOT_END)]
        out = SNAP_DIR / src.name
        # Deterministic write: no compression variance across runs.
        df.to_parquet(out, compression="snappy")
        entries[src.name] = {
            "rows": len(df),
            "start": str(df.index[0].date()),
            "end": str(df.index[-1].date()),
            "columns": list(df.columns),
            "sha256": sha256(out),
        }
        print(f"  frozen  {src.name:<32} {len(df):>6} rows  -> {entries[src.name]['end']}")

    manifest = {
        "snapshot": "dissertation_v1",
        "snapshot_end": SNAPSHOT_END,
        "created": "2026-07-31",   # decision + freeze date, fixed deliberately
        "note": "Immutable dissertation dataset. Experiments must read only from "
                "this directory. Live cache continues to append for the app.",
        "files": entries,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written: {MANIFEST}")
    print("Snapshot is now FROZEN. Re-running this script verifies, never overwrites.")
    return 0


def verify() -> int:
    manifest = json.loads(MANIFEST.read_text())
    bad = []
    for name, meta in manifest["files"].items():
        p = SNAP_DIR / name
        if not p.exists():
            bad.append(f"MISSING: {name}")
        elif sha256(p) != meta["sha256"]:
            bad.append(f"CHECKSUM MISMATCH: {name}")
        else:
            print(f"  ok      {name:<32} {meta['rows']:>6} rows  sha256 verified")
    if bad:
        print("\n*** SNAPSHOT INTEGRITY FAILURE ***")
        for b in bad:
            print(" ", b)
        return 2
    print(f"\nSnapshot '{manifest['snapshot']}' intact: "
          f"{len(manifest['files'])} files, end {manifest['snapshot_end']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(freeze())
