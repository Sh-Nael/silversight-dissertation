"""Read-only access to frozen, checksummed snapshots.

Experiments read data through a :class:`Snapshot` and nothing else. Opening a
snapshot verifies every file's SHA-256 against its manifest, so a reported number
can only ever come from the exact bytes that were frozen.

The live app will read the appending cache through the same schema; the snapshot
is the research lifecycle, the cache the product lifecycle (see D-06).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from xaglab.data.nodes import Node
from xaglab.data.sources import cache_filename
from xaglab.paths import SNAPSHOT_DIR

MANIFEST_NAME = "MANIFEST.json"


class SnapshotIntegrityError(RuntimeError):
    """A snapshot file is missing or its bytes no longer match the manifest."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Uniform schema: nanosecond date index, legacy 'value' column renamed to 'close'."""
    if list(df.columns) == ["value"]:
        df = df.rename(columns={"value": "close"})
    df = df.copy()
    df.index = pd.DatetimeIndex(df.index).as_unit("ns").normalize()
    df.index.name = "date"
    return df.sort_index()


@dataclass(frozen=True)
class Snapshot:
    name: str
    root: Path
    manifest: dict = field(repr=False)

    @classmethod
    def open(cls, name: str, verify: bool = True, base: Path = SNAPSHOT_DIR) -> Snapshot:
        root = base / name
        mpath = root / MANIFEST_NAME
        if not mpath.exists():
            raise FileNotFoundError(f"snapshot {name!r} has no manifest at {mpath}")
        snap = cls(name=name, root=root, manifest=json.loads(mpath.read_text()))
        if verify:
            snap.verify()
        return snap

    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp(self.manifest["snapshot_end"])

    @property
    def files(self) -> list[str]:
        return list(self.manifest["files"])

    def verify(self) -> None:
        problems = []
        for fname, meta in self.manifest["files"].items():
            p = self.root / fname
            if not p.exists():
                problems.append(f"missing: {fname}")
            elif sha256_file(p) != meta["sha256"]:
                problems.append(f"checksum mismatch: {fname}")
        if problems:
            raise SnapshotIntegrityError(f"snapshot {self.name!r}: " + "; ".join(problems))

    def read(self, fname: str) -> pd.DataFrame:
        if fname not in self.manifest["files"]:
            raise KeyError(f"{fname!r} is not part of snapshot {self.name!r}")
        p = self.root / fname
        if p.suffix == ".parquet":
            return _normalise(pd.read_parquet(p))
        if p.suffix == ".csv":
            return pd.read_csv(p)
        raise ValueError(f"unsupported snapshot file type: {fname}")

    def read_node(self, node: Node) -> pd.DataFrame | None:
        """The node's series from its primary source, or None if not in this snapshot."""
        fname = cache_filename(node.primary.kind, node.primary.symbol)
        return self.read(fname) if fname in self.manifest["files"] else None


def write_snapshot(
    name: str,
    frames: dict[str, pd.DataFrame],
    snapshot_end: str,
    created: str,
    note: str,
    extra: dict | None = None,
    base: Path = SNAPSHOT_DIR,
) -> Snapshot:
    """Freeze a new snapshot. Refuses to overwrite: snapshots are immutable by contract."""
    root = base / name
    if (root / MANIFEST_NAME).exists():
        raise FileExistsError(f"snapshot {name!r} already exists; snapshots are never rewritten")
    root.mkdir(parents=True, exist_ok=True)
    entries = {}
    for fname, df in sorted(frames.items()):
        p = root / fname
        if p.suffix == ".parquet":
            df.to_parquet(p, compression="snappy")
        elif p.suffix == ".csv":
            df.to_csv(p, index=False, lineterminator="\n")
        else:
            raise ValueError(f"unsupported snapshot file type: {fname}")
        entries[fname] = {"rows": len(df), "columns": list(df.columns), "sha256": sha256_file(p)}
    manifest = {"snapshot": name, "snapshot_end": snapshot_end, "created": created,
                "note": note, "files": entries, **(extra or {})}
    (root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return Snapshot.open(name, base=base)
