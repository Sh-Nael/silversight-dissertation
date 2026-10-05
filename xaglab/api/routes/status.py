"""System status: is the app healthy, is the data intact, which code is running.

This feeds the status chips in the web app's top bar and the Overview page.
"""

from __future__ import annotations

import json
import subprocess

from fastapi import APIRouter, Request
from pydantic import BaseModel

from xaglab import __version__
from xaglab.data.store import Snapshot, SnapshotIntegrityError
from xaglab.paths import ROOT, SNAPSHOT_DIR

router = APIRouter(tags=["status"])


class SnapshotStatus(BaseModel):
    name: str
    files: int
    snapshot_end: str
    verified: bool
    error: str | None = None


class CodeStatus(BaseModel):
    commit: str
    dirty: bool
    version: str


class Status(BaseModel):
    ok: bool
    code: CodeStatus
    snapshots: list[SnapshotStatus]
    runs: int


class Health(BaseModel):
    ok: bool


@router.get("/health", response_model=Health)
def health() -> Health:
    """Liveness check: the API process is up."""
    return Health(ok=True)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=False).stdout.strip()


def snapshot_statuses() -> list[SnapshotStatus]:
    out = []
    for manifest in sorted(SNAPSHOT_DIR.glob("*/MANIFEST.json")):
        name = manifest.parent.name
        meta = json.loads(manifest.read_text())
        try:
            Snapshot.open(name)  # recomputes every SHA-256
            out.append(SnapshotStatus(name=name, files=len(meta["files"]),
                                      snapshot_end=meta["snapshot_end"], verified=True))
        except SnapshotIntegrityError as exc:
            out.append(SnapshotStatus(name=name, files=len(meta["files"]),
                                      snapshot_end=meta["snapshot_end"], verified=False,
                                      error=str(exc)))
    return out


@router.get("/status", response_model=Status)
def status(request: Request) -> Status:
    """Code version, snapshot integrity (checksums recomputed on every call) and run count."""
    snaps = snapshot_statuses()
    base = request.app.state.runs_dir
    runs = sum(1 for p in base.glob("*/run.json")) if base.exists() else 0
    code = CodeStatus(commit=_git("rev-parse", "--short", "HEAD") or "unknown",
                      dirty=bool(_git("status", "--porcelain", "--untracked-files=no")),
                      version=__version__)
    return Status(ok=all(s.verified for s in snaps), code=code, snapshots=snaps, runs=runs)
