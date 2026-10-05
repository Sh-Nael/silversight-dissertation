"""Filesystem layout, resolved once. Every module imports paths from here."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
EXPERIMENTS_DIR = ROOT / "experiments"
CONFIG_DIR = EXPERIMENTS_DIR / "configs"
RUNS_DIR = EXPERIMENTS_DIR / "runs"
DEV_DIR = EXPERIMENTS_DIR / "dev"  # development runs (tuning stage only)
REPORTS_DIR = EXPERIMENTS_DIR / "reports"
TABLES_DIR = EXPERIMENTS_DIR / "tables"  # the table pack (xag tables)
FIGURES_DIR = EXPERIMENTS_DIR / "figures"  # the figure pack (npm run figures)
ENV_FILE = ROOT / ".env"
WEB_DIR = ROOT / "web"
WEB_DIST = WEB_DIR / "dist"
