# SilverSight: an Adaptive Inter-Market Driver Graph for Silver (XAG/USD)

The software and results of the MSc dissertation *An Adaptive Inter-Market Driver Graph for Silver (XAG/USD): Regime-Dependent Forecasting and Driver Attribution* (7CS077, University of Wolverhampton, 2026).

**Author:** Shamim Nael (2464263) · **Supervisor:** Dr Ajmal Shahbaz

AIM-DG forecasts silver's next-day and next-week direction and volatility from five related markets (gold, the US dollar index, the 10-year real yield, copper and the VIX), each read at three lags. Every month it re-selects which of these 15 links to use, with the uncertainty-guided evolutionary mechanism of Movahed, Toosi and Nael (2026), and it records which links it used, so each forecast carries an attribution. It is compared with eight other models on eight frozen walk-forward test folds (4,141 trading days, 4 January 2010 to 23 June 2026), with monthly refits and significance tests.

This repository holds everything needed to check the results: the code, the frozen data, the forecasts of every run, the result tables and figures, the tests, and a local web application for browsing it all.

## Main results

| | Finding | Evidence |
|---|---|---|
| Volatility | Forecastable: every model beats climatology (Diebold–Mariano p < 0.001). The static graph with all 15 links on is the best single model at 1 day (QLIKE −6.974). | `experiments/tables/T01`, `T04`, `T08` |
| Direction | Barely forecastable: 1-day hit rates of 50.4–53.8% against a 51.8% up-rate; no Brier score significantly better than climatology. | `T02`, `T03`, `T05` |
| Adaptation (main hypothesis) | AIM-DG does not beat the static graph: significantly worse volatility forecasts (1 day p 0.006, 5 days p 0.018), level on direction. | `T07`, `T15` |
| Why | The month-to-month usefulness of a link is mostly noise: optimising the selection harder is worse out of sample; ablations and settings change which links are chosen far more than accuracy. | `T13`, `T14`, `T22`, `T23` |
| Practical value | At a 10% coverage target the best models are right 57–58% of the time; a fixed stop-and-target rule on those days is profitable for the linear model (profit factor 1.41, p 0.006), on both buy and sell trades. Trading every day loses. | `T16`–`T21` |
| Edge dynamics | The selected links do not follow the four stress regimes (2011, 2013, 2020, 2022); only VIX at lag 1 persists. | `T24`–`T26` |

All 27 tables are listed in [`experiments/tables/index.md`](experiments/tables/index.md) and all 33 figures in [`experiments/figures/index.md`](experiments/figures/index.md).

## Run it on your own machine

**You need:** Python 3.13 and about 2 GB of disk space (most of it PyTorch). Developed and tested on Linux; macOS should work the same way. An NVIDIA GPU is needed only to re-train the neural models; everything else runs on a CPU.

### 1. Install

The quickest route uses [uv](https://docs.astral.sh/uv/), which also installs Python 3.13 if you don't have it:

```bash
# install uv (once): see https://docs.astral.sh/uv/getting-started/installation/
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/Sh-Nael/silversight-dissertation.git
cd silversight-dissertation

uv venv --python 3.13 .venv
# PyTorch for CPU (enough for everything except re-training the neural models)
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv -e ".[dev]"
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

To re-train the neural models on an NVIDIA GPU, install PyTorch from the CUDA index that matches your driver instead (for example `--index-url https://download.pytorch.org/whl/cu130`); see [pytorch.org](https://pytorch.org/get-started/locally/).

Without uv: create a Python 3.13 virtual environment (`python3.13 -m venv .venv`), activate it, then run the same two installs with `pip install` in place of `uv pip install --python .venv`.

### 2. Check the installation

```bash
xag verify      # every frozen data file matches its SHA-256 checksum
pytest -q       # 126 tests, about 2 minutes on a laptop CPU
```

### 3. Browse the results in the web application

```bash
xag ui
```

Then open **http://localhost:8000**. The app runs only on your machine (it listens on 127.0.0.1) and reads the run folders in `experiments/runs/`. Pages:

| Page | Shows |
|---|---|
| Overview | Silver's history with the regime windows and test folds; the headline scores |
| Runs | Every run with its manifest: code version, data checksums, settings, timings, tuning log |
| Compare | Scoreboards and significance tests between any runs; per-fold and cumulative loss differences |
| Calibration | Reliability diagrams; accuracy against coverage, for up and down calls separately |
| Signals | The stop-and-target signal layer: trades, win rates, profit factors, equity curves |
| Folds | The frozen walk-forward calendar |
| Data | Every market, every feature as the models see it, and the data-quality audit |
| Driver graph | Which links AIM-DG used in each month, their reliability, and the regime tests |

Every chart has an **Export figure** button (SVG or PNG). API documentation is at http://localhost:8000/docs. Stop the app with Ctrl+C.

### 4. Regenerate the tables

```bash
xag tables      # rewrites experiments/tables/ from the report files; nothing is recomputed
xag report linear-20260926-100937 gbm-20260926-100959 static_gnn-20260927-183956 \
           aimdg-20260928-120236 climatology-20260926-091404 --reference static_gnn
```

`xag report` recomputes the scoreboard and the significance tests from the stored forecasts of any runs you name (seconds).

### 5. Re-run the experiments (optional)

Each run writes a new folder in `experiments/runs/`; the stored runs are not touched.

```bash
xag run climatology ewma ar_garch                 # seconds
xag run linear gbm --jobs 8                       # minutes; tuning included
xag run lstm --tuning-from lstm-dev-r2-20260927-113901 --jobs 6              # GPU, about 25 min
xag run static_gnn --tuning-from static_gnn-dev-20260927-164105 --jobs 6     # GPU, about 1.5 h
xag run aimdg --tuning-from aimdg-dev-20260928-090733                        # GPU, about 2.5 h
```

Times are for a laptop with a 16-core CPU and an RTX 5070 Ti. `--tuning-from` reuses the settings chosen in a development run (in `experiments/dev/`), exactly as the reported test runs did; without it the tuning stage runs first (hours for the neural models). `xag models` lists every model, and `xag --help` every command.

## What is where

| Path | Contents |
|---|---|
| `xaglab/` | The Python package: data (`data/`), features (`features/`), models (`models/`, AIM-DG in `models/aimdg/`), evaluation and statistics (`eval/`), web API (`api/`), command line (`cli.py`) |
| `tests/` | Unit, leakage, integrity, statistical, model and API tests |
| `data/snapshots/` | The frozen data: `dissertation_v1` (market and FRED series, 3 Jan 2006 to 30 Jun 2026) and `calendar_v1` (709 CPI, payroll and FOMC release dates), each with a checksum manifest |
| `experiments/folds_v1.json` | The frozen walk-forward calendar |
| `experiments/runs/` | One folder per test run: forecasts (`predictions.parquet`), manifest (`run.json`), tuning log, timings, diagnostics |
| `experiments/dev/` | Development runs: the tuning stage only, used to choose designs |
| `experiments/reports/` | Report files (CSV) behind every table |
| `experiments/tables/`, `experiments/figures/` | The 27 result tables (CSV, Markdown, Word) and 33 figures (SVG, PNG), each with an index |
| `experiments/lab/` | Selection settings of the ablation and sensitivity study |
| `scripts/` | The verification and analysis scripts (planted regime break, search check, ablations, sensitivity) |
| `web/` | Source of the web front end; the built version is in `web/dist/` |

Code comments cite design decisions as `D-nn`; the dissertation's Chapter 3 and Appendix 3.C describe them.

## Data

Daily open, high, low and close prices of COMEX silver, gold and copper futures, the US dollar index and the VIX come from Yahoo Finance, and the 10-year real yield from FRED (Federal Reserve Bank of St. Louis). They were downloaded once and frozen on 31 July 2026; every experiment refuses to run if a file's checksum has changed. The data are included only so that the results can be reproduced; they remain subject to their providers' terms. No API key is needed to run anything here (one is needed only to download fresh data, see `.env.example`).

## Scope

The study uses public, aggregate market data only, involves no personal data and never trades. The signal-layer results test whether the forecasts carry usable information; they are not investment advice.

## Reference

Movahed, H., Toosi, T. and Nael, S. (2026) 'Adaptive graph-evolutionary framework for dynamic feature refinement in multi-label learning', *Scientific Reports*. doi: [10.1038/s41598-026-56255-5](https://doi.org/10.1038/s41598-026-56255-5).
