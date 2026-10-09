# SilverSight

Code, frozen data and every result behind my MSc dissertation (module 7CS077, University of Wolverhampton, 2026) on forecasting silver with an adaptive graph of related markets.

**Shamim Nael** · student 2464263 · supervised by Dr Ajmal Shahbaz

## In short

Silver is pulled by monetary forces (gold, the dollar, real interest rates), by industry (copper) and by market fear (the VIX). The study asks whether a model that *re-chooses every month* which of those markets to listen to beats one that always listens to all of them.

The model, AIM-DG, connects silver to the five markets at three look-back lengths (1, 5 and 21 trading days), which gives 15 possible links. Once a month a genetic algorithm decides which links stay on, using the reliability scoring of PG-GFE, the feature-refinement method I co-developed (Movahed, Toosi and Nael, 2026). Eight other models act as yardsticks, from a constant long-run forecast to a graph with all 15 links permanently on. Everything is scored on 4,141 trading days held back for testing (January 2010 to June 2026) in eight blocks, with monthly retraining.

## What came out

| Question | Answer | Tables |
|---|---|---|
| Can silver's volatility be forecast? | Yes. All eight models improve on the constant forecast (Diebold–Mariano p < 0.001); the all-links graph does best one day ahead (QLIKE −6.974) | T01, T04, T08 |
| Can its direction be forecast? | Hardly. One-day accuracy ranges from 50.4% to 53.8%, while "always up" would score 51.8%, and no model's Brier score is reliably better than the constant forecast | T02, T03, T05 |
| Does monthly re-selection help? | No. AIM-DG forecasts volatility significantly worse than the all-links graph (p 0.006 at one day, 0.018 at five) and direction no better | T07, T15 |
| Why not? | Which link is useful changes from month to month mostly by chance, so tuning the choice harder backfires, and the networks are slow to learn a new relationship | T13, T14, T22, T23 |
| Any practical use? | Some. Calling only the most confident days lifts accuracy to about 57–58% for the simpler models, and a fixed stop/target rule on those days earns money after costs with the linear model, whose buys and sells were each profitable (profit factor 1.41, p 0.006). Without that filter, trading daily loses or merely breaks even | T16–T21 |
| Do the chosen links track market regimes? | Not in a way the tests can confirm; only the one-day VIX link stays on for long | T24–T26 |

The full list of 27 tables is in [`experiments/tables/index.md`](experiments/tables/index.md); the 33 charts are listed in [`experiments/figures/index.md`](experiments/figures/index.md).

## Getting it running

You need Python 3.13 and roughly 2 GB free (mostly for PyTorch). I developed and tested it on Linux; macOS should behave the same. A graphics card is only needed if you want to retrain the neural networks.

### Install

[uv](https://docs.astral.sh/uv/) is the easiest route, since it fetches Python 3.13 for you:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh          # once

git clone https://github.com/Sh-Nael/silversight-dissertation.git
cd silversight-dissertation

uv venv --python 3.13 .venv
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv -e ".[dev]"
source .venv/bin/activate                                 # Windows: .venv\Scripts\activate
```

That installs the CPU build of PyTorch. For an NVIDIA card, take the install line for your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/) instead (for example `--index-url https://download.pytorch.org/whl/cu130`).

No uv? Make a Python 3.13 virtual environment yourself (`python3.13 -m venv .venv`), activate it, and run the same two installs with plain `pip install`.

### Check it

```bash
xag verify      # confirms that no data file has changed since it was frozen
pytest -q       # 126 tests; about two minutes on a laptop processor
```

### Look at the results

```bash
xag ui
```

and open **http://localhost:8000** in a browser. The app only listens on your own machine. It reads the stored runs and has pages for an overview, every run with its settings, side-by-side comparisons with significance tests, calibration and confident-day accuracy, the trading rule, the test calendar, the input data and features, and the links AIM-DG picked month by month. Any chart can be downloaded as SVG or PNG, and the API is described at http://localhost:8000/docs. Press Ctrl+C to stop it.

### Rebuild tables and checks

```bash
xag tables      # rewrites experiments/tables/ from the stored report files
xag report linear-20260926-100937 gbm-20260926-100959 static_gnn-20260927-183956 \
           aimdg-20260928-120236 climatology-20260926-091404 --reference static_gnn
python scripts/spring2013_gap.py      # the spring-2013 breakdown (dissertation Table E.2)
```

`xag report` recalculates scores and significance tests from the saved forecasts in a few seconds.

To redraw all 33 charts in one go you additionally need Node.js 22 or newer, with the app running:

```bash
xag ui &
cd web && npm ci && npx playwright install chromium
npm run figures
```

### Repeat the experiments

New runs go into new folders under `experiments/runs/`; the stored ones are left alone.

```bash
xag run climatology ewma ar_garch                 # seconds
xag run linear gbm --jobs 8                       # a few minutes, tuning included
xag run lstm --tuning-from lstm-dev-r2-20260927-113901 --jobs 6              # GPU, ~25 min
xag run static_gnn --tuning-from static_gnn-dev-20260927-164105 --jobs 6     # GPU, ~1.5 h
xag run aimdg --tuning-from aimdg-dev-20260928-090733                        # GPU, ~2.5 h
```

The timings come from a 16-core laptop with an RTX 5070 Ti. `--tuning-from` re-uses the hyperparameters found in a development run under `experiments/dev/`, which is how the reported runs were produced; leave it out and tuning runs first, which takes hours for the networks. `xag models` and `xag --help` list everything else.

## Where things are

- `xaglab/` — the Python package (data loading, features, all models with AIM-DG under `models/aimdg/`, scoring and statistics, the web API and the `xag` command)
- `tests/` — the automated checks, including the test that altering future data cannot change any past feature
- `data/snapshots/` — the frozen inputs (`dissertation_v1`: prices and yields, Jan 2006 – Jun 2026; `calendar_v1`: 709 CPI, payroll and FOMC dates), each with SHA-256 checksums
- `experiments/folds_v1.json` — the fixed test calendar
- `experiments/runs/` — one folder per test run with its daily forecasts, settings, tuning record, timings and diagnostics; each run notes the commit of the private working repository it came from, of which this repository is a cleaned-up copy
- `experiments/dev/` — tuning-only development runs
- `experiments/reports/`, `experiments/tables/`, `experiments/figures/` — raw report files, the formatted tables and the charts
- `experiments/lab/` — the selection settings used for the ablations and sensitivity runs
- `scripts/` — stand-alone checks and analyses (planted regime change, search quality, ablations, sensitivity, spring 2013)
- `web/` — the browser front end; `web/dist/` holds the ready-built version

Design decisions are referred to in code comments as `D-nn`; the dissertation explains them in Chapter 3 and lists them in Appendix C.

## Data and its use

Daily prices of COMEX silver, gold and copper futures plus the dollar index and the VIX come from Yahoo Finance; the 10-year inflation-protected Treasury yield comes from FRED. Both were downloaded once and frozen on 31 July 2026. They are included only so that the results can be checked, and they remain under their providers' terms. Nothing here needs an API key; one is required only to download fresh data (see `.env.example`).

The project uses public market data only, involves no personal information and never places trades. The trading-rule results show whether the forecasts contain usable information; they are not investment advice.

## Related paper

H. Movahed, T. Toosi and S. Nael, "Adaptive graph-evolutionary framework for dynamic feature refinement in multi-label learning", *Scientific Reports* 16, 27862 (2026). https://doi.org/10.1038/s41598-026-56255-5
