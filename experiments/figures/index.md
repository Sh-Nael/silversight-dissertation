# Figure pack

Exported by `npm run figures` (web/scripts/figures.mjs) from the running app, in the print palette.
SVG is the vector file (Word, Inkscape, LaTeX); PNG is 2400 px wide (300 dpi at 20 cm).
Figure numbers are working labels; renumber when placing a figure in the report. The captions are
descriptions of what the chart shows.

Exported 2026-10-01.

| Figure | Shows | Suggested chapter | Models | App page and chart | Files |
|---|---|---|---|---|---|
| F01 | Silver weekly close with the four regime windows and the start of each test fold | 1, 3 | – | `/` · silver-price-regimes | [SVG](F01-silver-price-regimes.svg) · [PNG](F01-silver-price-regimes.png) |
| F02 | The frozen walk-forward calendar: training spans, test blocks and tuning points | 3 | – | `/folds` · walk-forward-timeline | [SVG](F02-walk-forward-calendar.svg) · [PNG](F02-walk-forward-calendar.png) |
| F03 | Gold–silver error-correction term (z-score of the log ratio) | 3 | – | `/data` · gold-silver-ect | [SVG](F03-gold-silver-ect.svg) · [PNG](F03-gold-silver-ect.png) |
| F04 | QLIKE per fold, 1 day: each model minus the static graph (below zero: better) | 4 | linear, gbm, lstm, gru, static_gnn, aimdg | `/compare` · per-fold-qlike-h1-vs-static_gnn | [SVG](F04-per-fold-qlike-1d-vs-static-gnn.svg) · [PNG](F04-per-fold-qlike-1d-vs-static-gnn.png) |
| F05 | Cumulative QLIKE difference against the static graph, 1 day | 4 | linear, gbm, lstm, gru, static_gnn, aimdg | `/compare` · cumulative-qlike-h1-vs-static_gnn | [SVG](F05-cumulative-qlike-1d-vs-static-gnn.svg) · [PNG](F05-cumulative-qlike-1d-vs-static-gnn.png) |
| F06 | AIM-DG minus the static graph, QLIKE per fold, 1 day | 4 | static_gnn, aimdg | `/compare` · per-fold-qlike-h1-vs-static_gnn | [SVG](F06-per-fold-qlike-1d-aimdg-vs-static-gnn.svg) · [PNG](F06-per-fold-qlike-1d-aimdg-vs-static-gnn.png) |
| F07 | AIM-DG minus the static graph, cumulative QLIKE difference, 1 day | 4 | static_gnn, aimdg | `/compare` · cumulative-qlike-h1-vs-static_gnn | [SVG](F07-cumulative-qlike-1d-aimdg-vs-static-gnn.svg) · [PNG](F07-cumulative-qlike-1d-aimdg-vs-static-gnn.png) |
| F08 | AIM-DG minus the static graph, cumulative QLIKE difference, 5 days | 4 | static_gnn, aimdg | `/compare` · cumulative-qlike-h5-vs-static_gnn | [SVG](F08-cumulative-qlike-5d-aimdg-vs-static-gnn.svg) · [PNG](F08-cumulative-qlike-5d-aimdg-vs-static-gnn.png) |
| F09 | Brier score per fold, 1 day: each model minus climatology | 4 | climatology, ar_garch, linear, gbm, lstm, gru, static_gnn, aimdg | `/compare` · per-fold-brier-h1-vs-climatology | [SVG](F09-per-fold-brier-1d-vs-climatology.svg) · [PNG](F09-per-fold-brier-1d-vs-climatology.png) |
| F10 | Cumulative Brier difference against climatology, 1 day | 4 | climatology, ar_garch, linear, gbm, lstm, gru, static_gnn, aimdg | `/compare` · cumulative-brier-h1-vs-climatology | [SVG](F10-cumulative-brier-1d-vs-climatology.svg) · [PNG](F10-cumulative-brier-1d-vs-climatology.png) |
| F11 | Cumulative Brier difference against climatology, 5 days | 4 | climatology, ar_garch, linear, gbm, lstm, gru, static_gnn, aimdg | `/compare` · cumulative-brier-h5-vs-climatology | [SVG](F11-cumulative-brier-5d-vs-climatology.svg) · [PNG](F11-cumulative-brier-5d-vs-climatology.png) |
| F12 | Ensembles and their members, cumulative QLIKE difference against gradient boosting, 1 day | 4 | linear, gbm, static_gnn, ens_lin_gbm, ens_lin_gbm_sgnn | `/compare` · cumulative-qlike-h1-vs-gbm | [SVG](F12-cumulative-qlike-1d-ensembles-vs-gbm.svg) · [PNG](F12-cumulative-qlike-1d-ensembles-vs-gbm.png) |
| F13 | Ablations A1 and A2 and the static graph, cumulative QLIKE difference against AIM-DG, 1 day | 4 | static_gnn, aimdg, aimdg_A1, aimdg_A2 | `/compare` · cumulative-qlike-h1-vs-aimdg | [SVG](F13-cumulative-qlike-1d-ablations-vs-aimdg.svg) · [PNG](F13-cumulative-qlike-1d-ablations-vs-aimdg.png) |
| F14 | Reliability diagram, 1-day direction | 4 | linear, gbm, static_gnn, aimdg | `/calibration` · reliability-h1 | [SVG](F14-reliability-1d.svg) · [PNG](F14-reliability-1d.png) |
| F15 | Reliability diagram, 5-day direction | 4 | linear, gbm, static_gnn, aimdg | `/calibration` · reliability-h5 | [SVG](F15-reliability-5d.svg) · [PNG](F15-reliability-5d.png) |
| F16 | Static graph: forecast against realised volatility, 1 day (annualised, weekly means) | 4 | linear, gbm, static_gnn, aimdg | `/calibration` · volatility-static_gnn-h1 | [SVG](F16-volatility-static-gnn-1d.svg) · [PNG](F16-volatility-static-gnn-1d.png) |
| F17 | AIM-DG: forecast against realised volatility, 1 day (annualised, weekly means) | 4 | linear, gbm, static_gnn, aimdg | `/calibration` · volatility-aimdg-h1 | [SVG](F17-volatility-aimdg-1d.svg) · [PNG](F17-volatility-aimdg-1d.png) |
| F18 | Hit rate against coverage, 1 day, all calls (causal confidence gate) | 4 | ar_garch, linear, gbm, lstm, gru, static_gnn, aimdg | `/calibration` · coverage-causal-h1 | [SVG](F18-coverage-1d-both.svg) · [PNG](F18-coverage-1d-both.png) |
| F19 | Share of up (buy) calls that were right against coverage, 1 day | 4 | ar_garch, linear, gbm, lstm, gru, static_gnn, aimdg | `/calibration` · coverage-causal-h1 | [SVG](F19-coverage-1d-buy.svg) · [PNG](F19-coverage-1d-buy.png) |
| F20 | Share of down (sell) calls that were right against coverage, 1 day | 4 | ar_garch, linear, gbm, lstm, gru, static_gnn, aimdg | `/calibration` · coverage-causal-h1 | [SVG](F20-coverage-1d-sell.svg) · [PNG](F20-coverage-1d-sell.png) |
| F21 | Hit rate against coverage, 5 days, all calls (causal confidence gate) | 4 | ar_garch, linear, gbm, lstm, gru, static_gnn, aimdg | `/calibration` · coverage-causal-h5 | [SVG](F21-coverage-5d-both.svg) · [PNG](F21-coverage-5d-both.png) |
| F22 | Account equity of the signal and risk layer, 1-day buy and sell trades on the 10% most confident days | 4 | linear, gbm, static_gnn, aimdg, ens_lin_gbm_sgnn | `/signals` · equity-both-h1-c0.1 | [SVG](F22-equity-1d-top10-both.svg) · [PNG](F22-equity-1d-top10-both.png) |
| F23 | Account equity of the signal and risk layer, 1-day buy trades on the 10% most confident days | 4 | linear, gbm, static_gnn, aimdg, ens_lin_gbm_sgnn | `/signals` · equity-buy-h1-c0.1 | [SVG](F23-equity-1d-top10-buy.svg) · [PNG](F23-equity-1d-top10-buy.png) |
| F24 | Account equity of the signal and risk layer, 1-day sell trades on the 10% most confident days | 4 | linear, gbm, static_gnn, aimdg, ens_lin_gbm_sgnn | `/signals` · equity-sell-h1-c0.1 | [SVG](F24-equity-1d-top10-sell.svg) · [PNG](F24-equity-1d-top10-sell.png) |
| F25 | Account equity when trading every day, 1-day trades (the gate switched off) | 4 | linear, gbm, static_gnn, aimdg, ens_lin_gbm_sgnn | `/signals` · equity-both-h1-c1 | [SVG](F25-equity-1d-all-days.svg) · [PNG](F25-equity-1d-all-days.png) |
| F26 | Which of the 15 edges AIM-DG kept at each monthly selection, with the regime windows | 5 | – | `/graph` · edge-timeline-on | [SVG](F26-edge-timeline-active.svg) · [PNG](F26-edge-timeline-active.png) |
| F27 | Reliability of each edge at each monthly selection | 5 | – | `/graph` · edge-timeline-reliability | [SVG](F27-edge-timeline-reliability.svg) · [PNG](F27-edge-timeline-reliability.png) |
| F28 | Active share of each cluster's edges in stress months against the rest | 5 | – | `/graph` · cluster-activity-stress | [SVG](F28-cluster-activity-stress.svg) · [PNG](F28-cluster-activity-stress.png) |
| F29 | The driver graph AIM-DG selected in the 2011 silver spike (2011-04) | 5 | – | `/graph` · driver-graph-2011-04-04 | [SVG](F29-driver-graph-2011-04.svg) · [PNG](F29-driver-graph-2011-04.png) |
| F30 | The driver graph AIM-DG selected in the 2013 taper tantrum (2013-06) | 5 | – | `/graph` · driver-graph-2013-06-26 | [SVG](F30-driver-graph-2013-06.svg) · [PNG](F30-driver-graph-2013-06.png) |
| F31 | The driver graph AIM-DG selected in the 2020 COVID shock (2020-03) | 5 | – | `/graph` · driver-graph-2020-03-04 | [SVG](F31-driver-graph-2020-03.svg) · [PNG](F31-driver-graph-2020-03.png) |
| F32 | The driver graph AIM-DG selected in the 2022 rate shock (2022-09) | 5 | – | `/graph` · driver-graph-2022-09-12 | [SVG](F32-driver-graph-2022-09.svg) · [PNG](F32-driver-graph-2022-09.png) |
| F33 | The driver graph AIM-DG selected in the last selection of the test period (2026-06) | 5 | – | `/graph` · driver-graph-2026-06-04 | [SVG](F33-driver-graph-2026-06.svg) · [PNG](F33-driver-graph-2026-06.png) |
