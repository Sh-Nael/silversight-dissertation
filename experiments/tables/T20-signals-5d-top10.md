**T20. Signal and risk layer, 5-day trades at the 10% coverage target**

| Model | Side | Trades | Win rate | Profit factor | Mean net (bps) | p (mean > 0) | Total return | Max drawdown |
|---|---|---|---|---|---|---|---|---|
| linear | both | 774 | 53.4% | 1.19 | 29.8 | 0.111 | 31.2% | -6.7% |
| linear | buy | 685 | 53.4% | 1.17 | 27.5 | 0.148 | 21.1% | -7.4% |
| linear | sell | 89 | 52.8% | 1.42 | 47.4 | 0.126 | 9.3% | -1.6% |
| gbm | both | 602 | 50.7% | 1.25 | 37.5 | 0.107 | 22.6% | -8.1% |
| gbm | buy | 441 | 54.4% | 1.40 | 62.4 | 0.110 | 21.8% | -7.5% |
| gbm | sell | 161 | 40.4% | 0.76 | -30.6 | 0.674 | -0.4% | -4.9% |
| lstm | both | 578 | 52.8% | 1.25 | 37.7 | 0.152 | 2.0% | -9.1% |
| lstm | buy | 497 | 53.5% | 1.30 | 47.2 | 0.168 | 4.0% | -8.2% |
| lstm | sell | 81 | 48.1% | 0.83 | -20.8 | 0.749 | -2.0% | -2.9% |
| static_gnn | both | 564 | 54.3% | 1.30 | 41.2 | 0.063 | 9.1% | -10.4% |
| static_gnn | buy | 435 | 55.6% | 1.37 | 52.5 | 0.064 | 12.3% | -5.2% |
| static_gnn | sell | 129 | 49.6% | 1.03 | 2.9 | 0.469 | -2.8% | -8.3% |
| aimdg | both | 512 | 53.5% | 1.25 | 34.2 | 0.079 | 8.9% | -14.3% |
| aimdg | buy | 376 | 52.7% | 1.21 | 30.8 | 0.259 | 10.6% | -11.4% |
| aimdg | sell | 136 | 55.9% | 1.40 | 43.4 | 0.220 | -0.7% | -6.5% |
| ens_lin_gbm | both | 680 | 54.4% | 1.46 | 65.7 | 0.065 | 34.8% | -6.3% |
| ens_lin_gbm | buy | 553 | 55.7% | 1.51 | 78.3 | 0.063 | 27.5% | -4.6% |
| ens_lin_gbm | sell | 127 | 48.8% | 1.10 | 11.0 | 0.360 | 5.8% | -3.8% |
| ens_lin_gbm_sgnn | both | 654 | 54.0% | 1.40 | 57.9 | 0.063 | 34.8% | -4.3% |
| ens_lin_gbm_sgnn | buy | 522 | 55.4% | 1.45 | 68.8 | 0.049* | 27.9% | -3.6% |
| ens_lin_gbm_sgnn | sell | 132 | 48.5% | 1.14 | 14.9 | 0.295 | 5.4% | -3.6% |

*Rules fixed beforehand: entry at the close on confident days, stop 1σ, target 1.5σ of the forecast move, a day hitting both counts as the stop, 10 bps round trip, 1% risk per trade, one position at a time. p: one-sided test that the mean net return is positive. * p < 0.05. Signals: the causal gate of D-31, strictly above the threshold; the 10% target gave signals on 12.4–18.7% of the 4,141 test days (Trades, side 'both').*

Source: `signals-6.2/signals.csv`, `signals-ensemble/signals.csv`
