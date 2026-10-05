**T21. Signal and risk layer, 1-day trades on every day**

| Model | Side | Trades | Win rate | Profit factor | Mean net (bps) | p (mean > 0) | Total return | Max drawdown |
|---|---|---|---|---|---|---|---|---|
| linear | both | 4,141 | 48.4% | 1.03 | 2.0 | 0.205 | -28.9% | -62.9% |
| linear | buy | 2,645 | 49.8% | 1.08 | 5.1 | 0.057 | -8.4% | -35.2% |
| linear | sell | 1,496 | 45.9% | 0.94 | -3.4 | 0.930 | -6.6% | -22.4% |
| gbm | both | 4,141 | 50.4% | 1.08 | 4.9 | 0.029* | -0.4% | -51.0% |
| gbm | buy | 2,656 | 51.4% | 1.13 | 7.9 | 0.016* | 45.7% | -26.4% |
| gbm | sell | 1,485 | 48.6% | 0.99 | -0.5 | 0.651 | 1.0% | -23.8% |
| lstm | both | 4,141 | 47.8% | 1.00 | -0.3 | 0.508 | -46.0% | -67.1% |
| lstm | buy | 3,054 | 49.1% | 1.05 | 3.4 | 0.143 | -8.3% | -37.7% |
| lstm | sell | 1,087 | 44.2% | 0.79 | -10.7 | 0.999 | -30.9% | -38.6% |
| static_gnn | both | 4,141 | 47.0% | 0.95 | -2.9 | 0.856 | -61.8% | -76.3% |
| static_gnn | buy | 2,983 | 48.7% | 1.02 | 1.2 | 0.351 | -28.2% | -52.3% |
| static_gnn | sell | 1,158 | 42.7% | 0.76 | -13.6 | 1.000 | -35.5% | -39.7% |
| aimdg | both | 4,141 | 47.3% | 0.96 | -2.8 | 0.848 | -67.4% | -79.9% |
| aimdg | buy | 2,891 | 48.6% | 1.02 | 1.1 | 0.297 | -19.4% | -42.4% |
| aimdg | sell | 1,250 | 44.3% | 0.79 | -11.8 | 0.999 | -48.8% | -49.2% |
| ens_lin_gbm | both | 4,141 | 49.0% | 1.03 | 2.1 | 0.173 | -42.2% | -69.3% |
| ens_lin_gbm | buy | 2,692 | 50.4% | 1.09 | 5.7 | 0.034* | 33.1% | -25.8% |
| ens_lin_gbm | sell | 1,449 | 46.5% | 0.91 | -4.7 | 0.930 | -27.1% | -43.5% |
| ens_lin_gbm_sgnn | both | 4,141 | 48.5% | 1.01 | 0.4 | 0.400 | -37.7% | -66.8% |
| ens_lin_gbm_sgnn | buy | 2,776 | 50.0% | 1.06 | 4.0 | 0.094 | -9.6% | -41.3% |
| ens_lin_gbm_sgnn | sell | 1,365 | 45.6% | 0.87 | -7.0 | 0.980 | -40.6% | -43.8% |

*Rules fixed beforehand: entry at the close on confident days, stop 1σ, target 1.5σ of the forecast move, a day hitting both counts as the stop, 10 bps round trip, 1% risk per trade, one position at a time. p: one-sided test that the mean net return is positive. * p < 0.05.*

Source: `signals-6.2/signals.csv`, `signals-ensemble/signals.csv`
