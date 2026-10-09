**T19. Trading the forecasts with a stop and a target: 1-day holding, confident days only (10% target)**

| Model | Side | Trades | Winning trades | Gain/loss ratio | Net per trade (bps) | p (net > 0) | Account return | Worst drawdown |
|---|---|---|---|---|---|---|---|---|
| linear | both | 702 | 55.6% | 1.41 | 25.9 | 0.006* | 92.5% | -5.8% |
| linear | buy | 528 | 55.9% | 1.40 | 29.5 | 0.003* | 68.8% | -6.4% |
| linear | sell | 174 | 54.6% | 1.44 | 14.9 | 0.037* | 14.2% | -4.9% |
| gbm | both | 729 | 53.6% | 1.36 | 22.1 | <0.001* | 39.2% | -7.3% |
| gbm | buy | 562 | 55.2% | 1.38 | 25.9 | <0.001* | 25.8% | -7.4% |
| gbm | sell | 167 | 48.5% | 1.24 | 9.6 | 0.358 | 8.3% | -5.8% |
| lstm | both | 609 | 50.2% | 1.12 | 8.4 | 0.141 | 17.5% | -13.3% |
| lstm | buy | 495 | 52.3% | 1.18 | 13.8 | 0.120 | 23.4% | -8.0% |
| lstm | sell | 114 | 41.2% | 0.67 | -14.9 | 0.781 | -4.8% | -9.2% |
| static_gnn | both | 663 | 50.2% | 0.96 | -3.1 | 0.582 | 4.8% | -15.5% |
| static_gnn | buy | 564 | 52.0% | 1.02 | 1.5 | 0.416 | 15.1% | -8.7% |
| static_gnn | sell | 99 | 40.4% | 0.53 | -29.5 | 0.938 | -9.0% | -11.4% |
| aimdg | both | 625 | 49.9% | 1.09 | 5.7 | 0.163 | -13.4% | -27.8% |
| aimdg | buy | 552 | 50.5% | 1.11 | 7.2 | 0.276 | -4.5% | -20.4% |
| aimdg | sell | 73 | 45.2% | 0.91 | -5.4 | 0.122 | -9.3% | -9.8% |
| ens_lin_gbm | both | 747 | 54.5% | 1.37 | 23.8 | <0.001* | 58.0% | -11.9% |
| ens_lin_gbm | buy | 582 | 56.5% | 1.43 | 30.0 | <0.001* | 59.2% | -5.8% |
| ens_lin_gbm | sell | 165 | 47.3% | 1.05 | 2.2 | 0.538 | -1.7% | -11.3% |
| ens_lin_gbm_sgnn | both | 716 | 54.9% | 1.41 | 27.3 | 0.002* | 74.8% | -10.1% |
| ens_lin_gbm_sgnn | buy | 576 | 56.2% | 1.44 | 31.8 | <0.001* | 70.8% | -5.2% |
| ens_lin_gbm_sgnn | sell | 140 | 49.3% | 1.22 | 8.6 | 0.124 | 2.4% | -8.4% |

*Trades open at the close of a confident day, exit at a stop one forecast standard deviation away or a target one and a half away, and a day that touches both is booked as a stop. Each trade costs 10 basis points in total and risks 1% of the account, with at most one position open. These rules were set before any result. The p-value tests, one-sided, whether the average net result per trade is above zero; a star marks p below 0.05. Signals come from the confidence gate of D-31; with the 10% target they fell on 14.7–18.0% of the 4,141 test days (Trades column, side 'both').*

Source: `signals-6.2/signals.csv`, `signals-ensemble/signals.csv`
