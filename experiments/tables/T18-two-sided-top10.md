**T18. Two-sided scorecard at the 10% coverage target**

| Model | Horizon | Days selected | Up calls | Up right | Up p | Down calls | Down right | Down p |
|---|---|---|---|---|---|---|---|---|
| climatology | 1d | 20.1% | 833 | 53.8% | 0.514 | 0 | – | – |
| ewma | 1d | 20.1% | 833 | 53.8% | 0.514 | 0 | – | – |
| ar_garch | 1d | 11.3% | 449 | 55.7% | 0.503 | 17 | 47.1% | 0.685 |
| linear | 1d | 17.0% | 528 | 58.5% | 0.042* | 174 | 56.9% | 0.040* |
| gbm | 1d | 17.6% | 562 | 58.2% | 0.113 | 167 | 53.3% | 0.220 |
| lstm | 1d | 14.7% | 495 | 55.2% | 0.464 | 114 | 46.5% | 0.800 |
| gru | 1d | 17.8% | 609 | 51.9% | 0.661 | 128 | 43.8% | 0.934 |
| static_gnn | 1d | 16.0% | 564 | 55.0% | 0.534 | 99 | 44.4% | 0.886 |
| aimdg | 1d | 15.1% | 552 | 53.6% | 0.483 | 73 | 47.9% | 0.680 |
| climatology | 5d | 19.3% | 801 | 53.9% | 0.551 | 0 | – | – |
| ewma | 5d | 19.3% | 801 | 53.9% | 0.551 | 0 | – | – |
| ar_garch | 5d | 12.3% | 508 | 51.0% | 0.579 | 0 | – | – |
| linear | 5d | 18.7% | 685 | 55.0% | 0.502 | 89 | 50.6% | 0.500 |
| gbm | 5d | 14.5% | 441 | 57.4% | 0.654 | 161 | 39.1% | 0.892 |
| lstm | 5d | 14.0% | 497 | 55.9% | 0.526 | 81 | 48.1% | 0.598 |
| gru | 5d | 11.5% | 354 | 52.5% | 0.525 | 124 | 47.6% | 0.729 |
| static_gnn | 5d | 13.6% | 435 | 58.2% | 0.366 | 129 | 50.4% | 0.500 |
| aimdg | 5d | 12.4% | 376 | 56.1% | 0.303 | 136 | 58.1% | 0.221 |

*Confident days: the forecast's distance from 50% is strictly above the matching quantile of the model's previous 252 forecasts (causal gate, D-31). Because forecast confidence drifts over time, a gate set on the past selects more days than its target (Days selected). p: one-sided binomial test of the share right against the stricter of 50% and that side's base rate on the selected days, with n/h effective trials. * p < 0.05.*

Source: `phase4-aimdg/selective.csv`
