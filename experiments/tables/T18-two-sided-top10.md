**T18. Buy calls and sell calls judged separately, 10% coverage target**

| Model | Days ahead | Share of days | Buy calls | Buy calls correct | p (buy) | Sell calls | Sell calls correct | p (sell) |
|---|---|---|---|---|---|---|---|---|
| climatology | 1 | 20.1% | 833 | 53.8% | 0.514 | 0 | – | – |
| ewma | 1 | 20.1% | 833 | 53.8% | 0.514 | 0 | – | – |
| ar_garch | 1 | 11.3% | 449 | 55.7% | 0.503 | 17 | 47.1% | 0.685 |
| linear | 1 | 17.0% | 528 | 58.5% | 0.042* | 174 | 56.9% | 0.040* |
| gbm | 1 | 17.6% | 562 | 58.2% | 0.113 | 167 | 53.3% | 0.220 |
| lstm | 1 | 14.7% | 495 | 55.2% | 0.464 | 114 | 46.5% | 0.800 |
| gru | 1 | 17.8% | 609 | 51.9% | 0.661 | 128 | 43.8% | 0.934 |
| static_gnn | 1 | 16.0% | 564 | 55.0% | 0.534 | 99 | 44.4% | 0.886 |
| aimdg | 1 | 15.1% | 552 | 53.6% | 0.483 | 73 | 47.9% | 0.680 |
| climatology | 5 | 19.3% | 801 | 53.9% | 0.551 | 0 | – | – |
| ewma | 5 | 19.3% | 801 | 53.9% | 0.551 | 0 | – | – |
| ar_garch | 5 | 12.3% | 508 | 51.0% | 0.579 | 0 | – | – |
| linear | 5 | 18.7% | 685 | 55.0% | 0.502 | 89 | 50.6% | 0.500 |
| gbm | 5 | 14.5% | 441 | 57.4% | 0.654 | 161 | 39.1% | 0.892 |
| lstm | 5 | 14.0% | 497 | 55.9% | 0.526 | 81 | 48.1% | 0.598 |
| gru | 5 | 11.5% | 354 | 52.5% | 0.525 | 124 | 47.6% | 0.729 |
| static_gnn | 5 | 13.6% | 435 | 58.2% | 0.366 | 129 | 50.4% | 0.500 |
| aimdg | 5 | 12.4% | 376 | 56.1% | 0.303 | 136 | 58.1% | 0.221 |

*A day counts as confident when the forecast sits further from 50% than the chosen quantile of the 252 forecasts the model made before it (D-31; ties do not count). Confidence tends to rise over the years, so a threshold taken from the past lets through more days than the target ('Share of days'). Each p-value is a one-sided binomial test of whether that kind of call was right more often than both 50% and the share of selected days that actually moved that way, counting n/h independent trials. A star marks p below 0.05.*

Source: `phase4-aimdg/selective.csv`
