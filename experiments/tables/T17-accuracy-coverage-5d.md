**T17. Accuracy when a model calls only its most confident days (5 days ahead)**

| Model | Target 10% | Target 20% | Target 30% | Target 50% | All days |
|---|---|---|---|---|---|
| climatology | 53.9% (+0.0) | 54.4% (+0.0) | 54.7% (+0.0) | 53.6% (+0.0) | 51.9% (+0.0) |
| ewma | 53.9% (+0.0) | 54.4% (+0.0) | 54.7% (+0.0) | 53.6% (+0.0) | 51.9% (+0.0) |
| ar_garch | 51.0% (+0.0) | 52.4% (+0.0) | 52.7% (+0.0) | 51.6% (+0.0) | 52.0% (+0.1) |
| linear | 54.5% (+0.1) | 54.4% (-0.3) | 53.3% (-0.6) | 53.0% (+0.9) | 51.6% (-0.3) |
| gbm | 52.5% (-5.8) | 52.7% (-5.0) | 53.7% (-3.2) | 53.8% (+0.1) | 52.4% (+0.5) |
| lstm | 54.8% (-0.5) | 54.8% (-0.4) | 53.8% (-1.8) | 52.6% (+0.2) | 52.1% (+0.2) |
| gru | 51.3% (-1.3) | 52.1% (-0.4) | 52.7% (-0.1) | 51.9% (-1.0) | 50.3% (-1.6) |
| static_gnn | 56.4% (+0.2) | 54.7% (+0.0) | 54.4% (+0.1) | 52.6% (+0.9) | 51.5% (-0.4) |
| aimdg | 56.6% (+4.3) | 54.1% (+0.8) | 53.5% (+0.9) | 52.4% (-1.2) | 51.5% (-0.4) |

*A day counts as confident when the forecast sits further from 50% than the chosen quantile of the 252 forecasts the model made before it (D-31; ties do not count). Confidence tends to rise over the years, so a threshold taken from the past lets through more days than the target. For the fitted models the share of days actually used was 11.5–18.7% for the 10% target, 21.4–26.8% for 20%, 29.4–34.9% for 30% and 45.3–52.6% for 50%. Each cell gives the accuracy and, in brackets, its margin in points over always calling a rise on the same days; a star means a one-sided block-permutation p below 0.05. Climatology and EWMA only ever call a rise.*

Source: `phase4-aimdg/selective.csv`
