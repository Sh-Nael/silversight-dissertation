**T17. Hit rate on the most confident days, 5-day horizon**

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

*Confident days: the forecast's distance from 50% is strictly above the matching quantile of the model's previous 252 forecasts (causal gate, D-31). Because forecast confidence drifts over time, a gate set on the past selects more days than its target: the 10% target selected 11.5–18.7% of days for the learned models, 20% 21.4–26.8%, 30% 29.4–34.9%, 50% 45.3–52.6%. Cells: hit rate (edge over always-up on the same days, in points); * one-sided block permutation p < 0.05 for a positive edge. Climatology and EWMA always call up.*

Source: `phase4-aimdg/selective.csv`
