**T16. Hit rate on the most confident days, 1-day horizon**

| Model | Target 10% | Target 20% | Target 30% | Target 50% | All days |
|---|---|---|---|---|---|
| climatology | 53.8% (+0.0) | 55.4% (+0.0) | 54.2% (+0.0) | 54.0% (+0.0) | 51.8% (+0.0) |
| ewma | 53.8% (+0.0) | 55.4% (+0.0) | 54.2% (+0.0) | 54.0% (+0.0) | 51.8% (+0.0) |
| ar_garch | 55.4% (-0.2) | 54.4% (+0.9) | 54.1% (+0.2) | 53.3% (-0.9) | 51.8% (-0.1) |
| linear | 58.1% (+3.4) | 57.0% (-0.4) | 55.3% (-0.4) | 54.4% (+0.5) | 52.1% (+0.2) |
| gbm | 57.1% (+1.5) | 56.2% (+2.5) | 54.9% (+2.2)* | 54.6% (+1.8) | 53.8% (+2.0)* |
| lstm | 53.5% (-1.3) | 54.6% (+0.5) | 54.3% (+0.6) | 52.4% (-0.1) | 51.2% (-0.6) |
| gru | 50.5% (-2.2) | 52.5% (+0.0) | 52.9% (-0.1) | 52.9% (-0.2) | 51.1% (-0.8) |
| static_gnn | 53.4% (-1.7) | 53.2% (-0.6) | 53.9% (+0.1) | 53.5% (+0.7) | 50.4% (-1.4) |
| aimdg | 53.0% (-0.5) | 52.6% (+0.5) | 52.6% (-0.5) | 51.3% (-1.7) | 50.8% (-1.0) |

*Confident days: the forecast's distance from 50% is strictly above the matching quantile of the model's previous 252 forecasts (causal gate, D-31). Because forecast confidence drifts over time, a gate set on the past selects more days than its target: the 10% target selected 11.3–17.8% of days for the learned models, 20% 20.6–28.6%, 30% 30.1–36.9%, 50% 46.8–53.9%. Cells: hit rate (edge over always-up on the same days, in points); * one-sided block permutation p < 0.05 for a positive edge. Climatology and EWMA always call up.*

Source: `phase4-aimdg/selective.csv`
