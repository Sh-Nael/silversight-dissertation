**T01. How well each of the nine models forecast silver's volatility**

| Model | QLIKE (1-day) | QLIKE (5-day) | RMSE (1-day) | RMSE (5-day) | MZ R² (1-day) | MZ R² (5-day) |
|---|---|---|---|---|---|---|
| climatology | -6.653 | -5.046 | 0.0181 | 0.0292 | 0.000 | 0.000 |
| ewma | -6.925 | -5.287 | 0.0165 | 0.0247 | 0.033 | 0.110 |
| ar_garch | -6.940 | -5.306 | 0.0169 | 0.0251 | 0.029 | 0.094 |
| linear | -6.963 | **-5.315** | **0.0162** | 0.0237 | **0.081** | 0.165 |
| gbm | -6.945 | -5.310 | 0.0169 | 0.0246 | 0.028 | 0.111 |
| lstm | -6.951 | -5.308 | 0.0163 | 0.0235 | 0.051 | 0.207 |
| gru | -6.959 | -5.310 | 0.0165 | 0.0240 | 0.044 | 0.155 |
| static_gnn | **-6.974** | **-5.315** | 0.0163 | **0.0229** | 0.059 | **0.253** |
| aimdg | -6.953 | -5.300 | 0.0165 | 0.0240 | 0.052 | 0.199 |

*Scored over the 4,141 trading days from 4 Jan 2010 to 23 Jun 2026 (eight folds, refitted monthly). Smaller QLIKE and RMSE are better, a larger Mincer–Zarnowitz R² is better; bold marks the leader of each column, ties included.*

Source: `phase4-aimdg/scoreboard.csv`
