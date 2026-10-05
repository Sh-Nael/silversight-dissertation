**T22. Ablations and sensitivity of the selection settings (development data)**

| Setting | Joint loss | Diff. vs default | Points better | Wilcoxon p | Edges on | Edges moved |
|---|---|---|---|---|---|---|
| default | -4.7471 | +0.0000 | 0/17 | – | 2.75 | 0.00 |
| A1_no_evolution_top3 | -4.7431 | +0.0040 | 10/17 | 0.611 | 3.00 | 4.02 |
| A2_no_variance_penalty | -4.7442 | +0.0029 | 5/17 | 0.105 | 2.57 | 1.59 |
| A3_no_sparsity | -4.7479 | -0.0008 | 10/17 | 0.130 | 3.02 | 1.61 |
| all_edges_on | -4.7483 | -0.0013 | 12/17 | 0.225 | 15.00 | 12.25 |
| exhaustive_optimum | -4.7443 | +0.0028 | 4/17 | 0.031* | 2.24 | 1.33 |
| pop_20 | -4.7450 | +0.0021 | 7/17 | 0.562 | 2.94 | 1.76 |
| pop_100 | -4.7444 | +0.0027 | 4/17 | 0.093 | 2.37 | 1.24 |
| gens_30 | -4.7460 | +0.0011 | 4/17 | 0.241 | 3.27 | 1.55 |
| gens_300 | -4.7436 | +0.0035 | 2/17 | 0.004* | 2.37 | 1.04 |
| rate_0.15 | -4.7446 | +0.0025 | 5/17 | 0.098 | 2.22 | 1.39 |
| rate_0.5 | -4.7455 | +0.0016 | 10/17 | 0.404 | 3.20 | 1.86 |
| alpha_beta_1 | -4.7435 | +0.0035 | 4/17 | 0.015* | 2.55 | 1.25 |
| alpha_beta_0.25 | -4.7460 | +0.0010 | 8/17 | 0.644 | 2.65 | 1.31 |
| subwindow_10 | -4.7450 | +0.0021 | 6/17 | 0.159 | 2.61 | 1.51 |
| subwindow_42 | -4.7474 | -0.0004 | 9/17 | 0.669 | 2.65 | 1.24 |
| lam3_0.002 | -4.7438 | +0.0033 | 8/17 | 0.120 | 2.73 | 1.24 |
| lam3_0.01 | -4.7457 | +0.0014 | 9/17 | 0.644 | 2.45 | 1.63 |
| lam3_0.02 | -4.7465 | +0.0005 | 7/17 | 0.678 | 2.10 | 1.43 |
| top5_by_reliability | -4.7443 | +0.0028 | 9/17 | 0.854 | 5.00 | 5.39 |
| top8_by_reliability | -4.7480 | -0.0009 | 11/17 | 0.517 | 8.00 | 7.41 |

*Selection lab on cached networks at the 17 yearly tuning points (D-38). Joint loss: logloss1 + logloss5 + ½(QLIKE1 + QLIKE5), out of fold. Edges moved: edges that differ from the default's selection, per point. Development scores, not test performance. * p < 0.05.*

Source: `phase5-lab/lab_summary.csv`
