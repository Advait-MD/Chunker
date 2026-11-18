# `m4_dev_raptor7b` vs baseline `m4_seed_dev_row2`

Paired over shared questions: mean difference (A − B, in points for scores) with a 95% bootstrap CI, a two-sided paired-bootstrap p and a Wilcoxon signed-rank p.

| Dataset | Metric | n | A | B | Δ (95% CI) | p (bootstrap) | p (Wilcoxon) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qasper | f1 | 50 | 50.9 | 51.4 | -0.5 (-8.4, +6.9) | 0.901 | 0.948 |
| qasper | answer_in_context | 17 | 41.2 | 41.2 | +0.0 (+0.0, +0.0) | 1.000 | 1.000 |
| qasper | em | 50 | 24.0 | 24.0 | +0.0 (-8.0, +8.0) | 1.000 | 1.000 |
| qasper | acc | 50 | 30.0 | 32.0 | -2.0 (-10.0, +4.0) | 0.780 | 0.564 |
| qasper | recall@10 | 46 | 88.0 | 97.0 | -8.9 (-16.7, -2.2) | 0.004 | 0.026 |
| qasper | ndcg@10 | 46 | 43.6 | 53.9 | -10.2 (-14.8, -6.3) | 0.000 | 0.000 |
| qasper | prompt_tokens | 50 | 1612.7 | 1820.0 | -207.2 (-271.2, -146.0) | 0.000 | 0.000 |
| quality | f1 | 50 | 58.4 | 64.5 | -6.2 (-18.2, +6.0) | 0.324 | 0.377 |
| quality | em | 50 | 54.0 | 60.0 | -6.0 (-18.0, +6.0) | 0.452 | 0.366 |
| quality | acc | 50 | 68.0 | 74.0 | -6.0 (-18.0, +6.0) | 0.405 | 0.317 |
| quality | mc_acc | 50 | 72.0 | 74.0 | -2.0 (-12.0, +8.0) | 0.858 | 0.705 |
| quality | prompt_tokens | 50 | 2220.9 | 2530.1 | -309.2 (-375.0, -247.2) | 0.000 | 0.000 |
