# `m4_seed_dev_s7` vs baseline `m4_seed_dev_row2`

Paired over shared questions: mean difference (A − B, in points for scores) with a 95% bootstrap CI, a two-sided paired-bootstrap p and a Wilcoxon signed-rank p.

| Dataset | Metric | n | A | B | Δ (95% CI) | p (bootstrap) | p (Wilcoxon) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qasper | f1 | 50 | 46.2 | 51.4 | -5.2 (-14.0, +2.8) | 0.222 | 0.741 |
| qasper | answer_in_context | 17 | 41.2 | 41.2 | +0.0 (+0.0, +0.0) | 1.000 | 1.000 |
| qasper | em | 50 | 24.0 | 24.0 | +0.0 (-10.0, +10.0) | 1.000 | 1.000 |
| qasper | acc | 50 | 30.0 | 32.0 | -2.0 (-12.0, +8.0) | 0.850 | 0.705 |
| qasper | recall@10 | 46 | 88.0 | 97.0 | -8.9 (-16.7, -2.2) | 0.004 | 0.026 |
| qasper | ndcg@10 | 46 | 45.3 | 53.9 | -8.5 (-12.4, -5.2) | 0.000 | 0.000 |
| qasper | prompt_tokens | 50 | 1670.0 | 1820.0 | -149.9 (-210.6, -91.1) | 0.000 | 0.000 |
| quality | f1 | 50 | 58.6 | 64.5 | -5.9 (-20.2, +8.1) | 0.422 | 0.372 |
| quality | em | 50 | 54.0 | 60.0 | -6.0 (-20.0, +8.0) | 0.498 | 0.405 |
| quality | acc | 50 | 64.0 | 74.0 | -10.0 (-22.0, +2.0) | 0.169 | 0.132 |
| quality | mc_acc | 50 | 64.0 | 74.0 | -10.0 (-22.0, +2.0) | 0.164 | 0.132 |
| quality | prompt_tokens | 50 | 2239.7 | 2530.1 | -290.4 (-351.9, -231.7) | 0.000 | 0.000 |
