# `m4_seed_dev_s13` vs baseline `m4_seed_dev_row2`

Paired over shared questions: mean difference (A − B, in points for scores) with a 95% bootstrap CI, a two-sided paired-bootstrap p and a Wilcoxon signed-rank p.

| Dataset | Metric | n | A | B | Δ (95% CI) | p (bootstrap) | p (Wilcoxon) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qasper | f1 | 50 | 47.0 | 51.4 | -4.4 (-12.0, +2.1) | 0.212 | 0.322 |
| qasper | answer_in_context | 17 | 41.2 | 41.2 | +0.0 (+0.0, +0.0) | 1.000 | 1.000 |
| qasper | em | 50 | 22.0 | 24.0 | -2.0 (-10.0, +6.0) | 0.823 | 0.655 |
| qasper | acc | 50 | 26.0 | 32.0 | -6.0 (-16.0, +2.0) | 0.231 | 0.180 |
| qasper | recall@10 | 46 | 90.2 | 97.0 | -6.7 (-13.7, -1.1) | 0.012 | 0.041 |
| qasper | ndcg@10 | 46 | 46.6 | 53.9 | -7.3 (-10.8, -4.3) | 0.000 | 0.000 |
| qasper | prompt_tokens | 50 | 1706.2 | 1820.0 | -113.8 (-171.1, -55.8) | 0.000 | 0.001 |
| quality | f1 | 50 | 60.6 | 64.5 | -3.9 (-16.8, +8.5) | 0.559 | 0.491 |
| quality | em | 50 | 56.0 | 60.0 | -4.0 (-16.0, +8.0) | 0.647 | 0.527 |
| quality | acc | 50 | 66.0 | 74.0 | -8.0 (-20.0, +4.0) | 0.270 | 0.206 |
| quality | mc_acc | 50 | 72.0 | 74.0 | -2.0 (-14.0, +10.0) | 0.896 | 0.739 |
| quality | prompt_tokens | 50 | 2253.1 | 2530.1 | -277.0 (-325.4, -230.7) | 0.000 | 0.000 |
