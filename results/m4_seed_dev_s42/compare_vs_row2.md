# `m4_seed_dev_s42` vs baseline `m4_seed_dev_row2`

Paired over shared questions: mean difference (A − B, in points for scores) with a 95% bootstrap CI, a two-sided paired-bootstrap p and a Wilcoxon signed-rank p.

| Dataset | Metric | n | A | B | Δ (95% CI) | p (bootstrap) | p (Wilcoxon) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qasper | f1 | 50 | 44.8 | 51.4 | -6.7 (-14.3, -0.2) | 0.042 | 0.098 |
| qasper | answer_in_context | 17 | 41.2 | 41.2 | +0.0 (+0.0, +0.0) | 1.000 | 1.000 |
| qasper | em | 50 | 22.0 | 24.0 | -2.0 (-10.0, +4.0) | 0.791 | 0.564 |
| qasper | acc | 50 | 26.0 | 32.0 | -6.0 (-14.0, +0.0) | 0.089 | 0.083 |
| qasper | recall@10 | 46 | 85.9 | 97.0 | -11.1 (-19.8, -3.3) | 0.001 | 0.016 |
| qasper | ndcg@10 | 46 | 45.1 | 53.9 | -8.8 (-12.6, -5.6) | 0.000 | 0.000 |
| qasper | prompt_tokens | 50 | 1659.6 | 1820.0 | -160.3 (-213.4, -107.5) | 0.000 | 0.000 |
| quality | f1 | 50 | 64.9 | 64.5 | +0.4 (-11.9, +12.7) | 0.936 | 0.935 |
| quality | em | 50 | 60.0 | 60.0 | +0.0 (-12.0, +12.0) | 1.000 | 1.000 |
| quality | acc | 50 | 76.0 | 74.0 | +2.0 (-8.0, +12.0) | 0.850 | 0.705 |
| quality | mc_acc | 50 | 74.0 | 74.0 | +0.0 (-10.0, +10.0) | 1.000 | 1.000 |
| quality | prompt_tokens | 50 | 2214.1 | 2530.1 | -316.0 (-368.3, -265.5) | 0.000 | 0.000 |
