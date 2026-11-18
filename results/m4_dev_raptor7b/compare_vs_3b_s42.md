# `m4_dev_raptor7b` vs baseline `m4_seed_dev_s42`

Paired over shared questions: mean difference (A − B, in points for scores) with a 95% bootstrap CI, a two-sided paired-bootstrap p and a Wilcoxon signed-rank p.

| Dataset | Metric | n | A | B | Δ (95% CI) | p (bootstrap) | p (Wilcoxon) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qasper | f1 | 50 | 50.9 | 44.8 | +6.1 (-0.2, +13.1) | 0.057 | 0.035 |
| qasper | answer_in_context | 17 | 41.2 | 41.2 | +0.0 (+0.0, +0.0) | 1.000 | 1.000 |
| qasper | em | 50 | 24.0 | 22.0 | +2.0 (-4.0, +10.0) | 0.791 | 0.564 |
| qasper | acc | 50 | 30.0 | 26.0 | +4.0 (+0.0, +10.0) | 0.263 | 0.157 |
| qasper | recall@10 | 46 | 88.0 | 85.9 | +2.2 (+0.0, +6.5) | 0.737 | 0.317 |
| qasper | ndcg@10 | 46 | 43.6 | 45.1 | -1.5 (-5.2, +1.5) | 0.397 | 0.931 |
| qasper | prompt_tokens | 50 | 1612.7 | 1659.6 | -46.9 (-85.4, -8.1) | 0.016 | 0.006 |
| quality | f1 | 50 | 58.4 | 64.9 | -6.6 (-19.3, +6.0) | 0.297 | 0.312 |
| quality | em | 50 | 54.0 | 60.0 | -6.0 (-20.0, +8.0) | 0.478 | 0.405 |
| quality | acc | 50 | 68.0 | 76.0 | -8.0 (-20.0, +2.0) | 0.196 | 0.157 |
| quality | mc_acc | 50 | 72.0 | 74.0 | -2.0 (-12.0, +8.0) | 0.849 | 0.705 |
| quality | prompt_tokens | 50 | 2220.9 | 2214.1 | +6.8 (-55.1, +67.6) | 0.813 | 0.938 |
