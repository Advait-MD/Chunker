# `m4_seed_dev_s7` vs baseline `m4_seed_dev_s42`

Paired over shared questions: mean difference (A − B, in points for scores) with a 95% bootstrap CI, a two-sided paired-bootstrap p and a Wilcoxon signed-rank p.

| Dataset | Metric | n | A | B | Δ (95% CI) | p (bootstrap) | p (Wilcoxon) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qasper | f1 | 50 | 46.2 | 44.8 | +1.5 (-4.7, +7.0) | 0.585 | 0.177 |
| qasper | answer_in_context | 17 | 41.2 | 41.2 | +0.0 (+0.0, +0.0) | 1.000 | 1.000 |
| qasper | em | 50 | 24.0 | 22.0 | +2.0 (-6.0, +10.0) | 0.828 | 0.655 |
| qasper | acc | 50 | 30.0 | 26.0 | +4.0 (-6.0, +14.0) | 0.528 | 0.414 |
| qasper | recall@10 | 46 | 88.0 | 85.9 | +2.2 (+0.0, +6.5) | 0.737 | 0.317 |
| qasper | ndcg@10 | 46 | 45.3 | 45.1 | +0.2 (-2.6, +2.6) | 0.809 | 0.445 |
| qasper | prompt_tokens | 50 | 1670.0 | 1659.6 | +10.4 (-29.0, +50.1) | 0.611 | 0.744 |
| quality | f1 | 50 | 58.6 | 64.9 | -6.4 (-16.4, +3.6) | 0.234 | 0.305 |
| quality | em | 50 | 54.0 | 60.0 | -6.0 (-16.0, +4.0) | 0.343 | 0.257 |
| quality | acc | 50 | 64.0 | 76.0 | -12.0 (-22.0, -4.0) | 0.005 | 0.014 |
| quality | mc_acc | 50 | 64.0 | 74.0 | -10.0 (-20.0, +0.0) | 0.064 | 0.059 |
| quality | prompt_tokens | 50 | 2239.7 | 2214.1 | +25.6 (-24.4, +73.3) | 0.309 | 0.119 |
