# `m4_seed_dev_s13` vs baseline `m4_seed_dev_s42`

Paired over shared questions: mean difference (A − B, in points for scores) with a 95% bootstrap CI, a two-sided paired-bootstrap p and a Wilcoxon signed-rank p.

| Dataset | Metric | n | A | B | Δ (95% CI) | p (bootstrap) | p (Wilcoxon) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qasper | f1 | 50 | 47.0 | 44.8 | +2.3 (-0.3, +5.1) | 0.084 | 0.112 |
| qasper | answer_in_context | 17 | 41.2 | 41.2 | +0.0 (+0.0, +0.0) | 1.000 | 1.000 |
| qasper | em | 50 | 22.0 | 22.0 | +0.0 (-6.0, +6.0) | 1.000 | 1.000 |
| qasper | acc | 50 | 26.0 | 26.0 | +0.0 (-6.0, +6.0) | 1.000 | 1.000 |
| qasper | recall@10 | 46 | 90.2 | 85.9 | +4.3 (+0.0, +10.9) | 0.269 | 0.157 |
| qasper | ndcg@10 | 46 | 46.6 | 45.1 | +1.5 (+0.2, +3.2) | 0.016 | 0.085 |
| qasper | prompt_tokens | 50 | 1706.2 | 1659.6 | +46.6 (+9.0, +83.8) | 0.017 | 0.017 |
| quality | f1 | 50 | 60.6 | 64.9 | -4.4 (-12.0, +3.1) | 0.274 | 0.336 |
| quality | em | 50 | 56.0 | 60.0 | -4.0 (-12.0, +4.0) | 0.440 | 0.317 |
| quality | acc | 50 | 66.0 | 76.0 | -10.0 (-20.0, +0.0) | 0.068 | 0.059 |
| quality | mc_acc | 50 | 72.0 | 74.0 | -2.0 (-10.0, +6.0) | 0.844 | 0.655 |
| quality | prompt_tokens | 50 | 2253.1 | 2214.1 | +39.0 (+4.0, +74.0) | 0.028 | 0.061 |
