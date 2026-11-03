## Gold set (4 PDFs): 27 answerable, 5 unanswerable

| Condition | Token F1 [95% CI] | ROUGE-L | Gold-chunk hit | Refused unanswerable | Context tok | Summaries used | p vs dense (F1) |
|---|---|---|---|---|---|---|---|
| apex_gated | 0.234 [0.174, 0.301] | 0.204 | 0.41 | 0.40 | 367 | 0.72 | 0.164 |
| apex_ungated | 0.219 [0.165, 0.278] | 0.188 | 0.37 | 0.00 | 394 | 0.91 | 0.112 |
| closed_book | 0.195 [0.146, 0.255] | 0.162 | 0.00 | 0.00 | 0 | 0.00 | 0.048 |
| dense_top5 | 0.270 [0.202, 0.341] | 0.230 | 0.52 | 0.80 | 698 | 0.00 | 0.644 |
| dense | 0.290 [0.209, 0.368] | 0.258 | 0.48 | 0.60 | 351 | 0.00 | — |
| tree_ungated | 0.207 [0.152, 0.271] | 0.186 | 0.44 | 0.40 | 365 | 1.00 | 0.027 |
| tree_gated | 0.207 [0.155, 0.267] | 0.185 | 0.48 | 0.60 | 364 | 0.94 | 0.030 |
| dense_srcfilter | 0.288 [0.206, 0.371] | 0.255 | 0.48 | 0.60 | 353 | 0.00 | 0.765 |

## QuALITY (24 multiple-choice questions, 3 full articles)

| Condition | Accuracy [95% CI] | Hard-question acc | Context tok | Summaries used | p vs dense |
|---|---|---|---|---|---|
| apex_gated | 0.250 [0.083, 0.417] | 0.33 | 599 | 0.50 | 1.000 |
| apex_ungated | 0.333 [0.167, 0.542] | 0.33 | 618 | 0.62 | 0.162 |
| closed_book | 0.167 [0.042, 0.333] | 0.08 | 0 | 0.00 | 0.328 |
| dense_top5 | 0.333 [0.167, 0.500] | 0.33 | 1561 | 0.00 | 0.162 |
| dense | 0.250 [0.083, 0.417] | 0.25 | 543 | 0.00 | — |
| tree_ungated | 0.250 [0.083, 0.417] | 0.25 | 556 | 0.62 | 1.000 |
| tree_gated | 0.250 [0.083, 0.417] | 0.25 | 555 | 0.46 | 1.000 |

Chance accuracy on QuALITY is 0.25. p-values: paired t-test against budget-matched dense.
