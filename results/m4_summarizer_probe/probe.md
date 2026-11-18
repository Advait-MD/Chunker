# 3b vs 7b RAPTOR summarizer probe (M4)

60 level-1 clusters from the seed-42 trees, summarised by both models with the same prompt, temperature 0, 256-token cap and trimming. *Unsupported details*: numbers and capitalised names in the summary that never appear in its children (a deterministic proxy; wrong relations between names that do appear are not counted).

| Dataset | n | Words 3b / 7b | Capped 3b / 7b | Unsupported details per summary 3b / 7b | Summaries with any, 3b / 7b | 7b s per summary |
| --- | --- | --- | --- | --- | --- | --- |
| qasper | 10 | 150 / 111 | 10% / 0% | 0.20 / 0.50 | 10% / 50% | 22.7 |
| quality | 10 | 137 / 108 | 0% / 0% | 0.20 / 0.20 | 20% / 20% | 29.9 |
| custom | 10 | 146 / 120 | 20% / 10% | 1.10 / 1.00 | 40% / 70% | 27.8 |
| musique | 10 | 136 / 148 | 10% / 60% | 0.70 / 1.10 | 40% / 50% | 33.5 |
| hotpotqa | 10 | 140 / 131 | 40% / 40% | 0.40 / 0.40 | 30% / 40% | 30.5 |
| 2wiki | 10 | 109 / 104 | 30% / 40% | 0.60 / 0.40 | 40% / 40% | 25.6 |
| all | 60 | 136 / 120 | 18% / 25% | 0.53 / 0.60 | 30% / 45% | 29.2 |

3b output matched the stored tree summary in 60 of 60 clusters.

