# M5 threshold probe (dev, top 5 of row 4's reranked list)

## Relevance threshold θ_rel

Per cell: passages kept per question / token share kept / gold passages kept / non-gold passages dropped.

| θ_rel | 2wiki (n=50) | hotpotqa (n=50) | musique (n=50) | qasper (n=50) | quality (n=50) | all (n=250) |
| --- | --- | --- | --- | --- | --- | --- |
| off | 5.0 / 100% / 100% / 0% | 5.0 / 100% / 100% / 0% | 5.0 / 100% / 100% / 0% | 5.0 / 100% / 100% / 0% | 5.0 / 100% / — / 0% | 5.0 / 100% / 100% / 0% |
| -6.0 | 4.7 / 93% / 100% / 10% | 4.8 / 94% / 100% / 6% | 4.8 / 95% / 100% / 7% | 4.4 / 89% / 96% / 13% | 4.7 / 95% / — / 5% | 4.7 / 93% / 99% / 8% |
| -4.0 | 3.4 / 64% / 98% / 46% | 4.3 / 85% / 100% / 20% | 3.7 / 74% / 100% / 36% | 3.8 / 77% / 82% / 25% | 4.0 / 81% / — / 19% | 3.9 / 77% / 96% / 28% |
| -3.0 | 2.8 / 51% / 96% / 62% | 3.9 / 77% / 100% / 32% | 3.3 / 66% / 99% / 47% | 3.4 / 68% / 73% / 33% | 3.6 / 71% / — / 29% | 3.4 / 67% / 93% / 39% |
| -2.0 | 2.5 / 41% / 94% / 71% | 3.6 / 70% / 99% / 43% | 2.8 / 55% / 93% / 59% | 3.1 / 62% / 61% / 37% | 2.7 / 55% / — / 45% | 2.9 / 56% / 89% / 50% |
| -1.0 | 2.3 / 36% / 94% / 76% | 3.3 / 65% / 98% / 49% | 2.6 / 51% / 90% / 64% | 2.5 / 50% / 57% / 51% | 2.3 / 47% / — / 54% | 2.6 / 48% / 87% / 58% |
| 0.0 | 2.2 / 35% / 93% / 79% | 2.9 / 54% / 96% / 63% | 2.4 / 46% / 88% / 69% | 2.3 / 45% / 54% / 57% | 2.0 / 41% / — / 59% | 2.3 / 43% / 85% / 65% |
| 1.0 | 2.1 / 34% / 91% / 80% | 2.6 / 47% / 92% / 69% | 2.2 / 42% / 87% / 72% | 2.1 / 43% / 50% / 59% | 2.0 / 40% / — / 60% | 2.2 / 41% / 82% / 67% |
| 2.0 | 2.1 / 33% / 88% / 81% | 2.3 / 43% / 92% / 76% | 2.2 / 40% / 87% / 73% | 2.0 / 40% / 48% / 61% | 2.0 / 40% / — / 60% | 2.1 / 40% / 81% / 69% |

## Best reranker score in the pool (sufficiency signal)

| Dataset | group | n | p10 | median | p90 |
| --- | --- | --- | --- | --- | --- |
| 2wiki | answerable, gold in pool | 45 | 2.53 | 7.12 | 9.07 |
| 2wiki | unanswerable | 5 | -3.52 | -2.40 | 3.51 |
| hotpotqa | answerable, gold in pool | 45 | 2.82 | 6.60 | 9.91 |
| hotpotqa | unanswerable | 5 | -3.90 | -1.40 | 0.38 |
| musique | answerable, gold in pool | 44 | -0.52 | 3.52 | 6.74 |
| musique | answerable, no gold in pool | 1 | -3.23 | -3.23 | -3.23 |
| musique | unanswerable | 5 | -5.04 | -1.07 | -0.50 |
| qasper | answerable, gold in pool | 40 | -3.22 | 0.19 | 2.70 |
| qasper | answerable, no gold in pool | 6 | -5.93 | -3.38 | -0.79 |
| qasper | unanswerable | 2 | -5.31 | -3.53 | -1.76 |
| all | answerable, gold in pool | 174 | -0.61 | 4.44 | 8.97 |
| all | answerable, no gold in pool | 7 | -5.93 | -3.23 | -1.31 |
| all | unanswerable | 17 | -3.90 | -1.76 | 0.38 |

## Share sent to *insufficient* by `insufficient_below`

| Dataset | group | < -8.0 | < -6.0 | < -5.0 | < -4.0 | < -3.0 | < -2.0 | < -1.0 | < 0.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wiki | answerable, gold in pool | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| 2wiki | unanswerable | 0% | 0% | 0% | 0% | 20% | 60% | 60% | 60% |
| hotpotqa | answerable, gold in pool | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 2% |
| hotpotqa | unanswerable | 0% | 0% | 0% | 0% | 40% | 40% | 60% | 60% |
| musique | answerable, gold in pool | 0% | 0% | 0% | 0% | 0% | 0% | 2% | 11% |
| musique | answerable, no gold in pool | 0% | 0% | 0% | 0% | 100% | 100% | 100% | 100% |
| musique | unanswerable | 0% | 0% | 20% | 20% | 40% | 40% | 60% | 100% |
| qasper | answerable, gold in pool | 0% | 0% | 0% | 8% | 15% | 20% | 30% | 48% |
| qasper | answerable, no gold in pool | 0% | 17% | 33% | 33% | 67% | 67% | 83% | 100% |
| qasper | unanswerable | 0% | 0% | 50% | 50% | 50% | 50% | 100% | 100% |
| all | answerable, gold in pool | 0% | 0% | 0% | 2% | 3% | 5% | 7% | 14% |
| all | answerable, no gold in pool | 0% | 14% | 29% | 29% | 71% | 71% | 86% | 100% |
| all | unanswerable | 0% | 0% | 12% | 12% | 35% | 47% | 65% | 76% |
