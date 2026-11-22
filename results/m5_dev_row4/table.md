# Run `m5_dev_row4`

split **dev** · git `4039b8c641e515dec0f20d445057143fa168b424-dirty` · config `b202f19d7845` · generator `qwen2.5:7b` · Ollama 0.40.16 (vulkan)

Scores are % with 95% bootstrap CIs. Answer metrics cover all questions, with an abstention scored as the answer "Unanswerable". Retrieval metrics cover answerable questions with gold evidence.

## Answers

EM and F1 use the `short_answer` field; *F1 long* scores the full cited answer (as in M1).

| Dataset | n | EM | F1 | Acc (gold in answer) | F1 answerable | F1 long | MC acc | Generation failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wiki | 50 | 38.0 (24.0–52.0) | 39.5 (26.3–53.5) | 52.0 (38.0–66.0) | 35.0 (22.3–48.6) | 17.6 (10.3–26.1) | — | 0 |
| hotpotqa | 50 | 54.0 (40.0–68.0) | 67.1 (54.5–78.8) | 76.0 (64.0–88.0) | 70.1 (57.4–81.3) | 24.8 (17.6–32.9) | — | 0 |
| musique | 50 | 20.0 (10.0–32.0) | 28.1 (17.0–40.1) | 26.0 (14.0–38.0) | 26.8 (15.6–39.0) | 12.4 (6.8–19.2) | — | 0 |
| quality | 50 | 52.0 (38.0–66.0) | 56.4 (43.1–69.8) | 70.0 (58.0–82.0) | 56.4 (43.1–69.8) | 51.8 (40.1–63.2) | 72.0 (58.0–84.0) | 0 |
| qasper | 50 | 24.0 (12.0–36.0) | 51.6 (41.1–61.8) | 36.0 (22.0–50.0) | 51.6 (40.5–62.2) | 34.2 (26.2–42.4) | — | 0 |

## Retrieval

Recall/MRR/nDCG: the top 20 from the run's retriever, scored against gold evidence passages. *Answer in context*: a short gold answer occurs in the evidence actually packed into the prompt (deterministic context recall).

| Dataset | n | Recall@5 | Recall@10 | Recall@20 | MRR | nDCG@10 | Answer in context (n) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2wiki | 45 | 68.3 (61.1–75.6) | 70.0 (62.8–77.2) | 71.1 (63.9–78.3) | 100.0 (100.0–100.0) | 75.7 (70.8–80.7) | 69.2 (53.8–82.1) (39) |
| hotpotqa | 45 | 93.3 (87.8–97.8) | 97.8 (94.4–100.0) | 97.8 (94.4–100.0) | 95.4 (89.4–100.0) | 92.9 (87.9–96.8) | 93.0 (83.7–100.0) (43) |
| musique | 45 | 60.9 (53.9–68.1) | 64.8 (57.0–71.9) | 65.9 (58.5–73.0) | 92.6 (84.8–98.5) | 67.9 (61.4–73.8) | 47.7 (34.1–61.4) (44) |
| qasper | 46 | 80.3 (69.0–89.7) | 95.7 (90.2–100.0) | 98.9 (96.7–100.0) | 53.8 (43.4–63.2) | 60.6 (53.1–67.9) | 58.8 (35.3–82.4) (17) |

## Evidence and citations (M5, NLI)

*Context precision*: share of the passages given to the generator that cover a gold evidence passage. NLI columns (answered questions only, local DeBERTa-v3 NLI): *faithfulness* = share of answer sentences entailed by some evidence passage; *citation precision* = share of cited sentences entailed by their cited passages; *citation recall* = share of sentences with a valid citation. *Gen. input tokens*: the answer call's prompt.

| Dataset | Context precision | NLI n | Faithfulness | Citation precision | Citation recall | Gen. input tokens/q |
| --- | --- | --- | --- | --- | --- | --- |
| 2wiki | 36.0 (32.0–40.4) | 30 | 65.6 (50.6–80.0) | 66.1 (50.0–82.1) | 77.8 (66.7–88.3) | 1484 |
| hotpotqa | 37.3 (35.1–39.1) | 47 | 58.5 (44.7–72.3) | 59.3 (45.3–73.3) | 86.2 (76.6–93.6) | 1145 |
| musique | 30.7 (26.7–34.2) | 36 | 66.7 (50.0–80.6) | 66.7 (50.0–83.3) | 81.9 (68.1–94.4) | 1193 |
| quality | — | 50 | 74.0 (62.0–85.0) | 64.7 (50.0–82.4) | 60.8 (47.7–74.0) | 2514 |
| qasper | 22.2 (18.7–26.1) | 48 | 91.7 (83.3–97.9) | 91.4 (80.0–100.0) | 62.5 (49.3–75.3) | 1963 |

## Abstention and cost

| Dataset | Unanswerable | Abstain precision | Abstain recall | False-abstain rate | LLM calls/q | Tokens/q | Latency p50 | Latency p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wiki | 5 | 20% | 80% | 36% | 1.00 | 1561 | 45.0 s | 65.3 s |
| hotpotqa | 5 | 67% | 40% | 2% | 1.00 | 1219 | 19.0 s | 26.0 s |
| musique | 5 | 14% | 40% | 27% | 1.00 | 1262 | 12.9 s | 16.7 s |
| quality | 0 | — | — | 0% | 1.00 | 2588 | 23.5 s | 30.6 s |
| qasper | 2 | 50% | 50% | 2% | 1.00 | 2054 | 21.0 s | 31.0 s |

## F1 by hop count

- **musique:** 2-hop 35.9 (n=33), 3-hop 10.5 (n=13), 4-hop 21.4 (n=4)
