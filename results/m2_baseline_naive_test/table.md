# Run `m2_baseline_naive_test`

split **test** · git `bdce5cdb04a3498f4b53a1656696f3fdd08b70f6-dirty` · config `1e3044992dd5` · generator `qwen2.5:7b` · Ollama 0.40.16 (vulkan)

Scores are % with 95% bootstrap CIs. Answer metrics cover all questions, with an abstention scored as the answer "Unanswerable". Retrieval metrics cover answerable questions with gold evidence.

## Answers

EM and F1 use the `short_answer` field; *F1 long* scores the full cited answer (as in M1).

| Dataset | n | EM | F1 | Acc (gold in answer) | F1 answerable | F1 long | MC acc | Generation failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qasper | 250 | 30.0 (24.4–36.4) | 47.5 (42.4–52.7) | 37.6 (32.0–44.0) | 49.3 (44.4–54.9) | 27.5 (24.3–30.8) | — | 0 |
| musique | 250 | 16.0 (11.6–20.4) | 19.7 (15.0–24.4) | 21.6 (16.4–26.8) | 12.9 (9.0–17.2) | 12.5 (9.4–15.8) | — | 0 |
| quality | 250 | 47.6 (41.6–54.0) | 51.7 (45.4–57.6) | 66.8 (61.2–72.8) | 51.7 (45.4–57.6) | 57.5 (52.8–62.7) | 67.2 (61.6–73.6) | 0 |
| hotpotqa | 250 | 33.2 (27.2–38.8) | 39.4 (33.5–44.6) | 42.4 (35.6–48.4) | 37.1 (31.1–42.8) | 15.8 (12.5–18.8) | — | 0 |
| 2wiki | 250 | 14.0 (10.0–18.4) | 15.4 (11.1–19.8) | 21.2 (16.0–26.4) | 9.1 (5.4–12.6) | 11.0 (7.6–14.4) | — | 0 |
| custom | 32 | 15.6 (3.1–28.1) | 36.3 (25.9–46.6) | 15.6 (3.1–28.1) | 24.5 (19.5–29.5) | 45.7 (34.7–56.0) | — | 0 |

## Retrieval

Recall/MRR/nDCG: top-20 dense retrieval against gold evidence passages. *Answer in context*: a short gold answer occurs in the evidence actually packed into the prompt (deterministic context recall).

| Dataset | n | Recall@5 | Recall@10 | Recall@20 | MRR | nDCG@10 | Answer in context (n) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qasper | 234 | 66.6 (60.8–72.0) | 88.5 (84.1–92.3) | 98.2 (96.5–99.6) | 52.6 (48.0–57.5) | 56.3 (52.0–60.6) | 43.3 (33.7–52.9) (104) |
| musique | 225 | 21.4 (18.0–25.0) | 26.7 (23.1–30.1) | 35.4 (31.3–39.4) | 34.7 (29.7–39.9) | 23.2 (19.9–26.7) | 30.3 (24.4–37.1) (221) |
| hotpotqa | 225 | 47.8 (43.1–52.0) | 56.7 (51.8–61.1) | 67.8 (63.6–72.2) | 54.1 (49.0–59.4) | 45.7 (41.6–49.5) | 62.2 (55.7–69.2) (201) |
| 2wiki | 225 | 12.7 (9.7–16.1) | 16.9 (13.6–20.7) | 22.8 (18.9–26.9) | 18.1 (13.8–22.8) | 13.2 (10.3–16.3) | 17.1 (11.9–22.8) (193) |
| custom | 27 | 13.0 (1.9–25.9) | 21.6 (9.3–34.6) | 21.6 (9.3–34.6) | 10.8 (3.4–19.7) | 12.6 (4.8–21.8) | 0.0 (0.0–0.0) (2) |

## Abstention and cost

| Dataset | Unanswerable | Abstain precision | Abstain recall | False-abstain rate | LLM calls/q | Tokens/q | Latency p50 | Latency p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qasper | 13 | 15% | 15% | 5% | 1.00 | 1833 | 20.6 s | 28.8 s |
| musique | 25 | 15% | 80% | 52% | 1.00 | 1318 | 12.7 s | 20.4 s |
| quality | 0 | — | — | 0% | 1.00 | 2578 | 22.7 s | 29.4 s |
| hotpotqa | 25 | 15% | 60% | 37% | 1.00 | 1346 | 12.0 s | 16.5 s |
| 2wiki | 25 | 10% | 72% | 71% | 1.00 | 1598 | 12.6 s | 20.7 s |
| custom | 5 | 71% | 100% | 7% | 1.00 | 2081 | — s | — s |

## F1 by hop count

- **musique:** 2-hop 27.3 (n=139), 3-hop 12.5 (n=65), 4-hop 6.7 (n=46)
- **custom:** 1-hop 45.0 (n=21), 2-hop 19.0 (n=9), 3-hop 22.6 (n=2)
