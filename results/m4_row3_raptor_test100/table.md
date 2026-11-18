# Run `m4_row3_raptor_test100`

split **test** · git `4d7bfa372b4baa0aa416414bd17738d1f068f743-dirty` · config `be77b0066334` · generator `qwen2.5:7b` · Ollama 0.40.16 (vulkan)

Scores are % with 95% bootstrap CIs. Answer metrics cover all questions, with an abstention scored as the answer "Unanswerable". Retrieval metrics cover answerable questions with gold evidence.

## Answers

EM and F1 use the `short_answer` field; *F1 long* scores the full cited answer (as in M1).

| Dataset | n | EM | F1 | Acc (gold in answer) | F1 answerable | F1 long | MC acc | Generation failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wiki | 100 | 26.0 (18.0–35.0) | 28.3 (20.4–36.8) | 32.0 (23.0–41.0) | 26.2 (17.4–35.6) | 12.7 (8.4–17.3) | — | 0 |
| custom | 32 | 15.6 (3.1–28.1) | 36.1 (25.3–46.6) | 15.6 (3.1–28.1) | 24.2 (17.8–30.9) | 48.0 (37.5–57.9) | — | 0 |
| hotpotqa | 100 | 45.0 (35.0–54.0) | 56.4 (47.2–64.7) | 59.0 (50.0–68.0) | 55.4 (45.4–64.2) | 19.7 (15.2–25.0) | — | 0 |
| musique | 100 | 23.0 (15.0–31.0) | 29.8 (21.9–38.7) | 30.0 (21.0–40.0) | 24.2 (16.8–32.6) | 16.8 (11.0–23.1) | — | 0 |
| qasper | 100 | 24.0 (16.0–33.0) | 41.7 (33.7–49.8) | 31.0 (22.0–41.0) | 43.4 (35.8–51.2) | 23.2 (18.9–27.7) | — | 1 |
| quality | 100 | 40.0 (31.0–50.0) | 44.4 (35.6–54.3) | 60.0 (50.0–69.0) | 44.4 (35.6–54.3) | 55.2 (47.5–63.2) | 62.0 (52.0–72.0) | 0 |

## Retrieval

Recall/MRR/nDCG: the top 20 from the run's retriever, scored against gold evidence passages. *Answer in context*: a short gold answer occurs in the evidence actually packed into the prompt (deterministic context recall).

| Dataset | n | Recall@5 | Recall@10 | Recall@20 | MRR | nDCG@10 | Answer in context (n) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2wiki | 89 | 61.8 (56.5–67.4) | 68.5 (63.5–73.9) | 73.6 (68.3–78.9) | 76.9 (70.1–83.0) | 60.4 (56.4–64.5) | 52.8 (41.7–63.9) (72) |
| custom | 27 | 20.4 (7.4–35.2) | 29.6 (14.8–46.3) | 40.1 (22.2–58.0) | 17.6 (7.2–30.5) | 18.9 (7.8–32.2) | 0.0 (0.0–0.0) (2) |
| hotpotqa | 91 | 70.9 (64.8–76.4) | 82.4 (76.9–87.4) | 92.9 (89.0–96.2) | 74.3 (67.5–80.7) | 66.7 (62.0–71.0) | 75.6 (66.6–84.6) (78) |
| musique | 90 | 43.1 (37.5–49.3) | 51.8 (46.3–57.8) | 60.4 (55.2–66.0) | 64.3 (56.9–72.3) | 46.7 (41.5–52.3) | 42.0 (31.8–52.3) (88) |
| qasper | 95 | 47.6 (37.4–58.2) | 77.4 (68.2–85.8) | 92.7 (87.2–97.2) | 37.4 (29.7–44.6) | 43.1 (36.3–49.7) | 43.2 (29.5–56.8) (44) |

## Abstention and cost

| Dataset | Unanswerable | Abstain precision | Abstain recall | False-abstain rate | LLM calls/q | Tokens/q | Latency p50 | Latency p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wiki | 11 | 12% | 45% | 39% | 1.00 | 1346 | 14.1 s | 18.1 s |
| custom | 5 | 100% | 100% | 0% | 1.00 | 1783 | 16.8 s | 30.3 s |
| hotpotqa | 9 | 43% | 67% | 9% | 1.00 | 1305 | 13.0 s | 17.6 s |
| musique | 10 | 17% | 80% | 42% | 1.00 | 1293 | 14.1 s | 21.8 s |
| qasper | 4 | 0% | 0% | 6% | 1.00 | 1753 | 18.3 s | 23.8 s |
| quality | 0 | 0% | — | 1% | 1.00 | 2202 | 19.7 s | 32.6 s |

## F1 by hop count

- **custom:** 1-hop 44.6 (n=21), 2-hop 20.8 (n=9), 3-hop 15.1 (n=2)
- **musique:** 2-hop 37.2 (n=56), 3-hop 23.4 (n=26), 4-hop 15.9 (n=18)
