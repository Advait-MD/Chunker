# Run `m3_row4_rerank_test100`

split **test** · git `387e1de2bff7fe25bd117218cec729c7d32e4d36-dirty` · config `004599b3b760` · generator `qwen2.5:7b` · Ollama 0.40.16 (vulkan)

Scores are % with 95% bootstrap CIs. Answer metrics cover all questions, with an abstention scored as the answer "Unanswerable". Retrieval metrics cover answerable questions with gold evidence.

## Answers

EM and F1 use the `short_answer` field; *F1 long* scores the full cited answer (as in M1).

| Dataset | n | EM | F1 | Acc (gold in answer) | F1 answerable | F1 long | MC acc | Generation failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wiki | 100 | 31.0 (22.0–40.0) | 32.2 (23.1–41.6) | 39.0 (30.0–49.0) | 27.2 (19.0–36.8) | 15.2 (10.1–20.9) | — | 0 |
| custom | 32 | 15.6 (3.1–28.1) | 39.4 (29.8–49.6) | 18.8 (6.2–31.2) | 28.2 (23.2–34.1) | 49.8 (39.0–60.1) | — | 0 |
| hotpotqa | 100 | 50.0 (40.0–61.0) | 65.9 (57.8–73.9) | 66.0 (56.0–75.0) | 67.0 (58.4–74.8) | 22.3 (18.0–27.1) | — | 0 |
| musique | 100 | 22.0 (14.0–30.0) | 27.4 (19.0–35.7) | 30.0 (21.0–39.0) | 24.9 (17.2–33.4) | 12.3 (7.9–17.3) | — | 0 |
| qasper | 100 | 28.0 (19.0–37.0) | 50.0 (42.3–57.7) | 40.0 (30.0–50.0) | 52.1 (44.7–59.9) | 29.6 (25.1–34.3) | — | 0 |
| quality | 100 | 42.0 (33.0–51.0) | 46.2 (37.0–55.4) | 67.0 (58.0–76.0) | 46.2 (37.0–55.4) | 59.2 (51.0–67.0) | 67.0 (57.0–76.0) | 0 |

## Retrieval

Recall/MRR/nDCG: the top 20 from the run's retriever, scored against gold evidence passages. *Answer in context*: a short gold answer occurs in the evidence actually packed into the prompt (deterministic context recall).

| Dataset | n | Recall@5 | Recall@10 | Recall@20 | MRR | nDCG@10 | Answer in context (n) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2wiki | 89 | 70.5 (65.2–76.1) | 72.8 (67.4–77.8) | 74.2 (69.1–79.5) | 98.9 (95.5–100.0) | 77.2 (73.3–81.1) | 62.5 (50.0–73.6) (72) |
| custom | 27 | 40.7 (25.9–57.4) | 47.5 (32.1–64.2) | 56.2 (40.7–72.8) | 32.6 (20.2–46.3) | 33.4 (21.8–46.3) | 0.0 (0.0–0.0) (2) |
| hotpotqa | 91 | 95.1 (91.8–97.8) | 96.2 (93.4–98.4) | 96.7 (94.0–98.9) | 96.7 (94.0–98.9) | 93.3 (90.3–95.9) | 92.3 (85.9–97.4) (78) |
| musique | 90 | 57.1 (52.5–62.1) | 61.8 (57.0–67.2) | 64.1 (58.9–69.6) | 86.5 (80.3–92.2) | 62.5 (57.9–66.9) | 50.0 (38.6–60.2) (88) |
| qasper | 95 | 76.3 (67.9–84.7) | 87.8 (80.9–93.7) | 95.8 (91.8–99.1) | 55.1 (47.3–62.7) | 59.4 (52.0–66.4) | 52.3 (38.6–68.2) (44) |

## Abstention and cost

| Dataset | Unanswerable | Abstain precision | Abstain recall | False-abstain rate | LLM calls/q | Tokens/q | Latency p50 | Latency p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wiki | 11 | 20% | 73% | 37% | 1.00 | 1498 | 20.3 s | 33.1 s |
| custom | 5 | 100% | 100% | 0% | 1.00 | 2296 | 36.2 s | 53.3 s |
| hotpotqa | 9 | 56% | 56% | 4% | 1.00 | 1248 | 18.5 s | 26.7 s |
| musique | 10 | 12% | 50% | 42% | 1.00 | 1280 | 17.9 s | 29.5 s |
| qasper | 4 | 0% | 0% | 4% | 1.00 | 1981 | 30.9 s | 64.8 s |
| quality | 0 | — | — | 0% | 1.00 | 2573 | 38.2 s | 52.5 s |

## F1 by hop count

- **custom:** 1-hop 50.0 (n=21), 2-hop 19.1 (n=9), 3-hop 19.9 (n=2)
- **musique:** 2-hop 29.0 (n=56), 3-hop 35.7 (n=26), 4-hop 10.3 (n=18)
