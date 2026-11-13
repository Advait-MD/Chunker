# Run `m3_row2_hybrid_test100`

split **test** · git `387e1de2bff7fe25bd117218cec729c7d32e4d36-dirty` · config `03c4d870ec11` · generator `qwen2.5:7b` · Ollama 0.40.16 (vulkan)

Scores are % with 95% bootstrap CIs. Answer metrics cover all questions, with an abstention scored as the answer "Unanswerable". Retrieval metrics cover answerable questions with gold evidence.

## Answers

EM and F1 use the `short_answer` field; *F1 long* scores the full cited answer (as in M1).

| Dataset | n | EM | F1 | Acc (gold in answer) | F1 answerable | F1 long | MC acc | Generation failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qasper | 100 | 22.0 (14.0–30.0) | 42.3 (34.7–50.4) | 33.0 (24.0–42.0) | 44.0 (36.9–52.1) | 23.1 (18.7–27.7) | — | 0 |
| musique | 100 | 23.0 (16.0–32.0) | 27.4 (19.7–36.4) | 28.0 (20.0–37.0) | 22.6 (15.4–31.1) | 15.0 (9.8–21.3) | — | 0 |
| quality | 100 | 42.0 (33.0–52.0) | 45.6 (36.5–54.7) | 68.0 (59.0–77.0) | 45.6 (36.5–54.7) | 58.3 (50.8–65.9) | 65.0 (56.0–74.0) | 0 |
| hotpotqa | 100 | 47.0 (37.0–57.0) | 58.8 (50.1–68.2) | 61.0 (51.0–71.0) | 58.0 (48.4–66.7) | 21.0 (16.1–26.4) | — | 0 |
| 2wiki | 100 | 31.0 (22.0–40.0) | 33.8 (24.8–43.3) | 38.0 (28.0–47.0) | 26.7 (18.2–35.5) | 16.2 (10.6–22.0) | — | 0 |
| custom | 32 | 15.6 (3.1–28.1) | 41.2 (31.5–50.8) | 21.9 (9.3–37.5) | 30.4 (24.7–36.7) | 48.1 (37.7–58.4) | — | 0 |

## Retrieval

Recall/MRR/nDCG: the top 20 from the run's retriever, scored against gold evidence passages. *Answer in context*: a short gold answer occurs in the evidence actually packed into the prompt (deterministic context recall).

| Dataset | n | Recall@5 | Recall@10 | Recall@20 | MRR | nDCG@10 | Answer in context (n) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qasper | 95 | 61.4 (51.3–71.1) | 87.5 (80.5–93.4) | 95.8 (91.8–99.1) | 43.7 (36.5–51.3) | 50.4 (43.9–56.7) | 43.2 (29.5–56.8) (44) |
| musique | 90 | 45.5 (39.6–51.5) | 55.1 (49.3–61.3) | 64.1 (58.9–69.6) | 70.0 (62.5–77.3) | 50.9 (45.4–56.4) | 38.6 (28.4–48.9) (88) |
| hotpotqa | 91 | 79.1 (72.5–84.6) | 90.7 (86.8–94.5) | 96.7 (94.0–98.9) | 89.6 (84.8–94.4) | 80.7 (76.9–84.6) | 78.2 (69.2–85.9) (78) |
| 2wiki | 89 | 66.9 (61.8–71.9) | 69.9 (64.6–75.6) | 74.2 (69.1–79.5) | 94.8 (90.8–98.3) | 71.7 (68.0–75.7) | 56.9 (44.4–68.1) (72) |
| custom | 27 | 29.6 (14.8–46.3) | 37.0 (20.4–55.6) | 56.2 (40.7–72.8) | 25.2 (13.4–38.7) | 25.1 (13.2–38.3) | 0.0 (0.0–0.0) (2) |

## Abstention and cost

| Dataset | Unanswerable | Abstain precision | Abstain recall | False-abstain rate | LLM calls/q | Tokens/q | Latency p50 | Latency p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qasper | 4 | 0% | 0% | 7% | 1.00 | 1885 | 19.0 s | 27.4 s |
| musique | 10 | 15% | 70% | 43% | 1.00 | 1212 | 11.6 s | 17.7 s |
| quality | 0 | 0% | — | 1% | 1.00 | 2580 | 22.3 s | 28.5 s |
| hotpotqa | 9 | 35% | 67% | 12% | 1.00 | 1207 | 12.6 s | 17.7 s |
| 2wiki | 11 | 20% | 91% | 45% | 1.00 | 1338 | 13.4 s | 21.2 s |
| custom | 5 | 100% | 80% | 0% | 1.00 | 2136 | 20.5 s | 30.1 s |

## F1 by hop count

- **musique:** 2-hop 37.7 (n=56), 3-hop 20.8 (n=26), 4-hop 4.8 (n=18)
- **custom:** 1-hop 53.2 (n=21), 2-hop 18.2 (n=9), 3-hop 19.3 (n=2)
