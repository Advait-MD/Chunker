# Run `m4_row4_raptor_rerank_test100`

split **test** · git `4d7bfa372b4baa0aa416414bd17738d1f068f743-dirty` · config `2aa72b3136e6` · generator `qwen2.5:7b` · Ollama 0.40.16 (vulkan)

Scores are % with 95% bootstrap CIs. Answer metrics cover all questions, with an abstention scored as the answer "Unanswerable". Retrieval metrics cover answerable questions with gold evidence.

## Answers

EM and F1 use the `short_answer` field; *F1 long* scores the full cited answer (as in M1).

| Dataset | n | EM | F1 | Acc (gold in answer) | F1 answerable | F1 long | MC acc | Generation failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qasper | 100 | 27.0 (18.0–36.0) | 48.6 (40.8–56.5) | 38.0 (28.0–48.0) | 50.6 (43.0–58.3) | 28.9 (24.2–33.6) | — | 0 |
| musique | 100 | 23.0 (15.0–32.0) | 27.1 (19.1–35.8) | 33.0 (24.0–43.0) | 25.7 (17.3–34.9) | 11.8 (7.8–16.8) | — | 0 |
| quality | 100 | 40.0 (31.0–50.0) | 46.7 (37.9–55.5) | 57.0 (48.0–67.0) | 46.7 (37.9–55.5) | 50.8 (42.0–58.0) | 57.0 (47.0–66.0) | 0 |
| hotpotqa | 100 | 50.0 (40.0–60.0) | 64.2 (56.1–72.9) | 67.0 (57.0–76.0) | 62.8 (53.6–71.5) | 25.4 (20.3–31.2) | — | 0 |
| 2wiki | 100 | 28.0 (19.0–37.0) | 28.9 (20.0–37.9) | 38.0 (29.0–48.0) | 23.5 (15.0–31.6) | 16.1 (10.5–21.9) | — | 0 |
| custom | 32 | 15.6 (3.1–28.1) | 41.5 (31.1–51.7) | 15.6 (3.1–28.1) | 30.7 (22.9–39.2) | 48.8 (39.0–58.6) | — | 0 |

## Retrieval

Recall/MRR/nDCG: the top 20 from the run's retriever, scored against gold evidence passages. *Answer in context*: a short gold answer occurs in the evidence actually packed into the prompt (deterministic context recall).

| Dataset | n | Recall@5 | Recall@10 | Recall@20 | MRR | nDCG@10 | Answer in context (n) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qasper | 95 | 64.9 (54.9–74.9) | 81.4 (72.9–88.4) | 92.7 (87.2–97.2) | 44.1 (36.5–51.4) | 49.5 (42.3–56.2) | 50.0 (36.4–65.9) (44) |
| musique | 90 | 50.1 (45.3–54.8) | 58.4 (53.6–63.6) | 60.4 (55.2–66.0) | 81.0 (73.9–87.6) | 57.0 (52.5–61.5) | 48.9 (38.6–59.1) (88) |
| hotpotqa | 91 | 89.6 (85.2–93.4) | 91.8 (87.9–95.1) | 92.9 (89.0–96.2) | 95.2 (91.9–98.2) | 88.5 (84.7–92.0) | 89.7 (82.1–96.2) (78) |
| 2wiki | 89 | 66.3 (61.0–71.3) | 71.3 (66.0–76.7) | 73.6 (68.3–78.9) | 97.5 (94.1–100.0) | 74.6 (70.7–78.6) | 59.7 (48.6–70.8) (72) |
| custom | 27 | 27.8 (13.0–44.4) | 35.2 (18.5–53.7) | 40.1 (22.2–58.0) | 20.1 (9.5–33.1) | 22.2 (11.6–35.1) | 0.0 (0.0–0.0) (2) |

## Abstention and cost

| Dataset | Unanswerable | Abstain precision | Abstain recall | False-abstain rate | LLM calls/q | Tokens/q | Latency p50 | Latency p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qasper | 4 | 0% | 0% | 4% | 1.00 | 1867 | 20.4 s | 28.5 s |
| musique | 10 | 11% | 40% | 36% | 1.00 | 1369 | 16.0 s | 23.2 s |
| quality | 0 | — | — | 0% | 1.00 | 2152 | 21.0 s | 27.9 s |
| hotpotqa | 9 | 70% | 78% | 3% | 1.00 | 1333 | 16.1 s | 21.2 s |
| 2wiki | 11 | 20% | 73% | 37% | 1.00 | 1460 | 18.0 s | 24.4 s |
| custom | 5 | 83% | 100% | 4% | 1.00 | 1779 | 22.2 s | 34.7 s |

## F1 by hop count

- **musique:** 2-hop 25.7 (n=56), 3-hop 38.1 (n=26), 4-hop 15.9 (n=18)
- **custom:** 1-hop 54.3 (n=21), 2-hop 21.0 (n=9), 3-hop 0.0 (n=2)
