# Run `m4_seed_dev_row2`

split **dev** · git `bd2178a3f40b23a4214eda1f6ca8de604411a1f2-dirty` · config `c2649765878c` · generator `qwen2.5:7b` · Ollama 0.40.16 (vulkan)

Scores are % with 95% bootstrap CIs. Answer metrics cover all questions, with an abstention scored as the answer "Unanswerable". Retrieval metrics cover answerable questions with gold evidence.

## Answers

EM and F1 use the `short_answer` field; *F1 long* scores the full cited answer (as in M1).

| Dataset | n | EM | F1 | Acc (gold in answer) | F1 answerable | F1 long | MC acc | Generation failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qasper | 50 | 24.0 (12.0–36.0) | 51.4 (41.7–61.0) | 32.0 (20.0–44.0) | 49.4 (39.1–59.9) | 37.8 (29.9–45.4) | — | 0 |
| quality | 50 | 60.0 (46.0–74.0) | 64.5 (52.0–76.5) | 74.0 (62.0–84.0) | 64.5 (52.0–76.5) | 52.5 (41.5–63.3) | 74.0 (60.0–86.0) | 0 |

## Retrieval

Recall/MRR/nDCG: the top 20 from the run's retriever, scored against gold evidence passages. *Answer in context*: a short gold answer occurs in the evidence actually packed into the prompt (deterministic context recall).

| Dataset | n | Recall@5 | Recall@10 | Recall@20 | MRR | nDCG@10 | Answer in context (n) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qasper | 46 | 66.7 (54.3–79.3) | 97.0 (93.5–100.0) | 98.9 (96.7–100.0) | 47.7 (37.6–57.2) | 53.9 (47.5–60.2) | 41.2 (17.6–64.7) (17) |

## Abstention and cost

| Dataset | Unanswerable | Abstain precision | Abstain recall | False-abstain rate | LLM calls/q | Tokens/q | Latency p50 | Latency p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qasper | 2 | 50% | 100% | 4% | 1.00 | 1907 | 18.1 s | 25.3 s |
| quality | 0 | — | — | 0% | 1.00 | 2604 | 21.0 s | 27.6 s |
