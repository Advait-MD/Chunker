# Run `m4_seed_dev_s42`

split **dev** · git `bd2178a3f40b23a4214eda1f6ca8de604411a1f2-dirty` · config `be77b0066334` · generator `qwen2.5:7b` · Ollama 0.40.16 (vulkan)

Scores are % with 95% bootstrap CIs. Answer metrics cover all questions, with an abstention scored as the answer "Unanswerable". Retrieval metrics cover answerable questions with gold evidence.

## Answers

EM and F1 use the `short_answer` field; *F1 long* scores the full cited answer (as in M1).

| Dataset | n | EM | F1 | Acc (gold in answer) | F1 answerable | F1 long | MC acc | Generation failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qasper | 50 | 22.0 (12.0–34.0) | 44.8 (34.0–54.9) | 26.0 (14.0–38.0) | 44.5 (33.7–55.4) | 31.9 (24.7–39.1) | — | 0 |
| quality | 50 | 60.0 (48.0–72.0) | 64.9 (53.0–76.4) | 76.0 (64.0–86.0) | 64.9 (53.0–76.4) | 50.1 (38.4–61.5) | 74.0 (62.0–86.0) | 0 |

## Retrieval

Recall/MRR/nDCG: the top 20 from the run's retriever, scored against gold evidence passages. *Answer in context*: a short gold answer occurs in the evidence actually packed into the prompt (deterministic context recall).

| Dataset | n | Recall@5 | Recall@10 | Recall@20 | MRR | nDCG@10 | Answer in context (n) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qasper | 46 | 54.0 (40.2–67.8) | 85.9 (76.1–94.6) | 97.8 (94.6–100.0) | 41.0 (30.7–50.8) | 45.1 (38.2–52.4) | 41.2 (17.6–64.7) (17) |

## Abstention and cost

| Dataset | Unanswerable | Abstain precision | Abstain recall | False-abstain rate | LLM calls/q | Tokens/q | Latency p50 | Latency p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qasper | 2 | 25% | 50% | 6% | 1.00 | 1743 | 17.9 s | 23.2 s |
| quality | 0 | — | — | 0% | 1.00 | 2290 | 20.8 s | 25.4 s |
