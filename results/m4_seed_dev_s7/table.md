# Run `m4_seed_dev_s7`

split **dev** · git `bd2178a3f40b23a4214eda1f6ca8de604411a1f2-dirty` · config `3bcfde0db7d6` · generator `qwen2.5:7b` · Ollama 0.40.16 (vulkan)

Scores are % with 95% bootstrap CIs. Answer metrics cover all questions, with an abstention scored as the answer "Unanswerable". Retrieval metrics cover answerable questions with gold evidence.

## Answers

EM and F1 use the `short_answer` field; *F1 long* scores the full cited answer (as in M1).

| Dataset | n | EM | F1 | Acc (gold in answer) | F1 answerable | F1 long | MC acc | Generation failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qasper | 50 | 24.0 (12.0–36.0) | 46.2 (35.1–56.7) | 30.0 (18.0–42.0) | 46.1 (35.1–57.2) | 34.2 (26.8–41.5) | — | 0 |
| quality | 50 | 54.0 (40.0–68.0) | 58.6 (45.0–72.2) | 64.0 (50.0–78.0) | 58.6 (45.0–72.2) | 45.6 (34.9–57.5) | 64.0 (50.0–78.0) | 0 |

## Retrieval

Recall/MRR/nDCG: the top 20 from the run's retriever, scored against gold evidence passages. *Answer in context*: a short gold answer occurs in the evidence actually packed into the prompt (deterministic context recall).

| Dataset | n | Recall@5 | Recall@10 | Recall@20 | MRR | nDCG@10 | Answer in context (n) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qasper | 46 | 50.4 (36.9–64.9) | 88.0 (79.3–95.7) | 97.8 (94.6–100.0) | 40.7 (30.7–50.4) | 45.3 (38.9–51.9) | 41.2 (17.6–64.7) (17) |

## Abstention and cost

| Dataset | Unanswerable | Abstain precision | Abstain recall | False-abstain rate | LLM calls/q | Tokens/q | Latency p50 | Latency p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qasper | 2 | 25% | 50% | 6% | 1.00 | 1756 | 17.1 s | 23.2 s |
| quality | 0 | — | — | 0% | 1.00 | 2315 | 17.9 s | 23.4 s |
