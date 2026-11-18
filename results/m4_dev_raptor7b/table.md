# Run `m4_dev_raptor7b`

split **dev** · git `db55cc94ada50cc303fe6db5a415526173c59e35-dirty` · config `5027bacd78a0` · generator `qwen2.5:7b` · Ollama 0.40.16 (vulkan)

Scores are % with 95% bootstrap CIs. Answer metrics cover all questions, with an abstention scored as the answer "Unanswerable". Retrieval metrics cover answerable questions with gold evidence.

## Answers

EM and F1 use the `short_answer` field; *F1 long* scores the full cited answer (as in M1).

| Dataset | n | EM | F1 | Acc (gold in answer) | F1 answerable | F1 long | MC acc | Generation failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qasper | 50 | 24.0 (14.0–36.0) | 50.9 (40.3–61.2) | 30.0 (18.0–42.0) | 50.9 (40.4–61.4) | 34.2 (27.4–41.1) | — | 0 |
| quality | 50 | 54.0 (40.0–68.0) | 58.4 (45.1–71.2) | 68.0 (54.0–80.0) | 58.4 (45.1–71.2) | 49.9 (38.5–61.1) | 72.0 (58.0–84.0) | 0 |

## Retrieval

Recall/MRR/nDCG: the top 20 from the run's retriever, scored against gold evidence passages. *Answer in context*: a short gold answer occurs in the evidence actually packed into the prompt (deterministic context recall).

| Dataset | n | Recall@5 | Recall@10 | Recall@20 | MRR | nDCG@10 | Answer in context (n) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qasper | 46 | 54.7 (40.9–68.5) | 88.0 (79.3–95.7) | 97.8 (94.6–100.0) | 38.2 (28.4–47.6) | 43.6 (37.7–49.9) | 41.2 (17.6–64.7) (17) |

## Abstention and cost

| Dataset | Unanswerable | Abstain precision | Abstain recall | False-abstain rate | LLM calls/q | Tokens/q | Latency p50 | Latency p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qasper | 2 | 50% | 50% | 2% | 1.00 | 1701 | 19.6 s | 29.5 s |
| quality | 0 | — | — | 0% | 1.00 | 2295 | 22.1 s | 29.1 s |
