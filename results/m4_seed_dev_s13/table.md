# Run `m4_seed_dev_s13`

split **dev** · git `bd2178a3f40b23a4214eda1f6ca8de604411a1f2-dirty` · config `a8e5d74416ce` · generator `qwen2.5:7b` · Ollama 0.40.16 (vulkan)

Scores are % with 95% bootstrap CIs. Answer metrics cover all questions, with an abstention scored as the answer "Unanswerable". Retrieval metrics cover answerable questions with gold evidence.

## Answers

EM and F1 use the `short_answer` field; *F1 long* scores the full cited answer (as in M1).

| Dataset | n | EM | F1 | Acc (gold in answer) | F1 answerable | F1 long | MC acc | Generation failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qasper | 50 | 22.0 (12.0–34.0) | 47.0 (36.4–56.6) | 26.0 (14.0–38.0) | 46.9 (36.4–57.3) | 31.6 (24.3–38.8) | — | 0 |
| quality | 50 | 56.0 (42.0–70.0) | 60.6 (48.5–73.2) | 66.0 (52.0–80.0) | 60.6 (48.5–73.2) | 47.1 (36.1–58.6) | 72.0 (58.0–84.0) | 0 |

## Retrieval

Recall/MRR/nDCG: the top 20 from the run's retriever, scored against gold evidence passages. *Answer in context*: a short gold answer occurs in the evidence actually packed into the prompt (deterministic context recall).

| Dataset | n | Recall@5 | Recall@10 | Recall@20 | MRR | nDCG@10 | Answer in context (n) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qasper | 46 | 52.9 (40.2–65.6) | 90.2 (81.5–96.7) | 97.8 (94.6–100.0) | 41.9 (31.6–51.8) | 46.6 (39.8–53.1) | 41.2 (17.6–64.7) (17) |

## Abstention and cost

| Dataset | Unanswerable | Abstain precision | Abstain recall | False-abstain rate | LLM calls/q | Tokens/q | Latency p50 | Latency p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qasper | 2 | 25% | 50% | 6% | 1.00 | 1789 | 14.7 s | 20.9 s |
| quality | 0 | — | — | 0% | 1.00 | 2327 | 16.3 s | 21.2 s |
