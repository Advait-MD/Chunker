# Retrieval-only comparison (dev split)

Answerable questions with gold evidence. % with 95% bootstrap CIs; latency is the retriever alone (query embedding cached), including the reranker when enabled.

| Dataset | Variant | Retriever | n | Recall@5 | Recall@10 | Recall@20 | MRR | nDCG@10 | p50 ms | p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hotpotqa | row2 | `bm25+dense_ivf|weighted` | 6 | 83.3 (66.7–100.0) | 91.7 (75.0–100.0) | 91.7 (75.0–100.0) | 88.9 (66.7–100.0) | 81.4 (66.4–94.6) | 7.9 | 9.0 |
| hotpotqa | row4 | `bm25+dense_ivf|weighted|rerank` | 6 | 91.7 (75.0–100.0) | 91.7 (75.0–100.0) | 91.7 (75.0–100.0) | 88.9 (66.7–100.0) | 84.4 (61.2–98.0) | 17686.8 | 27773.1 |
