# Retrieval-only comparison (dev split)

Answerable questions with gold evidence. % with 95% bootstrap CIs; latency is the retriever alone (query embedding cached), including the reranker when enabled.

| Dataset | Variant | Retriever | n | Recall@5 | Recall@10 | Recall@20 | MRR | nDCG@10 | p50 ms | p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qasper | s42 | `bm25+dense_ivf|weighted` | 46 | 54.0 (40.2–67.8) | 85.9 (76.1–94.6) | 97.8 (94.6–100.0) | 41.0 (30.7–50.8) | 45.1 (38.2–52.4) | 6.6 | 12.5 |
| qasper | s7 | `bm25+dense_ivf|weighted` | 46 | 50.4 (36.9–64.9) | 88.0 (79.3–95.7) | 97.8 (94.6–100.0) | 40.7 (30.7–50.4) | 45.3 (38.9–51.9) | 2.3 | 4.1 |
| qasper | s13 | `bm25+dense_ivf|weighted` | 46 | 52.9 (40.2–65.6) | 90.2 (81.5–96.7) | 97.8 (94.6–100.0) | 41.9 (31.6–51.8) | 46.6 (39.8–53.1) | 2.1 | 4.2 |
