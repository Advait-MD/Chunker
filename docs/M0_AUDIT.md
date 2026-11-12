# M0 — Audit of existing APEX code against the Agent Build Spec

Date: 2026-11-12 · Branch: `advait-feature-1` @ `d0c254f`
Scope: `modules/`, `main.py`, `face.py`, `app.py`, `eval/`, local environment.
Rule followed (spec §0, rule 1): nothing has been rewritten yet. This file lists the gaps and the decisions needed.

## 1. Environment (measured)

| Item | Found | Spec assumption | Impact |
| --- | --- | --- | --- |
| GPU | Intel Arc integrated (~4 GB shared), no CUDA | 7–14B instruct model in VRAM | A 7B model on CPU/iGPU runs at roughly 2–6 tok/s, so the SINGLE_HOP p95 < 10 s target is at risk |
| RAM | 15.4 GB | — | 7–8B Q4 fits; 14B does not fit comfortably |
| Ollama models | `gemma3:1b`, `llama3.2:1b`, `nomic-embed-text` | 7–14B generator, a separate judge | 1B generator; the earlier audit found QuALITY at chance with gemma3:1b |
| Python | 3.10.11 | — | OK |
| chromadb | 1.5.5 | — | HNSW only |
| faiss | 1.13.2 installed | needed for IVF | OK |
| umap-learn | installed | needed for RAPTOR | OK |
| sentence-transformers | **broken import** (`cached_download` removed from huggingface_hub) | cross-encoder reranker, NLI | Needs a version pin or a direct `transformers` load |
| Tests | `eval/tests`: 11 passed | unit + integration per route | Only metrics/cache tests exist |

## 2. ChromaDB infrastructure (teammate's) — reuse plan

`modules/vector_store.py::ChromaStore` is solid and reusable: a persistent client, four collections (`document_chunks` 6198, `hierarchical_summaries` 72, `conversation_chunks` 155, `conversation_summaries` 0), deterministic ids, file-hash dedup, page and char-span metadata.

**Answer to open question 1 (IVF):** the existing setup does **not** implement IVF. All collections use Chroma HNSW (`document_chunks`: cosine, M=32, ef_search=200). There is no FAISS code outside `naive rag/` (which uses `IndexFlatIP`). The spec's proposed design applies: Chroma stays the source of truth, a FAISS `IndexIVFFlat` is built from Chroma embeddings keyed by node id, and HNSW becomes the ablation baseline.

Problems found in the store:

1. **Embedding model mismatch.** `ChromaStore` hard-wires `DefaultEmbeddingFunction()` (all-MiniLM-L6-v2 ONNX, 384-d), even though `config.ollama.embedding_model = nomic-embed-text` and the logs claim nomic. The spec says to embed through Ollama. Switching means re-embedding all 6198 chunks.
2. `conversation_chunks` uses `l2` space while `retrieval.py` converts every distance with `1 - dist` as if it were cosine.
3. Leaves and summaries live in separate collections with different schemas. The spec wants one node schema with `parent_ids[] / child_ids[] / build_hash`. Child links are partly in summary metadata; leaves have no `parent_ids`.

## 3. Component-by-component gaps

| Spec component | Existing code | Status | Gap |
| --- | --- | --- | --- |
| Single logged LLM wrapper (rule 2) | Raw `requests.post` in `face.py` (×2), `summarizer.py`, `eval/judge.py`, `run_fair_benchmark.py` | ❌ | No wrapper; no prompt hash, token or latency logging per call; no content-hash cache |
| Config in `config/*.yaml` (rule 4) | `modules/config.py` dataclasses + `.env`; `eval/configs/*.yaml` | ⚠️ | Many hard-coded values: temperatures 0.1/0.3/0.4 in `face.py:703-795`, summariser 0.3, GMM `max_iter`, thematic trigger word list, chitchat regexes |
| Seeds / temp 0 (rule 3) | GMM `random_state=42`; summariser temp **0.3**; generator 0.2 | ⚠️ | Summaries not at temp 0 |
| Ingestor | `document_loader.py`: PDF only (pymupdf), sentence-aware, 300 tok / 40 overlap | ⚠️ | No Markdown/HTML/TXT; defaults differ from 400/50 (config change) |
| RAPTOR tree | `clustering.py`: **PCA + StandardScaler** + diag GMM, BIC over k∈[2,8], soft threshold 0.20, plus a **silhouette gate** (not in spec); max 3 levels; source-aware per-doc clustering | ⚠️ | Spec: UMAP (global + local passes), soft threshold 0.1, recursive re-clustering of over-budget clusters, depth 4, summary cache by `sha256(child ids + text)`. The silhouette gate is APEX's own addition; the earlier fair benchmark showed no F1 effect |
| Collapsed-tree retrieval | `collapsed_tree_fusion` in `retrieval.py` | ✅ partial | Summaries are capped at 3 candidates and filtered by silhouette; no traversal-mode ablation |
| BM25 | `BM25Scorer` (rank_bm25) | ❌ | It only **re-scores the dense top-12**. It is not an independent retriever over all nodes, so it cannot recover exact-term misses (the whole point of failure #2) |
| IVF dense | — | ❌ | Not present (see §2) |
| RRF fusion | `reciprocal_rank_fusion` exists | ❌ | Imported in `face.py`, **never called**. Live ranking is a weighted linear blend 0.5 dense + 0.3 BM25 + 0.2 coverage |
| Cross-encoder rerank | — | ❌ | The "rerank" is the linear blend; no cross-encoder; sentence-transformers broken |
| Route-aware bias | +0.05 for summaries if a thematic keyword appears | ⚠️ | Keyword list hard-coded; no route enum |
| DSM router | `is_chitchat_query`, `is_common_chat_query`, `detect_document_intent` (regex) | ⚠️ | L0-like rules only. No L1 prototype vectors, no L2 classifier, no answer cache, no state machine / transition table, no routing logs, no rule miner |
| Evidence filter | Jaccard dedup (0.65), min relevance 0.25, token-budget packing (350 tok, 2 chunks) | ⚠️ | No sentence pruning, NLI contradiction, sufficiency check, rewrite-and-retry, or `EvidenceSet` contract. "Abstain" happens only implicitly when scores are low (which reroutes to chitchat) |
| Generator | `face.py` prompt with numbered sources | ⚠️ | Free text, not `{answer, citations[], confidence, abstained}` JSON |
| Citation verifier | — | ❌ | None |
| Multi-hop controller | — | ❌ | None |
| Memory | Conversation turns indexed in Chroma + clustering of chats | ⚠️ | Closer to recall memory than MemGPT; the spec puts this in v2. A prior fix was needed because episodic memory poisoned doc answers (`659e53e`) |
| Trace | Console prints | ❌ | No trace id, states, per-stage latency or LLM call counts |
| API | `app.py` FastAPI UI on :7860; `face.py` CLI | ⚠️ | No `/query` JSON endpoint returning answer + citations + trace id |
| Eval harness | `eval/harness.py`, `metrics.py` (F1, ROUGE-L, bootstrap CI, paired t / Wilcoxon), `judge.py`, `run_fair_benchmark.py` | ⚠️ good base | No Recall@k / MRR / nDCG, citation P/R, abstention P/R, or cost columns; no `make eval`; gold set is 32 items; no HotpotQA/2Wiki/MuSiQue; `run_ablation_matrix.py` gate/BIC/soft rows don't rebuild the index (no-ops); judge = generator family risk |
| Repo layout | Flat `modules/` | ⚠️ | Spec proposes an `apex/` package; migration is a decision, not a requirement |

## 4. Known integrity issues to carry forward (from the 2026-10/11 audits)

- Paper Table 1 "APEX" numbers come from `scratch/run_raptor_paper_3way_benchmark.py`, not the APEX pipeline, with n=4–6 questions and unequal token budgets.
- Budget-matched fair benchmark: dense top-k beat full APEX on gold F1 (0.290 vs 0.234, p=0.16). Mixing summaries into context hurt (p≈0.03).
- Systems-table indexing/RAM numbers are hard-coded.

Under the spec's honesty clause, M2 baseline numbers must replace these before any paper claims.

## 5. Owner decisions (2026-11-12)

| Question | Decision |
| --- | --- |
| IVF implementation | Not in the existing infra → add FAISS `IndexIVFFlat` keyed by Chroma node ids; HNSW is the ablation baseline |
| Generator | `qwen2.5:7b`; `llama3.2:1b` for the L2 router; judge from a different family (gemma3) |
| Embeddings | Switch to Ollama `nomic-embed-text`; build a fresh DB; leave `hierarchical_chroma_db/` untouched |
| Memory | v2 (not in v1) |
| Publication target | Main conference → full benchmark suite (QASPER, QuALITY, NarrativeQA, HotpotQA, 2Wiki, MuSiQue), larger n |
| Custom eval corpus | The 4 repo PDFs; extend `eval/dataset/gold.jsonl` from 32 to 100–200 items |
| Still open | Demo latency budget (Q6). A 7B model on this iGPU may miss SINGLE_HOP p95 < 10 s; this will be measured in M1 and reported |

## 6. Proposed path

- **M1** — Add `llm/ollama_client.py` (logged wrapper + content-hash cache) and `trace/tracer.py`. Route every existing call through them. Build the naive baseline (flat chunks, dense top-k, JSON-cited answer) on top of the existing `ChromaStore`. No rewrite of working modules.
- **M2** — Extend the existing `eval/` (keep `metrics.py`, `harness.py`), add retrieval/citation/abstention/cost metrics, a `Makefile`, plus a QASPER slice and a HotpotQA-distractor slice. Produce baseline numbers.
- **M3+** — Independent BM25 over all nodes + FAISS IVF + RRF + cross-encoder, then the remaining milestones in order.
