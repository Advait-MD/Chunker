# M1 — Naive baseline end-to-end

Branch: `m1-naive-baseline`. Done-when (spec): *flat chunks, dense retrieval and Ollama generation answer with citations.*

## What was built

New package `apex/`, alongside the existing code. Nothing in `modules/`, `face.py`, `main.py` or `app.py` was changed.

| File | Role |
| --- | --- |
| `configs/base.yaml` | All models, thresholds, paths and seeds (rule 4). Overlays deep-merge; `extends:` is supported; the config hash goes into every trace |
| `apex/llm/ollama_client.py` | **The single LLM wrapper** (rule 2). Generate + embed, content-hash SQLite cache, one JSONL record per call (`logs/llm_calls.jsonl`: prompt hash, model, tokens, latency, server time, cache hit, caller, pipeline state, trace id), seed + temperature passed on every call, loud errors and no silent fallbacks |
| `apex/trace/tracer.py` | Per-query trace: route, tier, states visited, per-stage latency, retrieved/filtered ids, every LLM call, token totals, git SHA (with `-dirty`), config hash → `logs/traces/<id>.json` |
| `apex/ingest/parsers.py`, `chunker.py` | PDF (reuses `modules.document_loader.read_pdf_pages`), Markdown by heading section, HTML minus boilerplate, TXT. Sentence-boundary chunks of ≤400 tokens with ~50 overlap that never cross a page (reuses `create_chunks_from_pages`) |
| `apex/index/embedder.py` | `nomic-embed-text` via the wrapper, with `search_document:` / `search_query:` prefixes and L2 normalisation |
| `apex/index/chroma_store.py` | `NodeStore`: one Chroma record per node in the spec §2.3 schema (`level, parent_ids, child_ids, doc_id, source_path, page, section, char_span, token_count, created_at, build_hash`), cosine HNSW, explicit embeddings |
| `apex/ingest/pipeline.py` | One-command ingest. `doc_id` = file SHA-256 and `build_hash` = chunker + embedder params; unchanged files are skipped, and a changed file replaces its old nodes |
| `apex/retrieve/dense.py` | Dense top-k over leaves (k=5) |
| `apex/generate/` | JSON-schema output `{answer, citations[], confidence, abstained}` at temp 0; budget packing assigns `[1..n]`; citation parser and checker (invalid ids, uncited sentences) |
| `apex/pipeline.py` | `NaivePipeline.answer()` → answer, resolved citations (source, page, char span, node id), trace id, totals |
| `apex/api/cli.py`, `server.py` | `python -m apex.api.cli ingest/query/stats`; FastAPI `POST /query`, `GET /health` |
| `tests/` | 30 unit and integration tests, offline via a fake Ollama transport (citation parser, config, wrapper cache/logging/trace binding, tracer, parsers, chunker, packing, end-to-end ingest→query→trace, cache reuse, changed-file replacement, abstention, invalid citations, FastAPI) |
| `scripts/smoke_naive.py` | Runs gold.jsonl through the baseline and records latency, citation and abstention behaviour |

## Deviations and [PROPOSED] choices to flag

1. **PDF parser is PyPDF2, not pymupdf.** I reused the teammate's working `read_pdf_pages` and its artefact cleaning (rule 1: don't rewrite working modules). Swapping in pymupdf is a one-function change if extraction quality turns out to matter.
2. **Token counts are approximate.** The existing `count_tokens` (word + punctuation regex) is used, not the generator's tokenizer. Budgets are therefore approximate; Ollama's real `prompt_eval_count` is logged for cost metrics.
3. **New `NodeStore` instead of extending `ChromaStore`.** `ChromaStore` hard-wires Chroma's MiniLM embedding function and a 4-collection layout. The new store keeps its conventions but takes embeddings from the logged wrapper (needed for rule 2 and the nomic decision). Its data lives in `apex_db/`; `hierarchical_chroma_db/` is untouched.
4. **List fields are stored as JSON strings** in Chroma metadata (`parent_ids`, `child_ids`, `char_span`) for portability.
5. **The naive baseline has no router.** Traces record `route=SINGLE_HOP`, `routing_tier=none`. FILTERING is a pass-through (budget packing only). Abstention comes only from the model's own `abstained` flag; the sufficiency check is M5.
6. **Retries:** transport errors (connection refused or timeout) are retried twice with the same request, then raise. Wrong-model or HTTP errors raise immediately.
7. **Name-keyed replacement:** a changed file's old nodes are found by file name, so two different files with the same name in different folders would collide. That's fine for the current corpus; revisit if corpora get nested.

8. **Embedding batch size.** Ollama 0.13.1 panics on nomic-embed-text when one input exceeds `num_batch` (default 512): "caching disabled but unable to fit entire input in a batch". Chunks capped at 400 *word* tokens can exceed 512 BPE tokens, especially code. `embedding.options: {num_ctx: 2048, num_batch: 2048}` fixes this. The first ingest crashed loudly on this, as intended, and resumed from the embedding cache.

## Hardware notes

- Ingest ran CPU-only ("offloaded 0/13 layers to GPU", 4 threads): embedding throughput was about 1.4 chunks/s.
- **Vulkan enabled 2026-11-12 at the owner's request:** `OLLAMA_VULKAN=1` is set as a persistent user environment variable and the Ollama app restarted. The server log shows `library=Vulkan … Intel(R) Arc(TM) Graphics`, 8.9 GiB shared, and 29/29 qwen2.5:7b layers plus the 8k KV cache on Vulkan0. Revert by clearing the variable and restarting Ollama; also set `ollama.backend: cpu`.

## Results

32 gold questions (16 single-hop, 11 multi-hop, 5 unanswerable). Each one runs as one dense top-5 retrieval plus one qwen2.5:7b generation at temperature 0, with ~1,870 prompt tokens on average. This is a behaviour check, not the M2 quality evaluation: no correctness metric is computed. Raw output is in `results/m1_smoke_vulkan/` (`per_query.jsonl`, `summary.json`); every trace is in `logs/traces/`.

### Latency: CPU vs Vulkan (Intel Arc iGPU)

| Backend | n | p50 | p95 | max | Prompt eval | Decode |
| --- | --- | --- | --- | --- | --- | --- |
| CPU, 4 threads, 0/29 layers on GPU | 18 (run stopped) | 206.8 s | 287.2 s | 317.7 s | ~9 tok/s | — |
| **Vulkan, 29/29 layers + KV on GPU** | 32 | **18.3 s** | **31.8 s** | 45.5 s | ~140 tok/s | ~8.7 tok/s |

Vulkan by question type:

| Type | p50 | p95 | Spec target |
| --- | --- | --- | --- |
| single-hop | 15.4 s | 20.5 s | **p95 < 10 s: missed** |
| multi-hop | 26.4 s | 39.6 s | — |
| unanswerable | 19.6 s | 21.4 s | — |

- Vulkan is about **11× faster** than CPU. `ollama_ps` in `summary.json` confirms that qwen2.5:7b and nomic-embed-text were fully resident in GPU memory. The query embedding takes ~0.2 s.
- **The single-hop target is still missed by about 2×.** Reading ~1,870 prompt tokens at ~140 tok/s takes ~13 s, about 80% of single-hop time. Decoding (~50–100 tokens) is most of the rest. The levers, in order of quality cost:
  - cut `generation.context_token_budget` (2500 → ~1000 saves ~6 s);
  - keep the Ollama KV/prompt cache warm for the fixed system prompt;
  - use a 3B generator;
  - relax the target to this hardware.
  
  These are owner decisions and none has been applied.
- The CPU row covers only the first 18 questions. That run was stopped when Vulkan was enabled, so its sample is smaller and not identical to the Vulkan one.

### Behaviour (Vulkan run)

| Type | n | Abstained | Answered with ≥1 citation | Answers with an uncited sentence | Invalid citation ids |
| --- | --- | --- | --- | --- | --- |
| single-hop | 16 | 4 | 12 | 3 | 0 |
| multi-hop | 11 | 1 | 10 | 9 | 0 |
| unanswerable | 5 | **5/5** | 0 | 0 | 0 |

- **No hallucinated citation ids** across 32 answers. Every `[n]` resolved to a packed evidence item.
- **Unanswerable: 5/5 correct abstentions.** All used the exact abstention string.
- **4 of 16 answerable single-hop questions abstained. All four are retrieval misses, not generation errors.**
  - sh-001 "Who wrote *The Road to React*?" retrieved five Steve Jobs chunks.
  - sh-006, sh-013 and sh-015 retrieved the right book but not the answering passage.
  
  Dense-only top-5 misses title, name and exact-term queries. This is the expected baseline weakness and the M3 hypothesis (BM25 + RRF + rerank).
- **One answer contradicts its own abstention flag.** sh-013 set `abstained=true` but wrote a speculative answer with a citation ("…it can be inferred…"). The M5 verifier must treat `abstained` and answer text that disagree as a failure.
- **Multi-hop answers usually leave some sentences uncited** (9 of 10 answered). The model cites once and then synthesises across several sentences. M5 needs to decide whether that counts as a citation failure or the checker should treat a trailing citation as covering the whole paragraph. Single-hop has the same pattern in 3 of 12 answers (e.g. sh-009 "The six-headed Scylla devoured six men." declares citations but puts none inline).
- Multi-hop single-shot answers were produced and cited, but their correctness is unmeasured until M2.

### Latency probe: 1000-token evidence budget

Overlay `configs/ablations/ctx_budget_1000.yaml` changes one setting, `generation.context_token_budget`, from 2500 to 1000. Retrieval is unchanged (top-5). On average 3.3 of the 5 chunks reach the prompt, against all 5 before. Run on Vulkan; output in `results/m1_smoke_vulkan_ctx1000/`. sh-010 and mh-004 packed exactly the same evidence as the 2500 run, so their prompts were identical and served from cache. They are excluded from the latency figures below; n is the number of fresh answers.

| Type | n | p50, 2500 → 1000 | p95, 2500 → 1000 | Prompt tokens | Abstained | Answers with an uncited sentence |
| --- | --- | --- | --- | --- | --- | --- |
| single-hop | 15 | 15.4 → **10.7 s** | 20.5 → **12.6 s** | 1884 → 1150 | 4 → 4 | 3 → 6 |
| multi-hop | 10 | 26.4 → 17.9 s | 39.6 → 28.1 s | 1777 → 1253 | 1 → 2 | 9 → 9 |
| unanswerable | 5 | 19.6 → 10.7 s | 21.4 → 12.1 s | 2006 → 1246 | 5 → 5 | 0 → 0 |

- **The single-hop target is still missed** (p95 12.6 s against < 10 s). Only 5 of 15 fresh single-hop answers came in under 10 s. Latency fell ~35–40%. Prompt tokens fell only 39% because the system prompt and formatting are a fixed overhead. Decoding at ~8.7 tok/s now accounts for a larger share of the time.
- **No invalid citation ids.** All 5 unanswerable questions still abstain.
- **Evidence loss shows up in three answers:**
  - sh-007 (Steve Jobs's birth and adoption) went from an answer to "the evidence does not specify".
  - mh-008 (Polyphemus → Poseidon's curse) went from an answer to abstaining. The 2500 answer relied on chunk [5], which no longer fits.
  - sh-013 (Macintosh launch year) went the other way: the speculative answer became a correct "1984". With less distracting context, the model stopped hedging.
- **More uncited sentences on single-hop answers** (3 → 6), even though every answer still carries at least one citation.
- **Conclusion:** a smaller budget alone does not reach the target, and it costs some evidence coverage, which cannot be quantified until M2. `base.yaml` therefore stays at 2500. The overlay is kept as a recorded ablation.

### Gold-set issues found (for M2)

- `gold_chunk_ids` use the old `modules/` chunker's id format (`<file>_chunk_<offset>_<hash>`). Retrieval recall@k against the new `apex_db` nodes needs either a mapping by text span or re-labelled gold.
- sh-002 (publication date of *The Road to React*) has a gold chunk inside `Steve Jobs (1).pdf`. This looks like a labelling error and should be checked before it is used as ground truth.

## Deviation 9: backend is part of the generate cache key

`ollama.backend` (`vulkan` | `cpu`) is a config value. It is included in the generate cache key and in every call-log record. CPU and Vulkan numerics differ, so a cached CPU answer must never be served, or reported as a latency, under Vulkan. Embedding cache keys deliberately exclude it, so the 2,646 document embeddings computed on CPU are reused rather than recomputed. Query embeddings now run on Vulkan. Any cosine drift from mixing backends is expected to be negligible but has not been measured. The label is not auto-detected. `scripts/smoke_naive.py` records `/api/ps` `size_vram` as evidence of where the models actually ran.
