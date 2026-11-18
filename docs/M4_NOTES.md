# M4 — RAPTOR hierarchical summary tree

Spec §2.3 and ablation row 3, plus the cumulative row 4 (row 3 + rerank), the collapsed-vs-traversal comparison and the 3-seed check. Built 13–17 Nov 2026 on the laptop (Intel Arc iGPU, Ollama 0.40.16 on Vulkan).

**Headline: the tree does not help.** Adding RAPTOR to row 2 changes F1 by −5.4 to +2.4 points (none significant), and once the reranker is in it adds nothing (−3.3 to +2.1, none significant). QuALITY multiple-choice accuracy drops 10 points (p = 0.018 on the seed-42 tree), but the 3-seed check shows a seed-to-seed range just as large, so that drop is not established; no seed beats the no-tree row. The spec's hypothesis "RAPTOR helps thematic and long-document QA" fails on QASPER and QuALITY. Per the honesty clause the result is kept and reported; whether the component stays is the owner's call (see the end).

## What was built

- `apex/index/raptor.py`: the tree builder. Leaves are the ingested chunks; each parent is a summary of a cluster of its children.
  - Clustering as in the RAPTOR reference code: UMAP to 10 dimensions, then a GMM whose size is chosen by BIC (up to 50 components), soft assignment at probability 0.1, a global then a local stage, and recursive re-clustering of any cluster whose children exceed 3,500 tokens (forced id-ordered splits when UMAP or GMM cannot split further).
  - Levels repeat until a level yields fewer than 3 clusters or depth 4.
  - Summaries are temperature-0 calls through the single logged LLM wrapper, cached by content hash; a content-hashed manifest makes rebuilds idempotent.
  - The tree lives in its own Chroma collection, `<leaf collection>__raptor_s<seed>`, holding copies of the leaves plus the summaries, so seeds and parameters never mix.
  - `build(sources=...)` restricts a per-document build to a set of documents (used by the seed check).
- `apex/generate/prompts/raptor_summary.txt`: the summary prompt.
- Retrieval: collapsed-tree search (every level in one pool, searched by the same hybrid retriever, `levels: [0..4]`) is the default; tree traversal (`retrieval.tree_search: traversal`, top `traversal_k` = 5 nodes per layer by cosine, descend to children, rank all selected) is the ablation.
- `suite tree <datasets> [--seed N] [--only-split dev|test] [--force]` builds trees.
- Configs: `configs/ablations/row3_raptor.yaml`, `row4_raptor_rerank.yaml`, `tree_collapsed_dense.yaml`, `tree_traversal.yaml`, `row3_raptor_s7.yaml`, `row3_raptor_s13.yaml`.
- Tests: `tests/unit/test_raptor.py` (clustering, budgets, trimming, manifest reuse).
- Pins: umap-learn 0.5.5, pynndescent 0.6.0, numba 0.64.0.

## Deviations and [PROPOSED] choices to flag

- **Summarizer is qwen2.5:3b, not the 7B generator** (owner's fast-plan decision, 13 Nov). About 2× faster on the iGPU; the trees still took about 14 h. The generator for every row stays qwen2.5:7b. The 3b model makes relation errors (below); `docs/M4_RAPTOR_REPORT.md` §4 compares it with the 7B.
- **[PROPOSED] Scope.** One tree per document for QASPER, QuALITY and custom (their questions are about one document and retrieval is filtered to it); one pooled tree per corpus for MuSiQue, HotpotQA and 2Wiki (questions span documents).
- **[PROPOSED] Summary length cap.** `summary_max_tokens: 256` with the prompt asking for about 100–150 tokens. 872 of 3,668 summaries (24%) hit the cap; 833 of those were trimmed back to the last full sentence (the other 39 had no full sentence to cut to). The first custom build had kept the cut-off text; it was rebuilt with trimming (tree schema 2) before anything was evaluated.
- **[PROPOSED] Seed check on dev documents only.** Re-seeding rebuilds the trees, about 14 h per seed for all corpora. Seeds 7 and 13 were built only for the QASPER and QuALITY dev documents; the pooled multi-hop trees were not re-seeded.
- Weighted fusion (0.3 dense / 0.7 BM25) is carried over from M3, as flagged there.

## Trees (seed 42, `results/m4_trees`)

| Corpus | Scope | Leaves | Summaries (L1/L2/L3/L4) | Capped / trimmed | Build time |
| --- | --- | --- | --- | --- | --- |
| custom | per document | 2,646 | 546 (451/81/12/2) | 86 / 82 | 2.1 h (+0.3 h rebuild) |
| QuALITY | per document | 1,625 | 383 (312/71) | 28 / 25 | 1.5 h |
| QASPER | per document | 4,896 | 1,154 (944/210) | 218 / 207 | 4.3 h |
| MuSiQue | pooled | 4,654 | 708 (580/109/17/2) | 256 / 246 | 2.7 h |
| HotpotQA | pooled | 2,943 | 525 (437/76/11/1) | 159 / 154 | 1.9 h |
| 2Wiki | pooled | 2,066 | 352 (291/51/9/1) | 125 / 119 | 1.3 h |
| **Total** | | 18,830 | **3,668** | 872 / 833 | **about 14.2 h** |

About 12–15 s per summary (reading about 280 tok/s, writing about 23 tok/s). No thermal throttling: the time per call stayed within 5% of a morning fit on the 14th; the 15th ran about 10% slower, under irregular mains power.

## Collapsed vs traversal (dev, retrieval only, `results/m4_tree_search_dev`)

Recall@10 against gold passages, % with 95% bootstrap CIs. Both dense-only rows use the same embeddings, so they compare like for like (traversal ranks by cosine, as in the RAPTOR paper).

| Dataset | n | Row 2 (no tree) | Row 3 collapsed, hybrid | Collapsed, dense | Traversal, dense |
| --- | --- | --- | --- | --- | --- |
| QASPER | 46 | 97.0 | 85.9 | 85.5 | 77.9 |
| MuSiQue | 45 | 59.1 | 55.7 | 18.1 | 16.1 |
| HotpotQA | 45 | 95.6 | 95.6 | 53.3 | 41.1 |
| 2Wiki | 45 | 68.3 | 66.7 | 10.6 | 7.8 |

- **Collapsed beats traversal on all four**, matching the RAPTOR paper, so collapsed stays the default. Traversal also caps recall at whatever its fixed number of selected nodes reaches (Recall@10 = Recall@20 on three datasets).
- QuALITY has no gold passages and custom has no dev split, so neither has retrieval rows.
- Gold evidence is original passages, so a summary in the top k always counts as a miss. Retrieval scores therefore understate what the tree contributes; answer F1 is the fair comparison.

## Results: test (first 100 questions per dataset plus custom 32)

F1 in points; Δ with a paired 95% bootstrap CI; p is paired bootstrap / Wilcoxon. Full tables: `results/m4_row3_raptor_test100/compare_vs_*.md`, `results/m4_row4_raptor_rerank_test100/compare_vs_*.md`.

### Row 3: row 2 + RAPTOR tree (`results/m4_row3_raptor_test100`)

| Dataset | M2 | Row 2 | Row 3 | Δ vs row 2 | p |
| --- | --- | --- | --- | --- | --- |
| QASPER | 44.5 | 42.3 | 41.7 | −0.6 (−5.6, +4.5) | 0.81 / 0.86 |
| MuSiQue | 15.6 | 27.4 | 29.8 | +2.4 (−3.9, +9.1) | 0.46 / 0.45 |
| QuALITY | 44.4 | 45.6 | 44.4 | −1.2 (−8.9, +6.5) | 0.78 / 0.73 |
| HotpotQA | 41.5 | 58.8 | 56.4 | −2.4 (−8.3, +3.4) | 0.42 / 0.43 |
| 2Wiki | 24.1 | 33.8 | 28.3 | −5.4 (−13.5, +2.1) | 0.17 / 0.11 |
| custom | 36.3 | 41.2 | 36.1 | −5.2 (−10.8, +0.2) | 0.06 / 0.09 |

QuALITY multiple-choice accuracy: 65 (row 2) → 62 (row 3), p = 0.48.

Retrieval ranking falls significantly everywhere with gold passages (nDCG@10 −4.2 to −14.0 points, all p < 0.01; Recall@10 −1.4 to −10.0), partly by construction (summaries count as misses).

### Row 4: row 3 + cross-encoder rerank (`results/m4_row4_raptor_rerank_test100`)

| Dataset | M3 row 4 (no tree) | Row 4 | Δ vs M3 row 4 | p | Δ vs row 3 | p |
| --- | --- | --- | --- | --- | --- | --- |
| QASPER | 50.0 | 48.6 | −1.4 (−5.7, +2.5) | 0.51 / 0.73 | **+6.9** (+1.5, +12.5) | 0.011 / 0.010 |
| MuSiQue | 27.4 | 27.1 | −0.2 (−8.4, +8.0) | 0.95 / 0.97 | −2.6 | 0.55 / 0.58 |
| QuALITY | 46.2 | 46.7 | +0.5 (−8.8, +10.0) | 0.92 / 0.71 | +2.3 | 0.55 / 0.53 |
| HotpotQA | 65.9 | 64.2 | −1.8 (−7.9, +4.6) | 0.59 / 0.59 | +7.7 (−0.9, +16.4) | 0.08 / 0.09 |
| 2Wiki | 32.2 | 28.9 | −3.3 (−11.4, +4.7) | 0.41 / 0.34 | +0.5 | 0.89 / 0.81 |
| custom | 39.4 | 41.5 | +2.1 (−2.5, +7.1) | 0.39 / 0.53 | +5.5 (−0.5, +12.7) | 0.08 / 0.16 |

- **QuALITY multiple-choice accuracy falls from 67 to 57** against M3's row 4 (−10.0, CI −18 to −2, p = 0.018 / 0.018). QuALITY is the long-document set where RAPTOR is supposed to help most. The test covers the seed-42 tree only; on dev the seed alone moves this metric by up to 10 points (see the 3-seed check), so the drop's size is not established.
- Row 4 against the M2 naive baseline: MuSiQue +11.6 (p = 0.008), HotpotQA +22.6 (p < 0.001); the other four are not significant. These gains are the same as M3's row 4 without the tree.

### Why the tree doesn't help

Summaries take a third of the generator's five context slots:

| Dataset | Packed passages that are summaries, row 3 / row 4 | Questions with at least one summary packed, row 3 / row 4 |
| --- | --- | --- |
| QASPER | 30% / 33% | 87 / 89 of 100 |
| MuSiQue | 27% / 31% | 77 / 82 |
| QuALITY | 35% / 38% | 89 / 93 |
| HotpotQA | 35% / 37% | 90 / 94 |
| 2Wiki | 39% / 42% | 93 / 94 |
| custom | 50% / 61% | 30 / 29 of 32 |

The reranker keeps them too (slightly more often). Likely reasons the summaries don't pay their way:
- They push specific original passages out of a fixed 5-passage budget, while most benchmark questions here need one specific fact.
- The 3b summarizer gets relations wrong (for example "his wife, Lisa Brennan-Jobs", "his son, Eve"), so a summary can mislead. A 7B summarizer fixes that, but the tree still only ties the no-tree row (below).
- A summary carries no page or span of its own, so its citations are coarser.

### Cost and latency

One LLM call per question in every row. Tokens per question: row 3 changes the prompt by −356 (custom) to +95 (HotpotQA) against row 2. p50 latency, row 2 → row 3: 11.6–22.3 s → 13.0–19.7 s.

Latency caveats:
- The first 281 row 3 questions (QASPER, MuSiQue, most of QuALITY) ran on 15 Nov under irregular mains power; their timings are unreliable.
- Row 4's p50 (16.0–22.2 s) is lower than M3's row 4 (17.9–38.2 s) because the reranker's score cache already held most leaf pairs from M3. It is not a speed-up.

Tree build cost: about 14.2 h of summarization on this iGPU for 3,668 summaries, once.

## 3-seed check (dev, `results/m4_seed_dev_*`)

Seeds 42, 7 and 13 for the QASPER (46 papers) and QuALITY (3 stories) dev documents, row 3 config, against the same 100 dev questions with no tree (row 2, `results/m4_seed_dev_row2`). Seeds 7 and 13 rebuilt only those documents: QASPER 210 and 213 summaries (77 and 99 reused from the cache, where clusters came out the same), QuALITY 8 and 9.

| | No tree (row 2) | Seed 42 | Seed 7 | Seed 13 | Mean of seeds | Seed-averaged Δ vs no tree |
| --- | --- | --- | --- | --- | --- | --- |
| QASPER Recall@10 (46) | 97.0 | 85.9 | 88.0 | 90.2 | 88.0 | |
| QASPER nDCG@10 (46) | 53.9 | 45.1 | 45.3 | 46.6 | 45.7 | |
| QASPER F1 (50) | 51.4 | 44.8 | 46.2 | 47.0 | 46.0 ± 1.1 | −5.4 (−12.7, +1.1) |
| QuALITY F1 (50) | 64.5 | 64.9 | 58.6 | 60.6 | 61.4 ± 3.3 | −3.1 (−14.9, +8.9) |
| QuALITY MC accuracy (50) | 74 | 74 | 64 | 72 | 70.0 ± 5.3 | −4.0 (−14.0, +6.0) |

- **No seed beats the no-tree row** on any measure; the negative result does not depend on the seed. QASPER is stable (F1 spread 2.2 points).
- **QuALITY swings by seed:** multiple-choice accuracy ranges 64–74 because its 50 dev questions sit on only 3 stories, so one different cluster moves many questions at once.
- **This qualifies the test result.** The −10 QuALITY multiple-choice drop on test (67 → 57, p = 0.018) was measured on the seed-42 tree only, and the seed-to-seed range on dev is just as large. The direction is consistent (no seed gains), but the size of the drop is not established; it should be read as "no gain, plausibly a loss", not as a significant loss.
- Seeds on test were not run (fast plan: 3 seeds on dev, 1 on test).

## Engineering notes

- **Ollama Vulkan runner memory grows about 8 GB per hour.** Its private memory rose from 2.7 GB to about 11 GB (3b) or 13.7 GB (7b) within an hour, pushing available RAM under 500 MB. Unloading the model (`keep_alive: 0`) resets it and Ollama reloads on demand; an hourly guard did this. No call was lost.
- The LLM log now records `done_reason`, so length-capped summaries are counted correctly (a first count of "0 capped" was wrong).
- LLM cache keys include the backend and Ollama version, so these caches don't carry over to another machine.

### 3b vs 7b summarizer (dev, 18 Nov)

`results/m4_summarizer_probe` and `results/m4_dev_raptor7b`; full write-up in `docs/M4_RAPTOR_REPORT.md` §4.

- **Summary level (the same 60 clusters):**
  - 7B summaries are shorter: 120 words against 136.
  - Reading the flagged items by hand: the 3b has 7 clear errors (4 misspelled names, 3 merged or invented relations); the 7B has 3 (invented numbers or dates where the passages had them stripped).
  - The automatic flag count favours neither model (0.53 vs 0.60 per summary), because most flags are harmless.
- **7B tree for the dev documents:** same clustering and seed; QASPER and QuALITY only.

| Dev, 100 questions | No tree | 3b tree (s42) | 7B tree |
| --- | --- | --- | --- |
| QASPER F1 | 51.4 | 44.8 | 50.9 (+6.1 vs 3b, p = 0.057 / 0.035; −0.5 vs no tree) |
| QASPER nDCG@10 | 53.9 | 45.1 | 43.6 (−10.2 vs no tree, p < 0.001) |
| QuALITY MC accuracy | 74 | 74 | 72 |

- On QASPER the better summaries win back the loss, but only tie the no-tree row.
- Summaries take the same share of context (about 30%), so crowding stays.
- QuALITY doesn't move.
- The 7B is 2.2× slower: about 31 h for all six trees, against 14.2 h. It doesn't change the decision below.

## Decision (owner, 17 Nov 2026)

**RAPTOR stays off by default** (`raptor.enabled: false` in `configs/base.yaml`; the full config won't turn it on). Rows 3 and 4 are reported as negative results, and rows 5–8 build on M3's row 4 (hybrid + rerank, no tree). The code, trees and results stay in the repo.

Why it failed, stage by stage (index, retrieval, rerank, context, answers, citations, abstention), and the 3b vs 7b summarizer comparison: `docs/M4_RAPTOR_REPORT.md`, with the trace in `results/m4_raptor_analysis` (`scripts/m4_raptor_analysis.py`).
