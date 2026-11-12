# M2 — Eval harness

Branch: `m2-eval-harness`, stacked on `m1-naive-baseline`. Done-when (spec): *baseline numbers exist on at least one long-document and one multi-hop dataset.*

## How to run

```bash
python scripts/fetch_benchmarks.py           # pinned HF downloads → data/benchmarks/ (gitignored), sha256 in MANIFEST.json
python -m apex.eval.suite index qasper musique quality hotpotqa 2wiki   # corpus files + one Chroma collection each
python -m apex.eval.suite run --config configs/base.yaml --split test   # → results/<run_id>/
python -m apex.eval.suite judge results/<run_id>                        # fill judge scores later, if run with --no-judge
python -m apex.eval.suite compare results/<A> results/<baseline>        # paired tests for a milestone PR delta
```

`make eval CONFIG=…` wraps the same commands, but `make` isn't installed on this Windows machine.

Each run writes `results/<run_id>/`:
- `run.json`: git SHA, config hash, the full config, model digests, Ollama version and backend.
- `metrics.json`, `table.md`.
- Per dataset, `per_query.jsonl`: the answer, citations, packed evidence, top-20 retrieved ids, per-query scores, judge output and cost.
- `traces/`, with one trace per query.

Runs resume: rows already written are skipped.

## Datasets (owner choice: all five, 250 test + 50 dev each)

| Dataset | Capability | Source (pinned commit) | Our dev / test | Retrieval scope | Unanswerable |
| --- | --- | --- | --- | --- | --- |
| QASPER | long-document QA over NLP papers | `allenai/qasper` parquet `06806e4` | official validation → dev, official test → test | the question's paper | native: 13/250 test |
| QuALITY | long stories, multiple choice | `tasksource/QuALITY` `4892b06` | validation split **by article** | the question's article | none (MC) |
| MuSiQue-Ans | 2–4 hop | `dgslibisey/MuSiQue` `c8f4f8c` | dev, random | pooled paragraphs of all 300 sampled questions | injected: 25/250 |
| HotpotQA (distractor) | 2-hop | `hotpotqa/hotpot_qa` `1908d6a` | validation, random | pooled | injected: 25/250 |
| 2WikiMultihopQA | 2-hop | `framolfese/2WikiMultihopQA` `fe713bf` | validation, random | pooled | injected: 25/250 |
| custom | owner's 4 PDFs | `eval/dataset/gold.jsonl` | all 32 → test | whole store | native: 5/32 |

- The sampled ids are committed in `apex/eval/splits/<name>.json` (seed 42), with the sha256 of the source files they were drawn from. A clean checkout plus `fetch_benchmarks.py` reproduces the exact questions.
- **Injected unanswerables** follow the MuSiQue-full construction. For a seeded 10% of the sampled pooled-set questions, the supporting paragraphs are removed from the corpus and the gold answer becomes "Unanswerable". A question is only chosen if its supporting paragraphs support no other sampled question. Caveat: the answer string can still appear in some other paragraph. This matters most for HotpotQA yes/no comparison questions.
- **Custom set:** its gold evidence was stored as old-chunker ids. `scripts/freeze_custom_gold.py` resolved them to text from `hierarchical_chroma_db/` and froze the result into `splits/custom.json`. **One correction was applied and recorded:** sh-002's gold chunk was a NeXT-computer passage. Its evidence now comes from sh-001's Road to React title page, which states the publication date. 36 of the 37 gold passages are matched by a new chunk at 82%+ shingle overlap. The 37th (sh-006) crosses a PDF page break; union recall (below) handles it.

## Metrics

All are in `apex/eval/metrics.py`, pure functions with unit tests.

| Group | Metrics | Notes |
| --- | --- | --- |
| Retrieval | Recall@5/10/20, MRR, nDCG@10 | Scored against gold **evidence text**, not chunk ids, so any chunker or tree can be evaluated. A chunk covers a gold passage when their word 5-gram shingles overlap by ≥ 50% of the smaller set. Recall also credits a passage covered by the union of the top-k chunks (gold split by a chunk boundary). QASPER takes the max over annotators' evidence sets. Computed from a separate untraced top-20 retrieval with the same query |
| Answer | EM and token F1 on the new `short_answer` field (SQuAD normalisation, max over aliases); **Acc** (normalised gold alias contained in the long or short answer); F1 on answerable questions only; *F1 long* on the full cited answer (M1-comparable); MC accuracy (QuALITY, plus its *hard* subset) | An abstention is scored as the answer "Unanswerable", as in QASPER's official evaluation, so correct abstentions earn credit. The custom set's gold answers are full sentences, so use *F1 long* there |
| Context | **answer in context**: a short gold alias (≤ 5 tokens, not yes/no) occurs in the evidence packed into the prompt | Deterministic context recall, replacing the judge's |
| Judge | faithfulness, citation precision and recall, answer relevance, context precision and recall | `apex/eval/judge.py`. **Off for M2** (`eval.judge_enabled: false`); see "Judge outcome" below |
| Abstention | precision, recall, false-abstain rate | "Unanswerable" is the positive class |
| Cost | LLM calls/query, tokens/query, latency p50/p95 (fresh generations only), % routed without an LLM | Router is n/a until M6. Judge calls are never counted |
| Rigour | 95% bootstrap CIs (1,000 resamples, seed 42) on every mean; `compare` gives paired-bootstrap and Wilcoxon p-values against a baseline run | Generation is temperature 0 with a fixed seed, so "3 seeds" only applies once a stochastic component exists (RAPTOR's GMM in M4) |

**Judge outcome: rejected for M2 (owner decision: move faithfulness and citation support to M5's NLI model).**

The judge went through three designs, each tested on real questions from the custom set:
1. **Two calls per question, per-sentence labels.** It claimed the context supports "Robin Wieruch wrote *The Road to React*" when none of the five passages mentions Wieruch or the book. It also labelled sentences "supported by their citations" with no supporting citation listed.
2. **One call, with every support label backed by a verbatim quote that is checked in code.** Unverified labels are overruled and counted. gemma3:4b quoted the *answer* instead of the passage, and left (sentence, citation) pairs out of a flat list.
3. **Citations nested under their sentence, plus one repair call that names the missing labels.** On 8 questions:
   - 2 of 8 (25%) were still incomplete after the repair call.
   - Context recall was 0 on all 8, including a correctly answered question whose gold passage was retrieved.
   - Answer relevance was 0 on two correct answers.
   - It took 22–74 s per question (more than generation), about 14 h for all sets.

The code is kept and tested: `quote_in`, the overrule counter, `JudgeIncomplete` handling, and a `judge` pass that can fill scores into an existing run. It also enforces a different model family from the generator. Re-enable it with a stronger judge or once the NLI scorer exists. Running the judge after generation, not inline, is required on this GPU: qwen2.5:7b and gemma3:4b don't fit in GPU memory together, and swapping them costs about 26 s per question.

## Deviations and [PROPOSED] choices to flag

1. **New `apex/eval/` package instead of extending `eval/`.** The old `eval/metrics.py` has no SQuAD normalisation, swallows scipy errors silently (rule 6), and scores retrieval by old-chunker ids. The old `eval/` is untouched and its tests still run.
2. **Benchmark corpora use a new `.jsonl` pre-segmented document format** (`apex/ingest/parsers.py: parse_jsonl_doc`): one `{section, text}` unit per line, through the same chunker, embedder and store as user files. Markdown would silently re-interpret paper text, for example `===` lines becoming headings, and drop HTML. Only benchmark ingests enable `.jsonl`; user ingests are unchanged.
3. **Wikipedia paragraphs are embedded as "title\ntext"**, the standard DPR-style input. Without it, a bridge entity's title wouldn't be searchable.
4. **`short_answer` added to the answer JSON (owner decision).** It holds the shortest exact answer, or the option letter for multiple choice, and EM/F1 score it, so they are comparable with RAPTOR or HippoRAG-style numbers. The cited-sentence `answer` is unchanged and still drives citations. This changes the M1 prompt, so the 32-question M1 smoke was re-run (`results/m1_smoke_ollama040/`). For QuALITY the option letter comes from `short_answer`; if that has no letter it is parsed from the answer, and `choice_source` records which.
5. **Ollama upgraded from 0.13.1 to 0.40.16 by the owner.** It was needed because every `gemma3:4b` tag requires a newer Ollama. 0.40 drops integrated GPUs by default ("dropping integrated GPU; to enable, set OLLAMA_IGPU_ENABLE=1"), so `OLLAMA_IGPU_ENABLE=1` is now set as a persistent user environment variable next to `OLLAMA_VULKAN=1`. qwen2.5:7b and gemma3:4b both load fully onto the Arc GPU (gemma 35/35 layers).
6. **The Ollama server version is now part of the generate cache key** and every log record. A runtime upgrade can change outputs, so a cached 0.13.1 answer must not count as a 0.40.16 result. Embedding keys still exclude it, so the indexes built on 0.13.1 are reused. Any drift between document vectors from 0.13.1 and query vectors from 0.40.16 is expected to be negligible but has not been measured.

## M1 smoke on the final setup (Ollama 0.40.16, `short_answer` prompt)

Same 32 custom questions as M1 (`results/m1_smoke_ollama040/`), compared with M1 on Ollama 0.13.1:

| Type | p50 | p95 | Abstained | Answers with an uncited sentence | Invalid ids |
| --- | --- | --- | --- | --- | --- |
| single-hop (16) | 15.4 → **22.2 s** | 20.5 → **27.2 s** | 4 → 2 | 3 → 2 | 0 |
| multi-hop (11) | 26.4 → 30.0 s | 39.6 → 36.2 s | 1 → 0 | 9 → 7 | 0 |
| unanswerable (5) | 19.6 → 22.8 s | 21.4 → 25.8 s | 5 → 5 | 0 | 0 |

- **Single-hop latency got about 35% worse**, so it is now further from the 10 s target. The Ollama upgrade and the longer prompt and output (the `short_answer` field) changed together, so this run cannot separate their effects.
- **A prompt regression was caught and fixed before the baseline.** The first `short_answer` prompt made qwen put the bare span into `answer` ("Telemachus", "Scylla", "1984") with no inline citation; 10 of 16 single-hop answers had an uncited sentence. Rule 6 now requires `answer` to always be complete cited sentences. A half-finished baseline run with the bad prompt was discarded.
- **Generation failures are now recorded, not fatal.** One QASPER question (in the discarded run) made qwen copy a passage until it hit `max_tokens=512`, which left invalid JSON. The generator now raises `GenerationError` ("truncated at max_tokens") and the eval runner records the question as a failure scoring 0. Failures are counted in the results table and left out of the latency figures. The CLI and API still raise.

## Results: naive baseline on the test split

Run `results/m2_baseline_naive_test/` (full table in `table.md`, per-question rows in `<dataset>/per_query.jsonl`), 12 Nov 2026, 14:41–20:40 IST.
- Setup: qwen2.5:7b on Ollama 0.40.16 (Vulkan, Arc iGPU), nomic-embed-text, flat chunks, dense top-k, 2,500-token context budget, M1 prompt plus `short_answer`.
- 1,282 questions, **0 generation failures**.
- `run.json` records git `bdce5cd-dirty`, because the M2 code was not committed yet; the M2 commit contains exactly the code that produced it.

All numbers are % with 95% bootstrap CIs.

| Dataset | n | EM | F1 | Acc | MC acc | Recall@10 | MRR | Answer in context | False-abstain rate | Abstain recall | Latency p50 / p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| QASPER | 250 | 30.0 (24.4–36.4) | 47.5 (42.4–52.7) | 37.6 | — | 88.5 (84.1–92.3) | 52.6 | 43.3 (n=104) | 5% | 15% (2/13) | 20.6 / 28.8 s |
| QuALITY | 250 | 47.6 | 51.7 | 66.8 | **67.2 (61.6–73.6)** | — | — | — | 0% | — | 22.7 / 29.4 s |
| MuSiQue | 250 | 16.0 (11.6–20.4) | 19.7 (15.0–24.4) | 21.6 | — | 26.7 (23.1–30.1) | 34.7 | 30.3 | 52% | 80% (20/25) | 12.7 / 20.4 s |
| HotpotQA | 250 | 33.2 (27.2–38.8) | 39.4 (33.5–44.6) | 42.4 | — | 56.7 (51.8–61.1) | 54.1 | 62.2 | 37% | 60% (15/25) | 12.0 / 16.5 s |
| 2Wiki | 250 | 14.0 (10.0–18.4) | 15.4 (11.1–19.8) | 21.2 | — | 16.9 (13.6–20.7) | 18.1 | 17.1 | 71% | 72% (18/25) | 12.6 / 20.7 s |
| custom | 32 | 15.6 | 36.3 (*F1 long* 45.7) | 15.6 | — | 21.6 (lower bound, see below) | 10.8 | — | 7% | 100% (5/5) | cached, see M1 smoke |

MuSiQue F1 by hops: 2-hop 27.3 (n=139), 3-hop 12.5 (n=65), 4-hop 6.7 (n=46).

**What it says**
- **Long-document retrieval is not the bottleneck; generation is.**
  - On QASPER, Recall@10 is 88.5 and Recall@20 98.2, yet F1 is 47.5.
  - On QuALITY (no retrieval metric: the gold evidence is unlabelled), MC accuracy is 67.2, with 0 abstentions on a 4-way task.
  - Gains here have to come from evidence selection and answering (M5), not recall.
- **Multi-hop is where naive dense retrieval breaks.**
  - Recall@10 is 26.7 on MuSiQue, 56.7 on HotpotQA and 16.9 on 2Wiki, and F1 falls with hop count.
  - The generator responds correctly by abstaining rather than guessing, but too often: the false-abstain rate is 52% / 37% / 71%.
  - Most of the lost F1 is abstentions on answerable questions whose evidence was never retrieved. This is the gap M3 (hybrid retrieval) and M7 (multi-hop) must close.
- **Abstention precision is low on the pooled sets (10–15%).** Almost all abstentions are false ones on answerable questions; recall of the injected unanswerables is 60–80%.
- **Latency misses the spec's 10 s single-hop target everywhere.** The p50 is 12–23 s; it is driven by prompt length (QuALITY has 2,578 tokens/question) and the ~8 tokens/s decode speed of a 7B model on the iGPU.
- **M2 done-when is met:** baseline numbers exist on two long-document sets (QASPER, QuALITY) and three multi-hop sets (MuSiQue, HotpotQA, 2Wiki).

**Caveats**
- **Custom-set retrieval scores are a lower bound.** Each question's gold is one passage chosen for the old chunker, while the four books often state the same fact in several places. For example, sh-003 is answered correctly ("Miyamoto Musashi", *F1 long* 0.88) from "Musashi wrote The Book of Five Rings…", which is not its gold passage, so it scores Recall@20 = 0. Use the custom set for answers (*F1 long*) and abstention, not for retrieval.
- **The custom answers are cache hits** from the `m1_smoke_ollama040` run: same prompt, model, backend and Ollama version. Their latency is therefore reported in the M1 smoke table above, not here.
- **EM/Acc on QuALITY score the option text against `short_answer`.** MC accuracy is the headline number there.
- **qwen mostly ignores rule 7 on QuALITY.** It wrote the option text instead of the letter in `short_answer` on 172 of 250 questions, so the letter was parsed from `answer`. On 7 questions no letter could be found, and they score as wrong. A prompt fix could raise MC accuracy by up to 2.8 points; it is not applied, because the baseline is frozen.
- **The lost F1 on multi-hop is mostly abstentions.** On answerable questions, abstentions cost 116 / 84 / 160 F1-units (MuSiQue / HotpotQA / 2Wiki), against 80 / 58 / 45 from wrong answers.
- **QASPER's "answer in context"** only covers the 104 questions with a short extractive gold. Its abstention numbers rest on 13 native unanswerables, so they are noisy.
- **The injected unanswerables** are the MuSiQue-full-style construction described above, not a native property of HotpotQA or 2Wiki.
- **Judge metrics are absent** (faithfulness, citation precision and recall). They arrive with the M5 NLI scorer, which will be run over these same `per_query.jsonl` rows.
