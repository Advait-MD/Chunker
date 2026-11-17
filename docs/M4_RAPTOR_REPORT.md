# Why RAPTOR did not work in APEX

M4 report, 17 Nov 2026. Data: `results/m4_raptor_analysis` (per-stage trace, `scripts/m4_raptor_analysis.py`), the row 3 and row 4 runs, `results/m4_tree_search_dev`, and the trees in `results/m4_trees`. Numbers are on the test split (first 100 questions per dataset plus custom 32), paired on the same questions, generator qwen2.5:7b at temperature 0.

**Owner decision (17 Nov): RAPTOR stays off by default.** Rows 5–8 build on M3's row 4 (hybrid + rerank, no tree). The tree code, the trees and these results stay in the repo as a reported negative result.

## 1. Verdict

The tree was meant to fix naive RAG's first failure: questions about a whole document or theme that no single chunk answers. It didn't:

- **Row 3 vs row 2** (tree added to hybrid retrieval): F1 −5.4 to +2.4 points, none significant.
- **Row 4 vs M3 row 4** (tree added under the reranker): F1 −3.3 to +2.1 points, none significant. **QuALITY multiple-choice accuracy fell from 67% to 57%** (p = 0.018), the one significant answer-level effect, on the long-story benchmark where RAPTOR should help most.

In short, summaries take about a third of the five passages the model reads. They are on-topic but vague, so they push out the specific passages most questions need. A smaller model wrote them and some contain errors. They also make citations coarser. The rest of this report traces that stage by stage.

## 2. Stage by stage

### 2.1 Index: building the tree

| | Value |
| --- | --- |
| Summaries | 3,668 over 18,830 leaves (one per 5–8 children) |
| Build time | about 14.2 h on the iGPU, median 13.2 s per summary |
| Compression | median 1,193 prompt tokens → 206-token summary (5.6×) |
| Hit the 256-token cap | 24–26%; 833 trimmed back to the last full sentence |

Effects:
- **A 5.6× compression keeps the gist and drops specifics**: numbers, dates, names of minor entities, who did what to whom. Those specifics are what most benchmark questions ask about.
- **The summarizer, qwen2.5:3b, makes errors.** Observed examples:
  - wrong relations ("his wife, Lisa Brennan-Jobs", "his son, Eve");
  - a character misread: "Knackenpast, a robot", when Knackenpast is a person whom a robot imitates;
  - a summary that stops mid-plot because of the length cap, before the trim existed.
  
  How often this happens is measured in section 4 (pending) and in M5.
- **Cost:** 14.2 h once. It is cached, but any change to the corpus, the seed or the prompt means a rebuild.

### 2.2 Retrieval: what the tree puts in the candidate list

Row 3 runs the same hybrid retriever over leaves and summaries together (collapsed tree).

| Dataset | Summaries in top 20 | Summary ranked first | Recall@5, row 2 → row 3 | Questions that lose a gold passage from the top 5 | that gain one |
| --- | --- | --- | --- | --- | --- |
| QASPER | 22% | 17% | 61.4 → 47.6 | 17% | 0% |
| QuALITY | 20% | 41% | (no gold passages) | | |
| custom | 36% | 38% | 29.6 → 20.4 | 11% | 0% |
| MuSiQue | 27% | 19% | 45.5 → 43.1 | 10% | 3% |
| HotpotQA | 33% | 34% | 79.1 → 70.9 | 16% | 0% |
| 2Wiki | 39% | 34% | 66.9 → 61.8 | 11% | 0% |

**Why summaries rank so high:**
- A summary packs the key terms of 5–8 passages into about 200 tokens.
- For BM25, that means many query words in a short text, which BM25 rewards. The hybrid weights BM25 at 0.7.
- For dense search, an average of several passages sits close to many queries about that document.
- So summaries act as "hubs" that score well for almost any question about their topic.

**Effect:**
- Gold passages are pushed out of the top 5 for 10–17% of questions, and almost none gain one.
- Part of the Recall@5 drop is built into how we score: a retrieved summary counts as a miss, because gold evidence is original passages. Answer-level results (2.5) are the fair test.
- Traversal search does even worse than collapsed search on all four datasets with gold passages (Recall@10, dense-only both: QASPER 85.5 vs 77.9, HotpotQA 53.3 vs 41.1). It commits to a few top-level summaries and only searches below them, so a wrong early choice can't be recovered.

### 2.3 Rerank: does the cross-encoder filter summaries out?

No. It keeps slightly *more* of them:

| Dataset | Summaries in the top 20 | in the top 5 before rerank (row 3) | after rerank (row 4) |
| --- | --- | --- | --- |
| QASPER | 22% | 30% | 33% |
| QuALITY | 20% | 35% | 38% |
| custom | 36% | 50% | 61% |
| MuSiQue | 27% | 27% | 31% |
| HotpotQA | 33% | 35% | 37% |
| 2Wiki | 39% | 39% | 42% |

**Why:**
- The reranker (bge-reranker-v2-m3) scores how relevant a passage is to the question.
- A summary of the right document mentions the question's topic and entities, so it scores as relevant even when it lacks the specific answer.
- The reranker's training data is query–passage relevance, not "contains the answer".

**Effect:** the reranker still helps on top of the tree, by picking better leaves:
- QASPER row 4 vs row 3: +6.9 F1 (p = 0.011).
- HotpotQA: +7.7 (p = 0.08).

But it can't undo the crowding, so row 4 only catches up to M3's row 4; it doesn't beat it.

### 2.4 Context: the five passages the model reads

| Dataset | Questions with at least one summary in context | Answer string in context, row 2 → row 3 | Answer only in a summary | Answer pushed out of context | Prompt tokens, row 2 → row 3 |
| --- | --- | --- | --- | --- | --- |
| QASPER | 87% | 51% → 48% | 4% | 5% | 1,801 → 1,667 |
| QuALITY | 89% | (answers are options) | | | 2,512 → 2,134 |
| custom | 94% | (long-form answers) | | | 2,045 → 1,689 |
| MuSiQue | 77% | 36% → 39% | 4% | 2% | 1,142 → 1,217 |
| HotpotQA | 90% | 66% → 64% | 5% | 5% | 1,137 → 1,232 |
| 2Wiki | 93% | 46% → 44% | 4% | 6% | 1,262 → 1,269 |

**Effects:**
- The context holds a fixed 5 passages, and summaries fill 27–50% of them.
- On the long-document sets, the model also sees **less evidence text in total**. QuALITY prompts are 15% shorter, because summaries are shorter than the passages they replace.
- About as many questions lose the answer from context (2–6%) as gain one that only a summary contains (4–5%), so the trade is roughly zero on fact questions.
- On QuALITY, a summary in context goes with the multiple-choice losses: all 14 questions that flipped from right to wrong (M3 row 4 → row 4) had summaries in context, an average of 1.9 of 5 passages. Only 4 flipped the other way.

### 2.5 Answers: when do summaries hurt?

F1 change, row 3 − row 2, grouped by how many summaries reached the context (points; n questions):

| Dataset | 0 summaries | 1 | 2 | 3+ |
| --- | --- | --- | --- | --- |
| QASPER | 0.0 (13) | **+3.6** (41) | −2.6 (32) | **−8.9** (14) |
| QuALITY | 0.0 (11) | 0.0 (34) | −5.5 (29) | +1.6 (26) |
| custom | 0.0 (2) | −13.9 (6) | −3.9 (8) | −3.2 (16) |
| MuSiQue | −4.3 (23) | **+9.9** (37) | −10.9 (24) | +14.8 (16) |
| HotpotQA | 0.0 (10) | **+2.9** (35) | **−10.5** (32) | 0.0 (23) |
| 2Wiki | −7.1 (7) | −4.8 (23) | −1.9 (42) | −11.0 (28) |

Grouped by the highest tree level in context:

| Dataset | Level 1 only | Level 2 and above |
| --- | --- | --- |
| HotpotQA | +3.7 | **−12.6** |
| 2Wiki | −1.9 | **−12.2** |
| QASPER | +2.3 | −4.6 |
| MuSiQue | +7.7 | −2.4 |

The groups are small, so read these as patterns, not significant effects. Even so, they are consistent:
1. **One low-level summary is mildly useful** on three of six datasets. Two or more, or any higher-level summary, hurts. Higher levels summarize summaries, so they are vaguer and repeat any earlier error.
2. **Comparison questions are hurt most:**

   | Question type | F1 change |
   | --- | --- |
   | HotpotQA comparison | −12.3 |
   | 2Wiki comparison | −19.9 |
   | HotpotQA bridge | −0.3 |
   | 2Wiki inference | +7.5 |

   A comparison needs two exact attributes ("which was born first?"). A summary merges entities and drops dates and numbers.
3. **Harder QuALITY questions are hurt:** "hard" questions −8.4, the others +8.4.
4. **The one place synthesis seems to help is long chains:** MuSiQue 4-hop questions +11.1 (n = 18), 3-hop +2.6, 2-hop −0.5. A summary that already links several passages can stand in for a missing hop. The sample is too small to claim more.

Examples where the answer was right without the tree and wrong with it, and the wrong answer cited a summary:
- **QASPER**, "How much time takes the training of DistilBERT?" Gold: "90 hours". Row 3 answered "4000 examples per batch". The level-1 summary mentions the batch size but not the training time.
- **QASPER**, "Do they annotate their own dataset?" Gold: "No". Row 3 said "Yes". The summary describes the datasets in a way that suggests the authors built one.
- **QuALITY**, "What was Boris Knackenpast's great accomplishment?" Gold: the Nobel prize. Row 3 picked "evading capture by the Russians". The summary calls Knackenpast "a robot" and is about the escape plot.
- **QuALITY**, "Why did they choose Mr. Braun to make the decision?" Gold: "he was a good gambler". Row 3 picked "we do not get a reason". The summary covers the plot but leaves out the detail.

### 2.6 Citations

| Dataset | Citations pointing to a summary, row 3 / row 4 | Citations with no page or span |
| --- | --- | --- |
| QASPER | 38% / 40% | 38% |
| QuALITY | 32% / 34% | 32% |
| custom | 57% / 66% | 57% |
| MuSiQue | 20% / 21% | 20% |
| HotpotQA | 22% / 25% | 22% |
| 2Wiki | 30% / 30% | 30% |

A summary has no page or character span. For the pooled multi-hop trees it has no source file either; its source is `__raptor__`. The spec asks every answer for "its citations (source file plus page or span)". A citation to a summary resolves to a stored node, so it passes the letter of the criterion, but a reader can't check it against the document. On custom, the owner's own corpus, two thirds of row 4's citations are of this kind.

F1 is higher when the answer cites a summary on four datasets (for example HotpotQA 64.2 vs 52.2). That is a selection effect, not a benefit: the model cites summaries on broader questions it can answer anyway. On custom it is the reverse: 25.6 when citing a summary vs 59.1 otherwise.

### 2.7 Abstention

| Dataset | Unanswerable questions | Abstention recall, row 2 → row 3 | M3 row 4 → row 4 |
| --- | --- | --- | --- |
| 2Wiki | 11 | **91% → 45%** | 73% → 73% |
| MuSiQue | 10 | 70% → 80% | 50% → 40% |
| HotpotQA | 9 | 67% → 67% | 56% → 78% |
| custom | 5 | 80% → 100% | 100% → 100% |

The counts are small. On 2Wiki, the tree made the model answer 5 more of its 11 unanswerable questions. A summary always offers plausible related text even when the facts are missing, and that encourages a guess. M5's evidence filter is the stage meant to catch this.

### 2.8 Cost and latency

- **LLM calls:** one per question in every row.
- **Prompt tokens:** −15% on QuALITY and custom, up to +8% on HotpotQA.
- **Latency:** no reliable comparison. Part of row 3 ran under irregular power, and row 4 reused the reranker cache from M3.
- **The real cost is the 14.2 h build.**

## 3. Root causes, ranked

1. **Mismatch with the questions.** Most benchmark questions here, even on "long-document" sets, ask for one specific fact. RAPTOR's advantage is thematic questions, and our sets contain few of them. Where we can see deep synthesis (MuSiQue 4-hop), the tree helps slightly.
2. **A fixed 5-passage budget with no quota per level.** Collapsed search lets summaries compete with leaves for the same 5 slots. They win 27–50% of them, because of how BM25, dense search and the reranker all score them (2.2, 2.3).
3. **Summary quality.** The 3b summaries compress 5.6×, drop decisive details and sometimes get facts wrong. Higher levels compound this.
4. **Citation granularity.** Summaries can't be traced to a page, and pooled ones can't be traced to a file.
5. **Scoring artifacts.** These explain part of the retrieval drop, but not the answer-level results:
   - Retrieval metrics count every summary as a miss.
   - QuALITY F1 penalizes answers that give only the option letter, so multiple-choice accuracy is the metric to trust there.

**Differences from the RAPTOR paper's setup** (Sarthi et al., ICLR 2024), as reported in the paper:
- Its summaries came from a much larger model (GPT-3.5).
- Its collapsed search filled about 2,000 tokens of context by relevance, not a fixed 5 passages.
- Its strongest QuALITY results used GPT-4 as the reader.

APEX uses a 3B summarizer, a 5-passage context and a 7B reader. These differences most plausibly explain why our result doesn't reproduce theirs. Of the three, the summarizer is the one we can test cheaply (section 4).

## 4. qwen2.5:3b vs qwen2.5:7b as the summarizer

### What we know now

| | qwen2.5:3b (used) | qwen2.5:7b |
| --- | --- | --- |
| Why | Your fast-plan choice, 13 Nov: about 2× faster | The generator; same family, bigger |
| Speed on this iGPU | median 13.2 s per summary (measured) | about 25–30 s expected (2× the 3b) |
| All six trees | about 14.2 h (measured) | about 28 h (estimated) |
| Ollama runner memory | grows from 2.7 to about 11 GB per hour | grows from 5.6 to about 13.7 GB per hour (seen in rows 3–4) |
| Weights | 1.9 GB | 4.7 GB |
| Observed quality | relation errors, a misread character, details dropped (2.1, 2.5) | expected fewer factual errors; not measured yet |

### What a 7B summarizer could and could not fix

| Root cause | Fixed by a 7B summarizer? |
| --- | --- |
| 3. Summary quality | Yes, plausibly: fewer wrong relations and fewer misreadings |
| 1. Question mismatch | No: a better summary is still a summary |
| 2. Crowding of the 5-passage context | No, and it may get worse: better summaries score as even more relevant |
| 4. Citation granularity | No |

So a 7B summarizer is unlikely on its own to turn the result positive. It would tell us how much of the loss is summary errors and how much is the design.

### Measurement prepared (runs next, about 2.6 h of GPU)

1. **Summary-level probe** (`scripts/m4_summarizer_probe.py`, about 25 min):
   - 60 level-1 clusters (10 per corpus) summarized by both models, with the same prompt, cap and trimming;
   - compares length, how often the cap is hit, speed, and an automatic check for numbers and names in the summary that appear nowhere in its passages;
   - I will also hand-check a sample for relation errors.
2. **End-to-end check** (about 2 h): 7B trees for the QASPER and QuALITY dev documents (`configs/ablations/row3_raptor7b.yaml`), then the dev answers compared with the 3b tree on the same 100 questions (seed check run `m4_seed_dev_s42`).

*Results go here when the runs finish.*

## 5. What might make a tree pay, if it is revisited (not planned)

- **A quota per level:** at most one summary in the 5 slots, or summaries only in addition to the 5 leaves.
- **Use summaries for routing only:** the M6 router could send only questions it marks as thematic to the tree.
- **Expand on citation:** when a summary is retrieved, put its children in the context instead of the summary, so citations stay page-level.
- **A context budget in tokens, as in the paper:** about 2,000 tokens instead of 5 passages.
- **A better summarizer:** see section 4.

Each would need its own runs. None is in the M5–M8 plan.
