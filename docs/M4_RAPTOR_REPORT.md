# Why RAPTOR did not work in APEX

M4 report, 17 Nov 2026; section 4 (3b vs 7b summarizer) added 18 Nov. Data: `results/m4_raptor_analysis` (per-stage trace, `scripts/m4_raptor_analysis.py`), the row 3 and row 4 runs, `results/m4_tree_search_dev`, and the trees in `results/m4_trees`. Numbers are on the test split (first 100 questions per dataset plus custom 32), paired on the same questions, generator qwen2.5:7b at temperature 0.

**Owner decision (17 Nov): RAPTOR stays off by default.** Rows 5–8 build on M3's row 4 (hybrid + rerank, no tree). The tree code, the trees and these results stay in the repo as a reported negative result.

## 1. Verdict

The tree was meant to fix naive RAG's first failure: questions about a whole document or theme that no single chunk answers. It didn't:

- **Row 3 vs row 2** (tree added to hybrid retrieval): F1 −5.4 to +2.4 points, none significant.
- **Row 4 vs M3 row 4** (tree added under the reranker): F1 −3.3 to +2.1 points, none significant. **QuALITY multiple-choice accuracy fell from 67% to 57%** (p = 0.018 on the seed-42 tree), on the long-story benchmark where RAPTOR should help most.
- **3 seeds on dev:** no seed beats the no-tree row on QASPER or QuALITY (QASPER F1 −4.4 to −6.7; QuALITY multiple-choice 64–74% against 74% without the tree). QASPER is stable across seeds, but QuALITY swings by up to 10 points, as much as the test drop. So the negative result holds for every seed, while the QuALITY drop's size is not established.

In short, summaries take about a third of the five passages the model reads. They are on-topic but vague, so they push out the specific passages most questions need. A smaller model wrote them and some contain errors. They also make citations coarser. A 7B summarizer removes the name and relation errors and wins back the QASPER loss, but it only ties the no-tree row (section 4). The rest of this report traces that stage by stage.

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
- On QuALITY, a summary in context goes with the multiple-choice losses: all 14 questions that flipped from right to wrong (M3 row 4 → row 4) had summaries in context, an average of 1.9 of 5 passages. Only 4 flipped the other way. How many flip depends on the tree's seed (dev: 64–74% accuracy across 3 seeds), because QuALITY's questions cluster on a few stories.

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

### 2.8 Seed stability (dev, `docs/M4_NOTES.md` 3-seed check)

| | No tree | Seed 42 | Seed 7 | Seed 13 |
| --- | --- | --- | --- | --- |
| QASPER F1 | 51.4 | 44.8 | 46.2 | 47.0 |
| QuALITY MC accuracy | 74% | 74% | 64% | 72% |

The tree's clustering is random (UMAP and the GMM take a seed). Every seed is at or below the no-tree row, so the failure is not an unlucky tree. But on QuALITY the seed alone decides whether the tree costs 0 or 10 points.

### 2.9 Cost and latency

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

APEX uses a 3B summarizer, a 5-passage context and a 7B reader. These differences most plausibly explain why our result doesn't reproduce theirs. Of the three, the summarizer is the one we can test cheaply. Section 4 tests it: a 7B summarizer brings QASPER back to the no-tree level but not above it.

## 4. qwen2.5:3b vs qwen2.5:7b as the summarizer

Measured 18 Nov 2026. Data: `results/m4_summarizer_probe` (summary-level probe), the 7B trees `bench_qasper__raptor7b_s42` and `bench_quality__raptor7b_s42` (`configs/ablations/row3_raptor7b.yaml`, logs in `results/m4_trees`), and the dev run `results/m4_dev_raptor7b` with `compare_vs_3b_s42.md` and `compare_vs_row2.md`.

**Short answer:** the 7B writes shorter, cleaner summaries and wins back the QASPER loss (F1 44.8 → 50.9, against 51.4 with no tree). It does not make the tree a gain: on QASPER it ties the no-tree row, on QuALITY nothing changes, and retrieval ranking is just as low. Summary errors explain part of the QASPER loss. Crowding and the question mismatch explain the rest, and a better summarizer doesn't fix those.

### 4.1 Cost

| | qwen2.5:3b (used) | qwen2.5:7b |
| --- | --- | --- |
| Why | Your fast-plan choice, 13 Nov | The generator; same family, bigger |
| Speed on this iGPU | median 13.2 s per summary | median 29.2 s per summary in the probe; 27 s in the QASPER tree build (2.2×) |
| QASPER dev tree (46 papers, 216 summaries) | about 47 min (estimated from the median) | 99 min (measured) |
| All six trees | about 14.2 h (measured) | about 31 h (estimated from the 2.2×) |
| Ollama runner memory | grows from 2.7 to about 11 GB per hour | grew to 13.3 GB within an hour on 18 Nov; one unload needed |
| Weights | 1.9 GB | 4.7 GB |

### 4.2 Summary level: the same 60 clusters, both models

10 level-1 clusters per corpus from the seed-42 trees, `random.Random(0)`. Both models got the same prompt, temperature 0, the 256-token cap and the same trimming. The 3b output matched the stored tree summary in 60 of 60 clusters, so the 3b side is the tree that was evaluated.

| Corpus | Words 3b / 7b | Hit the cap 3b / 7b | Flagged details per summary 3b / 7b | 7B s per summary |
| --- | --- | --- | --- | --- |
| QASPER | 150 / 111 | 10% / 0% | 0.20 / 0.50 | 22.7 |
| QuALITY | 137 / 108 | 0% / 0% | 0.20 / 0.20 | 29.9 |
| custom | 146 / 120 | 20% / 10% | 1.10 / 1.00 | 27.8 |
| MuSiQue | 136 / 148 | 10% / 60% | 0.70 / 1.10 | 33.5 |
| HotpotQA | 140 / 131 | 40% / 40% | 0.40 / 0.40 | 30.5 |
| 2Wiki | 109 / 104 | 30% / 40% | 0.60 / 0.40 | 25.6 |
| all 60 | 136 / 120 | 18% / 25% | 0.53 / 0.60 | 29.2 |

*Flagged details* are numbers and capitalized names in a summary that appear nowhere in its passages: an automatic check. By that count the 7B looks no better, 0.60 against 0.53 per summary, and 45% against 30% of summaries with any. **Reading the flagged items by hand reverses that.** Most flags are harmless:
- an abbreviation or its expansion: "NMT", "GRUs", "Neural Machine Translation";
- a title added to a name that is in the passages: "Professor Arnold Parnell", "CEO Bill Hewlett";
- an adjective form: "Swedish", "Filipino";
- correct general knowledge: "Telemachus", "World War II", the French and Indian War "1754–1763".

The clear errors were these:

| | qwen2.5:3b | qwen2.5:7b |
| --- | --- | --- |
| Misspelled names | 4: "Donna Meagow" (Meagle), "Sue Effnesss" (Fennessy), "Mount Palpamar" (Palomar), "Anne Devers-Mills" (Devereux-Mills) | 0 |
| Merged or invented relations | 3: "Tom Bernan" (Tom and Dallas Bernan are two people); "Jerry Jobs, Steve Jobs' brother" (Jerry York, a board member); Demeter as the mother of Pelias and Neleus | 0 found |
| Invented numbers or dates | 0 | 3: Oulujärvi's area as "112.5 square kilometers" and East Lake's depth as "12.2 meters" (the MuSiQue passages have these numbers stripped, and the 7B filled the gaps); the option backdating in "December 2001" (the passage says October) |
| Total clear errors in 60 summaries | 7 | 3 |

So the two models fail differently:
- The 3b garbles names and invents relations between the people it names. That is the "his son, Eve" kind of error from 2.1.
- The 7B keeps names and relations right, but fills missing numbers from its own knowledge, sometimes wrongly. The MuSiQue cap hits come from the 7B listing more facts per cluster (60% against 10%).

Each error and what the passages say is listed in `results/m4_summarizer_probe/hand_check.md`. This is a reading of the flagged items only, 60 summaries per model. Relation errors between names that do appear in the passages are only counted where a flag led to them, so both columns are lower bounds. M5's NLI model can score this properly.

### 4.3 End to end: a 7B tree for the dev documents

Same tree shape as the 3b seed-42 tree: the same clustering and seed, 216 QASPER summaries (39 at level 2+) and 8 QuALITY summaries. Only the summary text differs.
- 7B summaries are about 20% shorter: QASPER 113 words against 144, QuALITY 103 against 131.
- 2 of 224 hit the cap.

Then the row 3 config on the same 100 dev questions, the 7B generator, temperature 0:

| | No tree (row 2) | 3b tree (seed 42) | 7B tree | 7B − 3b | 7B − no tree |
| --- | --- | --- | --- | --- | --- |
| QASPER F1 | 51.4 | 44.8 | 50.9 | **+6.1** (−0.2, +13.1), p = 0.057 / 0.035 | −0.5 (−8.4, +6.9), p = 0.90 |
| QASPER Recall@10 | 97.0 | 85.9 | 88.0 | +2.2, p = 0.74 | **−8.9** (−16.7, −2.2), p = 0.004 |
| QASPER nDCG@10 | 53.9 | 45.1 | 43.6 | −1.5, p = 0.40 | **−10.2**, p < 0.001 |
| QuALITY F1 | 64.5 | 64.9 | 58.4 | −6.6 (−19.3, +6.0), p = 0.30 | −6.2, p = 0.32 |
| QuALITY MC accuracy | 74 | 74 | 72 | −2.0, p = 0.85 | −2.0, p = 0.86 |
| Prompt tokens per question (QASPER / QuALITY) | 1,820 / 2,530 | 1,660 / 2,214 | 1,613 / 2,221 | | |

p values are paired bootstrap / Wilcoxon.

How much of the context the summaries take:

| | QASPER 3b / 7B | QuALITY 3b / 7B |
| --- | --- | --- |
| Summaries in the top 20 | 23% / 23% | 18% / 17% |
| Summaries among the 5 passages the model reads | 30% / 31% | 27% / 23% |
| Questions with at least one summary in context | 88% / 90% | 96% / 82% |
| Citations that point to a summary | 41% / 37% | 26% / 15% |

What this shows:
- **On QASPER the summarizer was most of the answer-level loss.** The 7B summaries take the same share of context as the 3b ones, so the gain comes from better summary text, not from fewer summaries. It is borderline significant: Wilcoxon p = 0.035, bootstrap p = 0.057.
- **Even so, the 7B tree only ties the no-tree row; it does not beat it.** Retrieval ranking stays about 10 nDCG points lower, because summaries still take original passages' places (root cause 2). A good summary of a paper still doesn't answer a question about one table (root cause 1).
- **On QuALITY the 7B tree changes nothing measurable.** MC accuracy is 72 against 74, within the 64–74 seed spread from 2.8. Only 8 summaries cover the 3 dev stories, so this check is small.
- **The QuALITY F1 drop (−6.6) is mostly the letter-only answer effect** noted in section 3. F1 penalizes "B" against the option text; MC accuracy, which handles that, hardly moves.

### 4.4 Verdict on the summarizer

| Root cause (section 3) | Fixed by a 7B summarizer? | Measured |
| --- | --- | --- |
| 3. Summary quality | Partly | Name and relation errors 7 → 0 in 60 summaries; QASPER F1 +6.1 over the 3b tree. The 7B fills in missing numbers instead. |
| 1. Question mismatch | No | QASPER F1 ties the no-tree row; it doesn't beat it |
| 2. Crowding of the 5-passage context | No | Same summary share in context (30–31% on QASPER); nDCG@10 −10.2 against no tree |
| 4. Citation granularity | Slightly | Fewer citations point to summaries (41 → 37% QASPER, 26 → 15% QuALITY), but those still have no page or span |

Building all six trees with the 7B would take about 31 h on this iGPU for no expected gain over the no-tree row. **This supports keeping RAPTOR off.** If the tree is revisited (section 5), use the 7B summarizer. The extra time is worth it for the quality, but on its own it doesn't fix the design.

## 5. What might make a tree pay, if it is revisited (not planned)

- **A quota per level:** at most one summary in the 5 slots, or summaries only in addition to the 5 leaves.
- **Use summaries for routing only:** the M6 router could send only questions it marks as thematic to the tree.
- **Expand on citation:** when a summary is retrieved, put its children in the context instead of the summary, so citations stay page-level.
- **A context budget in tokens, as in the paper:** about 2,000 tokens instead of 5 passages.
- **A better summarizer:** see section 4.

Each would need its own runs. None is in the M5–M8 plan.
