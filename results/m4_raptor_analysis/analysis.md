# Why RAPTOR did not help: per-stage trace (M4)

Test, first 100 questions per dataset plus custom 32; same questions in every run. Summary = a tree node above level 0. Answer-in-context is a normalised substring match of a gold answer.

## Index (seed-42 tree builds, qwen2.5:3b)

3745 summary calls; median prompt 1193 tokens → median summary 206 tokens (compression 5.6×); 26% hit the 256-token cap (of calls with done_reason logged); median 13.2 s per summary.

## Retrieval (row 3: hybrid over leaves + summaries)

| Dataset | Summaries in top 20 | Summary at rank 1 | Recall@5 row 2 → row 3 | Questions losing gold from top 5 | gaining |
| --- | --- | --- | --- | --- | --- |
| qasper | 22% | 17% | 61.4 → 47.6 | 17% | 0% |
| quality | 20% | 41% | nan → nan | nan% | nan% |
| custom | 36% | 38% | 29.6 → 20.4 | 11% | 0% |
| musique | 27% | 19% | 45.5 → 43.1 | 10% | 3% |
| hotpotqa | 33% | 34% | 79.1 → 70.9 | 16% | 0% |
| 2wiki | 39% | 34% | 66.9 → 61.8 | 11% | 0% |

## Context (the 5 passages the generator reads)

| Dataset | Summaries in context, row 3 | row 4 (after rerank) | Answer in context row 2 → row 3 | Answer only in a summary | Answer pushed out | Prompt tokens row 2 → row 3 |
| --- | --- | --- | --- | --- | --- | --- |
| qasper | 30% | 33% | 51% → 48% | 4% | 5% | 1801 → 1667 |
| quality | 35% | 38% | 4% → 3% | 0% | 1% | 2512 → 2134 |
| custom | 50% | 61% | 0% → 0% | 0% | 0% | 2045 → 1689 |
| musique | 27% | 31% | 36% → 39% | 4% | 2% | 1142 → 1217 |
| hotpotqa | 35% | 37% | 66% → 64% | 5% | 5% | 1137 → 1232 |
| 2wiki | 39% | 42% | 46% → 44% | 4% | 6% | 1262 → 1269 |

## Answers: F1 change row 3 − row 2 by how many summaries were packed (n, points)

| Dataset | 0 | 1 | 2 | 3+ | Win / loss / tie | Row 4 vs M3 row 4: win / loss / tie |
| --- | --- | --- | --- | --- | --- | --- |
| qasper | +0.0 (n=13) | +3.6 (n=41) | -2.6 (n=32) | -8.9 (n=14) | 17/17/66 | 18/18/64 |
| quality | +0.0 (n=11) | +0.0 (n=34) | -5.5 (n=29) | +1.6 (n=26) | 12/12/76 | 18/20/62 |
| custom | +0.0 (n=2) | -13.9 (n=6) | -3.9 (n=8) | -3.2 (n=16) | 6/11/15 | 8/7/17 |
| musique | -4.3 (n=23) | +9.9 (n=37) | -10.9 (n=24) | +14.8 (n=16) | 9/10/81 | 13/15/72 |
| hotpotqa | +0.0 (n=10) | +2.9 (n=35) | -10.5 (n=32) | +0.0 (n=23) | 7/10/83 | 8/12/80 |
| 2wiki | -7.1 (n=7) | -4.8 (n=23) | -1.9 (n=42) | -11.0 (n=28) | 9/12/79 | 8/11/81 |

By the highest tree level packed (n, points):

- qasper: L0 only +0.0 (n=13); L1 +2.3 (n=49); L2+ -4.6 (n=38)
- quality: L0 only +0.0 (n=11); L1 -4.0 (n=44); L2+ +1.3 (n=45)
- custom: L0 only +0.0 (n=2); L1 -10.2 (n=11); L2+ -2.8 (n=19)
- musique: L0 only -4.3 (n=23); L1 +7.7 (n=52); L2+ -2.4 (n=25)
- hotpotqa: L0 only +0.0 (n=10); L1 +3.7 (n=55); L2+ -12.6 (n=35)
- 2wiki: L0 only -7.1 (n=7); L1 -1.9 (n=62); L2+ -12.2 (n=31)

## Citations

| Dataset | Cites to a summary, row 3 / row 4 | Cites with no page or span, row 3 | F1 when citing a summary (n) | F1 otherwise (n) |
| --- | --- | --- | --- | --- |
| qasper | 38% / 40% | 38% | 43.8 (61) | 38.3 (39) |
| quality | 32% / 34% | 32% | 46.3 (57) | 42.0 (43) |
| custom | 57% / 66% | 57% | 25.6 (22) | 59.1 (10) |
| musique | 20% / 21% | 20% | 31.2 (24) | 29.3 (76) |
| hotpotqa | 22% / 25% | 22% | 64.2 (35) | 52.2 (65) |
| 2wiki | 30% / 30% | 30% | 30.5 (35) | 27.2 (65) |

## Question types: F1 change row 3 − row 2 (n, points)

- quality: hard -8.4 (n=57); not hard +8.4 (n=43)
- custom: 1-hop -8.6 (n=21); 2-hop +2.6 (n=9); 3-hop -4.3 (n=2)
- musique: 2-hop -0.5 (n=56); 3-hop +2.6 (n=26); 4-hop +11.1 (n=18); type=2hop -0.5 (n=56); type=3hop1 +6.0 (n=21); type=3hop2 -11.6 (n=5); type=4hop1 +0.0 (n=10); type=4hop2 +40.0 (n=5); type=4hop3 +0.0 (n=3)
- hotpotqa: 2-hop -2.4 (n=100); type=bridge -0.3 (n=83); type=comparison -12.3 (n=17)
- 2wiki: 2-hop -5.4 (n=100); type=bridge_comparison -3.7 (n=27); type=comparison -19.9 (n=23); type=compositional -2.9 (n=35); type=inference +7.5 (n=15)

## QuALITY multiple choice, M3 row 4 → row 4

14 questions flipped right → wrong, 4 wrong → right; 14 of the 14 losses had a summary in context (mean 1.9 of 5 passages).

## Examples: right without the tree, wrong with it, citing a summary

**qasper qasper:84ee6180d3267115ad27852027d147fb86a33135**: How much time takes the training of DistilBERT?
- gold ['on 8 16GB V100 GPUs for approximately 90 hours', '90 hours']; row 2 said '90 hours'; row 3 said '4000 examples per batch'
- cited L1 summary: "DistilBERT is a distilled version of BERT, 40% smaller, 60% faster, and retaining 97% of its language understanding capabilities. This work introduces a triple loss combining language modeling, distillation, and cosine-distance losses to achieve this. DistilBERT outperforms BERT on various downstream tasks, including IMDb sentiment classification and SQuAD question answering, with only 40% fewer parameters. The model is initialized from a teacher model and trained with gradient accumulation up to 4000 examples per batch. Distillation is applied during pre-training and fine-tuning. The study de"

**qasper qasper:38b29b0dcb87868680f9934af71ef245ebb122e4**: Do they annotate their own dataset?
- gold ['No', 'No']; row 2 said 'No'; row 3 said 'Yes'
- cited L1 summary: "To compare the effectiveness of three models on the AZ task, they were trained on a common ACL dataset. Word2vec embeddings were also trained with varying parameters. The impact of different domains was evaluated by training the first model on various corpora. Table. TABREF12 lists the characteristics of word embeddings based on different models and datasets. In an imbalanced dataset, synthetic Minority Over-sampling Technique (SMOTE) was applied to the original dataset to address class imbalance. Table. TABREF16 shows the distribution of rhetorical categories in the test dataset, with OWN and"

**quality quality:61430_X9N4VIUX_7**: Why were Jorgenson and Ganti not put to death?

Options:
(A) It was never ordered
(B) They had intellectually outsmarted the Thrid by making it seem a mistake to kill them
(C) They had ally Witnesses in the government that secretly kept them alive
(D) They proved to be useful in their resourcefulness

Start your answer with the letter of the correct option, e.g. "(B) ...".
- gold ['It was never ordered']; row 2 said 'It was never ordered'; row 3 said 'A'
- cited L1 summary: "Jorgenson and Ganti, stranded on a barren island, find themselves in a dire situation with limited resources. They discover a cave and learn that they are prisoners, with food and water being dropped by a copter only when no one picks it up. Ganti theorizes that the copter will stop dropping supplies if no one claims them. Jorgenson, a business man, and Ganti, a Thrid, devise a plan to make the copter land by causing a mistake. They practice with makeshift weapons and rope, eventually convincing the copter to land. However, their experiments lead to the copter dropping supplies again, and they"

**quality quality:51256_MZNDC998_9**: What was Boris Knackenpast's great accomplishment?

Options:
(A) Evading capture by the Americans
(B) Evading capture by the Russians
(C) Pretending to be a robot
(D) Nobel prize for literature

Start your answer with the letter of the correct option, e.g. "(B) ...".
- gold ['Nobel prize for literature']; row 2 said 'Nobel prize for literature'; row 3 said 'Evading capture by the Russians'
- cited L1 summary: "Colonel Pashkov, now known as Gospodin Pashkov, arrives at the Gustavus room in Moscow's Palace, where he is greeted by King Gustavus IX, Professor Kristin, and Monsieur Fanti, the make-up surgeon. Pashkov is recognized as Colonel James, but his appearance has been altered. He is summoned to meet with the Nobel Committee, but is interrupted by a manservant who informs him that Boris Knackenpast is waiting in the Gustavus room. Pashkov enters the room to find Knackenpast, a robot, being operated by Monsieur Fanti. Pashkov is suspicious of the situation and decides to take action, ordering the g"

**quality quality:22958_8T1HU0MH_5**: Why did they choose Mr. Braun to make the decision about the object?

Options:
(A) He was a good gambler
(B) He was going to run for Congress
(C) We do not get a reason
(D) His family was in the city so it mattered more

Start your answer with the letter of the correct option, e.g. "(B) ...".
- gold ['He was a good gambler']; row 2 said 'He was a good gambler'; row 3 said 'We do not get a reason'
- cited L1 summary: "In a chaotic week, Abner Longman Braun, a wealthy businessman, was in New York when the Polish freighter Ludmilla laid an egg in the harbor. The CIA, unaware of Braun's involvement, was tasked with identifying the egg. Harry Anderton, the Harbor Defense chief, initially identified the egg as a defective compartment bulkhead. Braun's involvement was only discovered after the news broke, despite the CIA's usual method of entering stories at the pay-off. The egg's origin remained a mystery, as Braun's team could not determine if it was a bomb or a harmless object. Braun's background, including hi"

**quality quality:63523_STSHLFEA_3**: What is the Oans' unusual advantage?

Options:
(A) They have the human's fire weapons. 
(B) They emit flames. 
(C) The strength of their arms. 
(D) Their eyes cut the night. 

Start your answer with the letter of the correct option, e.g. "(B) ...".
- gold ["They have the human's fire weapons. "]; row 2 said "They have the human's fire weapons"; row 3 said 'A'
- cited L1 summary: "Her arms laden with Manno and wild berries, she promised to provide a well-stocked table. Ro warned her not to wander, fearing her next meeting with the Oan would be less fortunate. Charlotte slept beside Carlson, but now that daylight had arrived, she moved away shyly. Ro noticed Grimm seemed annoyed by this. Ro suggested the Oan were cowards only because of their weapons, and they could make more fire sticks. Grimm disagreed, stating they needed more advanced machinery. Ro challenged him, accusing him of cowardice. The professor proposed using the space sphere, which was guarded by the Oan. "

