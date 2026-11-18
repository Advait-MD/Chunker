# Hand check of the flagged summary details (18 Nov 2026)

Each automatic flag in `pairs.jsonl` was looked up in the cluster's child passages. A flag counts as a clear error only when the passages contradict the summary or lack the detail it states. Harmless flags were not counted:
- abbreviations and expansions;
- titles added to names that appear;
- adjective forms;
- correct general knowledge;
- the same name with different hyphen spacing.

Relation errors between names that appear in the passages were found only where a flag pointed to them, so these counts are lower bounds.

## qwen2.5:3b: 7 clear errors in 60 summaries

| Corpus | Node | Summary says | Passages say |
| --- | --- | --- | --- |
| HotpotQA | …b6747aa5 | "Donna Meagow" | Donna Meagle |
| 2Wiki | …585012b8 | "Sue Effnesss" | Sue Fennessy |
| 2Wiki | …585012b8 | "Anne Devers-Mills" | Anne Devereux-Mills |
| QuALITY | …1130eb88 | "Mount Palpamar" | Mount Palomar |
| QuALITY | …6996257a | "Tom Bernan" | Tom and Dallas Bernan are two crew members |
| custom | …f4a998c6 | "Jerry Jobs, Steve Jobs' brother" | "Jerry" is Jerry York; no brother |
| custom | …c459e70d | Poseidon impregnates "Demeter", mother of Pelias and Neleus | the mother is not Demeter, and Demeter is not named in the passage |

## qwen2.5:7b: 3 clear errors in 60 summaries

| Corpus | Node | Summary says | Passages say |
| --- | --- | --- | --- |
| MuSiQue | …85998c91 | Oulujärvi "area of 112.5 square kilometers" | the number is stripped ("With an area of it is…"); the real area is about 928 km² |
| MuSiQue | …85998c91 | East Lake "average depth of 12.2 meters" | the number is stripped ("The average depth is ,") |
| custom | …80af2a7f | options "backdated in December 2001" | "an October date, when the stock was $18.30" |
