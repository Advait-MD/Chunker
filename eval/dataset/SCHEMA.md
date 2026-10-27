# APEX Evaluation Dataset Schema (`eval/dataset/SCHEMA.md`)

Each line in `dataset/gold.jsonl` is a self-contained JSON object representing a single evaluation test case.

## Field Definitions

| Field Name | Type | Description | Constraints |
| :--- | :--- | :--- | :--- |
| `id` | `str` | Unique case identifier (e.g. `sh-001`, `mh-005`, `un-002`). | Non-empty, unique across the dataset. |
| `question` | `str` | The input question presented to the RAG pipeline. | Non-empty natural language string. |
| `gold_answer` | `str` or `null` | The reference answer. Null if `answerable` is false. | Non-empty string when `answerable: true`; must be `null` when `answerable: false`. |
| `gold_chunk_ids` | `list[str]` | Ground-truth ChromaDB chunk IDs containing the necessary facts. | List of string chunk IDs. Must be empty (`[]`) when `answerable: false`. |
| `hops` | `int` | Cognitive reasoning demand / number of hops required to answer. | `1` for single-hop factual lookups; `2` or higher for multi-hop relational synthesis. |
| `answerable` | `bool` | Whether the corpus contains sufficient evidence to answer. | `true` for standard answerable queries; `false` for adversarial traps and out-of-corpus queries. |
| `status` | `str` (optional) | Verification status of the case. | `"VERIFIED"` if human-checked; `"TODO_HUMAN_REVIEW"` if candidate or unverified. |

## Invariants & Rules

1. **Unanswerable Abstention**:
   - If `answerable == false`, then `gold_chunk_ids` MUST be `[]` and `gold_answer` MUST be `null`.
   - The expected system behavior on unanswerable cases is **explicit abstention** (admitting lack of information or pointing out historical/logical impossibility).
2. **Dataset Composition**:
   - Minimum 30 total cases.
   - At least 10 multi-hop cases (`hops >= 2`).
   - At least 5 unanswerable/adversarial cases (`answerable == false`).
