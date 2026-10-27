# APEX RAG Evaluation Harness

This evaluation harness benchmarks APEX's retrieval quality, generation fidelity, and latency against structured ground-truth test cases across technical, historical, and multi-document corpora.

---

## 1. Directory Structure

```
eval/
├── README.md                 # How to run and configure the harness
├── NOTES.md                  # Inspection notes on APEX pipeline interface
├── calibration.md            # 20 audited cases for human judge verification
├── dataset/
│   ├── gold.jsonl            # 32 evaluation cases (single-hop, multi-hop, unanswerable)
│   └── SCHEMA.md             # Dataset field definitions and constraints
├── configs/
│   ├── bm25.yaml             # Pure lexical BM25 ranking
│   ├── vector.yaml           # Pure dense vector search
│   ├── hybrid.yaml           # Hybrid BM25 + dense vector with RAPTOR tree summaries
│   ├── hybrid_raptor_off.yaml# Hybrid retrieval without hierarchical summaries
│   └── hybrid_raptor_on.yaml # Full APEX hierarchical clustering pipeline
├── cache/                    # Content-addressed LLM response disk cache (gitignored)
├── harness.py                # Main runner: config hashing, pipeline query, judge scoring
├── metrics.py                # Retrieval (Recall@k, MRR), Generation, and Cost metrics
├── judge.py                  # Strict LLM-as-a-judge (Llama 3.2:1b) with JSON schema validation
├── report.py                 # Multi-run comparator and diff table generator
├── tests/
│   ├── test_metrics.py       # Hand-computed unit tests for metrics
│   ├── test_cache.py         # Response cache unit tests
│   └── test_harness_smoke.py # End-to-end smoke test with pipeline and judge stubs
└── runs/                     # Run artifacts keyed by 12-char canonical config hash
    └── <config_hash>/
        ├── cases.jsonl       # Raw per-case evaluation traces
        └── summary.json      # Aggregated metrics and configuration metadata
```

---

## 2. Prerequisites & Setup

1. **Ollama Server**: Ensure local Ollama is running (`ollama serve`).
2. **Pinned Models**:
   ```bash
   ollama pull gemma3:1b        # APEX generation model
   ollama pull llama3.2:1b      # Independent judge model (Hard Rule 5)
   ollama pull nomic-embed-text # Embedding model
   ```
3. **Python Dependencies**:
   `pytest`, `pyyaml`, `numpy` are pre-installed in the workspace.

---

## 3. Running Unit & Invariant Tests

All unit tests run locally without making network calls or modifying data:

```bash
python -m pytest eval/tests/
```

Test coverage includes:
- Hand-computed retrieval metrics (`recall@k`, `mrr`, hop splitting).
- Response cache idempotency, temperature sensitivity, and bypass flags.
- End-to-end smoke test with deterministic stubs.

---

## 4. Running the Harness

To evaluate APEX against a specific configuration:

```bash
# Full hierarchical RAG pipeline (RAPTOR on)
python -m eval.harness --config eval/configs/hybrid_raptor_on.yaml

# Ablation: Hybrid retrieval without RAPTOR cluster summaries
python -m eval.harness --config eval/configs/hybrid_raptor_off.yaml

# Pure BM25 lexical baseline
python -m eval.harness --config eval/configs/bm25.yaml

# Pure dense vector baseline
python -m eval.harness --config eval/configs/vector.yaml
```

### CLI Flags:
- `--config <path>` (required): Path to configuration YAML.
- `--dataset <path>` (optional): Path to evaluation dataset (default: `eval/dataset/gold.jsonl`).
- `--runs-dir <path>` (optional): Directory where run outputs are stored (default: `eval/runs`).
- `--no-cache`: Force fresh LLM generation and judge calls, bypassing disk cache.

---

## 5. Comparing Runs & Generating Diff Tables

To compare two configurations (e.g. comparing RAPTOR enabled vs disabled, or hybrid vs vanilla):

```bash
python -m eval.report eval/runs/<hash_a> eval/runs/<hash_b>
```

Outputs a formatted Markdown diff table displaying:
- Retrieval metrics (`recall@1`, `recall@3`, `recall@5`, `recall@10`, `mrr`).
- Hop-sliced retrieval (`recall@5_single_hop`, `recall@5_multi_hop`).
- Generation quality (`faithfulness`, `correctness`, `abstention_rate`, `hallucination_rate`).
- Performance & cost (`mean_llm_calls`, `p50_latency`, `p95_latency`).
- Statistical significance flag (`*`) for deltas exceeding the estimated run-to-run noise spread.

---

## 6. Hard Rules Enforced

1. **Pipeline Immutability**: The harness observes APEX without altering retrieval or generation algorithms.
2. **Deterministic Sampling**: `temperature: 0` is strictly enforced across generation and judge calls.
3. **Independent Judge Model**: Generator (`gemma3:1b`) and Judge (`llama3.2:1b`) are strictly different to eliminate self-preference bias.
4. **Reproducibility**: All runs are stored in directories named after the SHA-256 hash (first 12 characters) of the canonical sorted-key JSON representation of their configuration YAML.
