"""
eval/tests/test_harness_smoke.py — End-to-end smoke test for APEX Evaluation Harness.
Runs on a 3-case dataset using stub pipeline and judge components per HARNESS.md Section 11.
"""

import os
import json
import pytest
from eval.harness import run_evaluation, compute_config_hash


class StubPipeline:
    """Mock pipeline returning fixed deterministic responses."""
    def answer(self, question: str) -> dict:
        return {
            "answer": "Robin Wieruch wrote The Road to React.",
            "retrieved_ids": ["chunk_01", "chunk_02"],
            "llm_calls": 1
        }


class StubJudge:
    """Mock judge returning deterministic validation scores."""
    def judge_case(self, question, answer, retrieved_chunks, gold_answer, answerable):
        return {
            "faithfulness": True,
            "correctness": True,
            "abstained": not answerable,
            "reasoning": "Verified by stub judge",
            "judge_error": False
        }


def test_harness_smoke_end_to_end(tmp_path):
    # 1. Create a 3-case test dataset
    dataset_file = tmp_path / "smoke_dataset.jsonl"
    cases = [
        {
            "id": "smoke-sh-01",
            "question": "Who wrote The Road to React?",
            "gold_answer": "Robin Wieruch",
            "gold_chunk_ids": ["chunk_01"],
            "hops": 1,
            "answerable": True
        },
        {
            "id": "smoke-mh-02",
            "question": "What is the relation between Void and Clean Code?",
            "gold_answer": "Clarity without obstruction",
            "gold_chunk_ids": ["chunk_01", "chunk_02"],
            "hops": 2,
            "answerable": True
        },
        {
            "id": "smoke-un-03",
            "question": "What hooks did Steve Jobs use in 1984?",
            "gold_answer": None,
            "gold_chunk_ids": [],
            "hops": 1,
            "answerable": False
        }
    ]

    with open(dataset_file, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")

    # 2. Define valid config
    config = {
        "name": "smoke_test_config",
        "retrieval": {
            "mode": "hybrid",
            "bm25_weight": 0.5,
            "vector_weight": 0.5,
            "top_k": 5,
            "ivf_nprobe": 8
        },
        "indexing": {
            "raptor": True,
            "chunk_size": 512,
            "chunk_overlap": 64,
            "embedding_model": "nomic-embed-text",
            "chroma_version": "1.5.5",
            "seed": 42
        },
        "generation": {
            "model": "gemma3:1b",
            "temperature": 0
        },
        "judge": {
            "model": "llama3.2:1b",
            "temperature": 0
        }
    }

    runs_dir = tmp_path / "runs"
    cfg_hash = compute_config_hash(config)

    # 3. Execute evaluation harness with stubs
    result = run_evaluation(
        config_dict=config,
        dataset_path=str(dataset_file),
        runs_dir=str(runs_dir),
        cache_enabled=False,
        pipeline_stub=StubPipeline(),
        judge_stub=StubJudge()
    )

    # 4. Assert artifacts exist and are well-formed
    run_folder = runs_dir / cfg_hash
    assert run_folder.exists()

    cases_file = run_folder / "cases.jsonl"
    assert cases_file.exists()

    summary_file = run_folder / "summary.json"
    assert summary_file.exists()

    # Read cases.jsonl
    with open(cases_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 3
    assert lines[0]["id"] == "smoke-sh-01"
    assert lines[0]["retrieved_ids"] == ["chunk_01", "chunk_02"]
    assert lines[2]["answerable"] is False
    assert lines[2]["abstained"] is True

    # Read summary.json
    with open(summary_file, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    assert summary_data["config_hash"] == cfg_hash
    metrics = summary_data["summary"]
    assert metrics["total_cases"] == 3
    assert metrics["answerable_cases"] == 2
    assert metrics["unanswerable_cases"] == 1
    assert metrics["recall@1"] > 0
    assert metrics["faithfulness"] == 1.0
    assert metrics["abstention_rate"] == 1.0
    assert metrics["hallucination_rate"] == 0.0
