"""
eval/harness.py — Main Test Harness Runner for APEX Evaluation Suite.
Performs config hashing, dataset execution, judge evaluation, caching, and result persistence.
"""

import os
import sys
import time
import json
import yaml
import hashlib
import argparse
from typing import Dict, Any, List, Optional

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)

from eval.cache import ResponseCache
from eval.metrics import compute_all_metrics
from eval.judge import LLMJudge


def compute_config_hash(config_dict: Dict[str, Any]) -> str:
    """The config hash is the first 12 characters of SHA-256 over the canonical JSON form (sorted keys)."""
    canonical = json.dumps(config_dict, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:12]


def validate_config(config_dict: Dict[str, Any]) -> None:
    """Enforce hard rules 4 and 5 from HARNESS.md."""
    gen_cfg = config_dict.get("generation", {})
    judge_cfg = config_dict.get("judge", {})

    gen_model = gen_cfg.get("model")
    judge_model = judge_cfg.get("model")
    gen_temp = float(gen_cfg.get("temperature", 0.0))
    judge_temp = float(judge_cfg.get("temperature", 0.0))

    # Hard Rule 4: Temperature must be 0 for all generation and judge calls
    if gen_temp != 0.0:
        raise ValueError(f"Hard Rule 4 Violation: Generation temperature must be 0, got {gen_temp}")
    if judge_temp != 0.0:
        raise ValueError(f"Hard Rule 4 Violation: Judge temperature must be 0, got {judge_temp}")

    # Hard Rule 5: The judge model must differ from the generator model. If they are the same, stop and report it.
    if gen_model == judge_model:
        raise ValueError(
            f"Hard Rule 5 Violation: Judge model ('{judge_model}') cannot be the same as generator model ('{gen_model}'). STOPPING."
        )


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load JSONL evaluation dataset."""
    cases = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def build_pipeline(config_dict: Dict[str, Any], stub: Optional[Any] = None) -> Any:
    """Instantiate the APEX pipeline or use a provided stub."""
    if stub is not None:
        return stub

    # Add project root to sys.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from face import RAGInterface
    from modules.config import config as apex_config

    retrieval_cfg = config_dict.get("retrieval", {})
    indexing_cfg = config_dict.get("indexing", {})
    gen_cfg = config_dict.get("generation", {})

    raptor = bool(indexing_cfg.get("raptor", True))
    top_k = int(retrieval_cfg.get("top_k", 5))
    gen_model = gen_cfg.get("model", "gemma3:1b")
    gen_temp = float(gen_cfg.get("temperature", 0.0))
    retrieval_mode = retrieval_cfg.get("mode", "hybrid")

    # Apply configuration parameters to APEX
    apex_config.retrieval.enable_collapsed_retrieval = raptor
    apex_config.retrieval.include_summaries = raptor
    apex_config.retrieval.include_conversation_memory = False

    pipeline = RAGInterface(
        ollama_model=gen_model,
        temperature=gen_temp,
        max_context_chunks=top_k,
        include_summaries=raptor,
        verbose=False
    )

    if retrieval_mode == "bm25":
        pipeline.dense_weight = 0.0
        pipeline.bm25_weight = 0.80
    elif retrieval_mode == "vector":
        pipeline.dense_weight = 0.80
        pipeline.bm25_weight = 0.0
    else:
        pipeline.dense_weight = float(retrieval_cfg.get("vector_weight", 0.50))
        pipeline.bm25_weight = float(retrieval_cfg.get("bm25_weight", 0.30))

    return pipeline


def run_evaluation(
    config_dict: Dict[str, Any],
    dataset_path: str = "eval/dataset/gold.jsonl",
    runs_dir: str = "eval/runs",
    cache_enabled: bool = True,
    pipeline_stub: Optional[Any] = None,
    judge_stub: Optional[Any] = None
) -> Dict[str, Any]:
    """Execute evaluation harness for a specific configuration."""
    validate_config(config_dict)

    cfg_hash = compute_config_hash(config_dict)
    run_dir = os.path.join(runs_dir, cfg_hash)
    os.makedirs(run_dir, exist_ok=True)

    cases_file = os.path.join(run_dir, "cases.jsonl")
    summary_file = os.path.join(run_dir, "summary.json")

    cases = load_dataset(dataset_path)
    pipeline = build_pipeline(config_dict, stub=pipeline_stub)

    cache = ResponseCache(cache_dir="eval/cache", enabled=cache_enabled)
    judge_cfg = config_dict.get("judge", {})
    judge = judge_stub or LLMJudge(
        model=judge_cfg.get("model", "llama3.2:1b"),
        temperature=float(judge_cfg.get("temperature", 0.0)),
        cache=cache
    )

    # 1. Pipeline execution phase
    print(f"\n--- Phase 1/2: Generating pipeline responses ({len(cases)} cases) ---")
    pipeline_results = []
    for idx, case in enumerate(cases, 1):
        q = case["question"]
        t0 = time.perf_counter()
        try:
            res = pipeline.answer(q)
            ans = res.get("answer", "")
            retrieved_ids = res.get("retrieved_ids", [])
            retrieved_chunks = res.get("retrieved_chunks", getattr(pipeline, "_last_retrieved_chunks", []))
            llm_calls = res.get("llm_calls", 1)
        except Exception as e:
            ans = f"PIPELINE_ERROR: {str(e)}"
            retrieved_ids = []
            retrieved_chunks = []
            llm_calls = 0
        latency = time.perf_counter() - t0

        pipeline_results.append({
            "case": case,
            "answer": ans,
            "retrieved_ids": retrieved_ids,
            "retrieved_chunks": retrieved_chunks,
            "llm_calls": llm_calls,
            "latency": latency
        })
        print(f"[{idx:02d}/{len(cases):02d}] ⚡ Answered {case.get('id')} in {latency:.2f}s (calls: {llm_calls})", flush=True)

    # 2. Judge evaluation phase (avoids generator <-> judge model thrashing in local Ollama)
    print(f"\n--- Phase 2/2: LLM-as-a-Judge Evaluation ({len(cases)} cases) ---", flush=True)
    evaluated_cases = []
    with open(cases_file, "w", encoding="utf-8") as cases_out:
        for idx, item in enumerate(pipeline_results, 1):
            case = item["case"]
            q = case["question"]
            ans = item["answer"]
            retrieved_chunks = item["retrieved_chunks"]
            gold_ans = case.get("gold_answer")
            answerable = case.get("answerable", True)

            judge_res = judge.judge_case(
                question=q,
                answer=ans,
                retrieved_chunks=retrieved_chunks,
                gold_answer=gold_ans,
                answerable=answerable
            )

            case_record = {
                "id": case.get("id", f"case_{idx}"),
                "question": q,
                "answer": ans,
                "gold_answer": gold_ans,
                "retrieved_ids": item["retrieved_ids"],
                "gold_chunk_ids": case.get("gold_chunk_ids", []),
                "hops": case.get("hops", 1),
                "answerable": answerable,
                "latency": item["latency"],
                "llm_calls": item["llm_calls"],
                "faithfulness": judge_res.get("faithfulness"),
                "correctness": judge_res.get("correctness"),
                "abstained": judge_res.get("abstained"),
                "reasoning": judge_res.get("reasoning"),
                "judge_error": judge_res.get("judge_error", False)
            }

            cases_out.write(json.dumps(case_record, ensure_ascii=False) + "\n")
            cases_out.flush()
            evaluated_cases.append(case_record)

            status_sym = "✅" if (case_record.get("correctness") or (not answerable and case_record.get("abstained"))) else "❌"
            print(f"[{idx:02d}/{len(cases):02d}] {status_sym} Judged {case.get('id')}: corr={case_record.get('correctness')}, faith={case_record.get('faithfulness')}", flush=True)

    # 3. Compute aggregate metrics
    aggregates = compute_all_metrics(evaluated_cases)
    summary_payload = {
        "config_hash": cfg_hash,
        "config": config_dict,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_path": dataset_path,
        "summary": aggregates
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"\n✨ Completed run {cfg_hash} successfully.")
    print(f"   Summary written to {summary_file}")
    return summary_payload


def main():
    parser = argparse.ArgumentParser(description="APEX Evaluation Harness Runner")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--dataset", default="eval/dataset/gold.jsonl", help="Path to gold dataset JSONL")
    parser.add_argument("--runs-dir", default="eval/runs", help="Directory for run outputs")
    parser.add_argument("--no-cache", action="store_true", help="Bypass LLM response disk cache")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    run_evaluation(
        config_dict=config_dict,
        dataset_path=args.dataset,
        runs_dir=args.runs_dir,
        cache_enabled=not args.no_cache
    )


if __name__ == "__main__":
    main()
