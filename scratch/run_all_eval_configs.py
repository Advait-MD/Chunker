"""
Executes all 5 evaluation configurations sequentially.
Uses caching and line-buffering to ensure smooth, reliable runs.
"""

import os
import sys
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from eval.harness import run_evaluation, compute_config_hash

CONFIG_FILES = [
    "eval/configs/hybrid_raptor_off.yaml",
    "eval/configs/bm25.yaml",
    "eval/configs/vector.yaml",
    "eval/configs/hybrid.yaml",
]

for cfg_path in CONFIG_FILES:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg_hash = compute_config_hash(cfg)
    summary_path = os.path.join("eval/runs", cfg_hash, "summary.json")
    if os.path.exists(summary_path):
        print(f"\n[SKIP] {cfg.get('name')} ({cfg_hash}) already completed.")
        continue

    print(f"\n{'='*70}")
    print(f"RUNNING CONFIG: {cfg.get('name')} [Hash: {cfg_hash}]")
    print(f"{'='*70}")
    run_evaluation(
        config_dict=cfg,
        dataset_path="eval/dataset/gold.jsonl",
        runs_dir="eval/runs",
        cache_enabled=True
    )

print("\n🎉 ALL CONFIGURATIONS COMPLETED!")
