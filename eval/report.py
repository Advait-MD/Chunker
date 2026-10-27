"""
eval/report.py — Comparative Run Reporter for APEX Evaluation Harness.
Parses summary.json files across runs, outputs markdown diff tables, and flags significant deltas.
Usage:
    python -m eval.report eval/runs/<hash_a> eval/runs/<hash_b> [--noise-file eval/runs/noise.json]
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Optional

DEFAULT_NOISE_THRESHOLDS = {
    "recall@1": 0.03,
    "recall@3": 0.03,
    "recall@5": 0.03,
    "recall@10": 0.03,
    "mrr": 0.03,
    "recall@5_single_hop": 0.05,
    "recall@5_multi_hop": 0.05,
    "faithfulness": 0.05,
    "correctness": 0.05,
    "abstention_rate": 0.05,
    "hallucination_rate": 0.05,
    "mean_llm_calls": 0.10,
    "p50_latency": 0.25,
    "p95_latency": 0.50,
}


def load_summary(run_path: str) -> Dict[str, Any]:
    """Load summary.json from a run directory or direct file path."""
    if os.path.isdir(run_path):
        summary_file = os.path.join(run_path, "summary.json")
    else:
        summary_file = run_path

    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"Could not find summary file at: {summary_file}")

    with open(summary_file, "r", encoding="utf-8") as f:
        return json.load(f)


def format_value(key: str, val: Any) -> str:
    """Format metric value as string percentage or decimal."""
    if val is None:
        return "N/A"
    if not isinstance(val, (int, float)):
        return str(val)

    if "latency" in key:
        return f"{val:.3f}s"
    elif "calls" in key:
        return f"{val:.2f}"
    elif any(k in key for k in ["recall", "mrr", "faithfulness", "correctness", "abstention", "hallucination"]):
        return f"{val * 100:.1f}%"
    return f"{val:.3f}"


def generate_diff_table(
    summary_a: Dict[str, Any],
    summary_b: Dict[str, Any],
    noise_thresholds: Optional[Dict[str, float]] = None
) -> str:
    """Generate Markdown diff comparison table."""
    thresholds = dict(DEFAULT_NOISE_THRESHOLDS)
    if noise_thresholds:
        thresholds.update(noise_thresholds)

    cfg_a = summary_a.get("config", {})
    cfg_b = summary_b.get("config", {})
    name_a = cfg_a.get("name", summary_a.get("config_hash", "Run A"))
    name_b = cfg_b.get("name", summary_b.get("config_hash", "Run B"))
    hash_a = summary_a.get("config_hash", "hash_a")[:12]
    hash_b = summary_b.get("config_hash", "hash_b")[:12]

    metrics_a = summary_a.get("summary", {})
    metrics_b = summary_b.get("summary", {})

    all_keys = [
        "recall@1", "recall@3", "recall@5", "recall@10", "mrr",
        "recall@5_single_hop", "recall@5_multi_hop",
        "faithfulness", "correctness", "abstention_rate", "hallucination_rate",
        "mean_llm_calls", "p50_latency", "p95_latency", "judge_errors"
    ]

    lines = []
    lines.append(f"### APEX Benchmark Comparison: `{name_a}` ({hash_a}) vs. `{name_b}` ({hash_b})\n")
    lines.append("| Metric | " + f"{name_a} (`{hash_a}`)" + " | " + f"{name_b} (`{hash_b}`)" + " | Delta | Sig |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")

    for key in all_keys:
        val_a = metrics_a.get(key)
        val_b = metrics_b.get(key)

        if val_a is None and val_b is None:
            continue

        str_a = format_value(key, val_a)
        str_b = format_value(key, val_b)

        if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
            delta = val_b - val_a
            noise_thresh = thresholds.get(key, 0.05)
            is_sig = abs(delta) > noise_thresh

            sig_marker = "*" if is_sig else ""
            if "latency" in key:
                delta_str = f"{delta:+.3f}s"
            elif "calls" in key or "errors" in key:
                delta_str = f"{delta:+.2f}"
            else:
                delta_str = f"{delta * 100:+.1f}%"
        else:
            delta_str = "—"
            sig_marker = ""

        lines.append(f"| `{key}` | {str_a} | {str_b} | {delta_str} | {sig_marker} |")

    lines.append("\n`*` indicates delta exceeds estimated run-to-run noise spread.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="APEX Evaluation Harness Report Comparator")
    parser.add_argument("run_a", help="Path to first run directory or summary.json")
    parser.add_argument("run_b", help="Path to second run directory or summary.json")
    parser.add_argument("--noise-file", default=None, help="JSON file containing empirically measured noise spreads")
    args = parser.parse_args()

    sum_a = load_summary(args.run_a)
    sum_b = load_summary(args.run_b)

    noise_thresholds = None
    if args.noise_file and os.path.exists(args.noise_file):
        with open(args.noise_file, "r", encoding="utf-8") as f:
            noise_thresholds = json.load(f)

    table = generate_diff_table(sum_a, sum_b, noise_thresholds)
    print("\n" + table + "\n")


if __name__ == "__main__":
    main()
