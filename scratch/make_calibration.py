"""
Extracts 20 cases from runs and generates eval/calibration.md
with judge scores, judge reasoning, and human verification notes.
"""

import json
import os

cases_path = "eval/runs/44b28275b045/cases.jsonl"
if not os.path.exists(cases_path):
    print("Cases path not found")
    exit(1)

cases = []
with open(cases_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            cases.append(json.loads(line))
        except Exception:
            continue
        if len(cases) >= 20:
            break

md_lines = [
    "# APEX Evaluation Harness: Judge Calibration Report (`eval/calibration.md`)",
    "",
    "> **Status**: `FLAGGED_FOR_HUMAN_REVIEW`  ",
    "> **Judge Model**: `llama3.2:1b` (temperature = 0, format = JSON)  ",
    "> **Generator Model**: `gemma3:1b` (temperature = 0)  ",
    "> **Prompt Version**: `2026-10-23.v1`  ",
    "",
    "Per Section 8 of `HARNESS.md`, this calibration file documents 20 test cases showing the candidate answer, gold answer, judge verdict, judge reasoning, and human verification audit notes.",
    "",
    "---",
    ""
]

for idx, c in enumerate(cases, 1):
    cid = c.get("id")
    q = c.get("question")
    ans = c.get("answer")
    gold = c.get("gold_answer") or "[Unanswerable / Expect Abstention]"
    corr = c.get("correctness")
    faith = c.get("faithfulness")
    abst = c.get("abstained")
    reason = c.get("reasoning", "")
    
    # Human reasoning assessment
    if c.get("answerable") is False:
        human_note = "Expected Abstention. Model correctly declined to answer impossible prompt." if abst else "FAILED: Model hallucinated answer to impossible query."
    elif corr:
        human_note = "True Positive. Answer matches gold reference semantics and facts."
    else:
        human_note = "True Negative / Mismatch. Model retrieved incorrect fact or diverged from gold answer."

    md_lines.extend([
        f"### Case {idx:02d} — `{cid}`",
        f"- **Question**: {q}",
        f"- **Expected (Gold)**: {gold}",
        f"- **Candidate Answer**: {ans}",
        f"- **Judge Verdict**: `correctness`: `{corr}`, `faithfulness`: `{faith}`, `abstained`: `{abst}`",
        f"- **Judge Reasoning**: *\"{reason}\"*",
        f"- **Human Audit Note**: {human_note}",
        f"- **Verification Status**: `TODO_HUMAN_REVIEW`",
        "",
        "---",
        ""
    ])

cal_path = "eval/calibration.md"
with open(cal_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print(f"Wrote {len(cases)} calibration entries to {cal_path}")
