"""APEX CLI.

  python -m apex.api.cli ingest <folder-or-file> [--force]
  python -m apex.api.cli query "question" [--source FILE] [--json]
  python -m apex.api.cli stats
All commands accept --config configs/<overlay>.yaml (or env APEX_CONFIG).
"""

from __future__ import annotations

import argparse
import json
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _check_models(comps, needed):
    have = comps.client.installed_models()
    missing = [m for m in needed if m not in have and f"{m}:latest" not in have]
    if missing:
        sys.exit("Missing Ollama models: " + ", ".join(missing) + "\nRun: " + " && ".join(f"ollama pull {m}" for m in missing))


def cmd_ingest(args):
    from apex.config import load_config
    from apex.ingest.pipeline import ingest_path
    from apex.pipeline import Components

    comps = Components(load_config(args.config))
    _check_models(comps, [comps.cfg["models"]["embedder"]])
    t0 = time.perf_counter()
    report = {"added_nodes": 0, "skipped_files": 0}
    for path in args.path:
        r = ingest_path(path, comps.cfg, comps.store, comps.embedder, force=args.force)
        report["build_hash"] = r["build_hash"]
        report["added_nodes"] += r["added_nodes"]
        report["skipped_files"] += r["skipped_files"]
    report["seconds"] = round(time.perf_counter() - t0, 1)
    report["total_nodes"] = comps.store.count()
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))


def cmd_query(args):
    from apex.pipeline import build_pipeline

    pipe = build_pipeline(args.config)
    _check_models(pipe.c, [pipe.c.cfg["models"]["embedder"], pipe.c.cfg["models"]["generator"]])
    res = pipe.answer(args.question, source=args.source)
    res.pop("_trace", None)
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return
    print(f"\n{res['answer']}\n")
    for c in res["citations"]:
        loc = f"p. {c['page']}" if c["page"] is not None else ""
        span = f" chars {c['char_span'][0]}–{c['char_span'][1]}" if c.get("char_span") else ""
        print(f"  [{c['cite_id']}] {c['source']} {loc}{span}  ({c['node_id']})")
    t = res["totals"]
    print(f"\nabstained={res['abstained']} confidence={res['confidence']} route={res['route']}")
    print(f"llm_calls={t['llm_calls']} tokens={t['prompt_tokens']}+{t['completion_tokens']} total={t['total_ms']/1000:.1f}s")
    if res["uncited_sentences"]:
        print(f"uncited sentences: {len(res['uncited_sentences'])}")
    print(f"trace: {res['trace_path']}")


def cmd_stats(args):
    from apex.config import load_config
    from apex.pipeline import Components

    comps = Components(load_config(args.config))
    print(json.dumps({"store": str(comps.store.path), "nodes": comps.store.count(),
                      "sources": comps.store.sources(), "config_hash": comps.cfg["_meta"]["config_hash"]}, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(prog="apex")
    p.add_argument("--config", default=None, help="overlay YAML on configs/base.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("ingest"); i.add_argument("path", nargs="+", help="files or folders"); i.add_argument("--force", action="store_true")
    i.set_defaults(fn=cmd_ingest)
    q = sub.add_parser("query"); q.add_argument("question"); q.add_argument("--source")
    q.add_argument("--json", action="store_true"); q.set_defaults(fn=cmd_query)
    s = sub.add_parser("stats"); s.set_defaults(fn=cmd_stats)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
