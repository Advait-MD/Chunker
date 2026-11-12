"""Command line for the eval harness (what `make eval` calls).

  python -m apex.eval.suite splits  <datasets...>        sample fixed dev/test ids (once)
  python -m apex.eval.suite index   <datasets...>        write corpus files and ingest them
  python -m apex.eval.suite run     [--config C] [--datasets a,b] [--split test] [--run-id R] [--limit N] [--no-judge]
  python -m apex.eval.suite judge   <run_dir> [--datasets a,b]
  python -m apex.eval.suite compare <run_dir_A> <run_dir_baseline>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from apex.config import load_config  # noqa: E402
from apex.eval import datasets as D  # noqa: E402

DEFAULT = ["qasper", "musique", "quality", "hotpotqa", "2wiki", "custom"]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="apex.eval.suite")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("splits"); s.add_argument("datasets", nargs="+"); s.add_argument("--overwrite", action="store_true")
    s = sub.add_parser("index"); s.add_argument("datasets", nargs="+"); s.add_argument("--config")
    s = sub.add_parser("run")
    s.add_argument("--config"); s.add_argument("--datasets", default=",".join(DEFAULT))
    s.add_argument("--split", default="test", choices=["dev", "test"]); s.add_argument("--run-id")
    s.add_argument("--limit", type=int); s.add_argument("--no-judge", action="store_true")
    s = sub.add_parser("judge"); s.add_argument("run_dir"); s.add_argument("--datasets")
    s = sub.add_parser("compare"); s.add_argument("run_a"); s.add_argument("run_b"); s.add_argument("--out")
    a = ap.parse_args(argv)

    if a.cmd == "splits":
        cfg = load_config()
        for n in a.datasets:
            print(D.make_splits(n, cfg, overwrite=a.overwrite))
    elif a.cmd == "index":
        from apex.ingest.pipeline import ingest_path
        from apex.pipeline import Components
        for n in a.datasets:
            cfg = load_config(a.config, overrides=D.bench_overrides(n))
            print(f"[{n}] corpus:", D.build_corpus(n, cfg))
            comps, t = Components(cfg), time.time()
            rep = ingest_path(D.corpus_dir(n, cfg), cfg, comps.store, comps.embedder, log=lambda *_: None)
            print(f"[{n}] +{rep['added_nodes']} nodes, {rep['skipped_files']} files cached, "
                  f"{comps.store.count()} in '{cfg['store']['collection']}' ({time.time() - t:.0f}s)")
    elif a.cmd == "run":
        from apex.eval.run import run_eval
        d = run_eval(a.config, [x for x in a.datasets.split(",") if x], split=a.split, run_id=a.run_id,
                     limit=a.limit, judge=not a.no_judge)
        print(f"\nwrote {d / 'table.md'}")
    elif a.cmd == "judge":
        from apex.eval.run import judge_run
        judge_run(Path(a.run_dir), a.datasets.split(",") if a.datasets else None)
    elif a.cmd == "compare":
        from apex.eval.report import compare
        text = compare(Path(a.run_a), Path(a.run_b))
        if a.out:
            Path(a.out).write_text(text, encoding="utf-8")
        print(text)


if __name__ == "__main__":
    main()
