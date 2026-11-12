"""Benchmark loaders, fixed dev/test splits, corpus builders and per-dataset indexes.

Every dataset is normalised to `EvalItem`s plus a corpus of pre-segmented documents
(`.jsonl`, one unit per line) that goes through the *same* ingest path as user files.

Retrieval scope:
  * QASPER, QuALITY: questions are about one document; retrieval is filtered to it (`scope`).
  * MuSiQue, HotpotQA, 2Wiki: the paragraphs of all sampled questions (dev + test) are pooled into
    one corpus per dataset (open-corpus setting), so distractors come from other questions too.
  * custom: the owner's 4 PDFs, already in the default store.

Unanswerable questions (spec: inject ~10%): QASPER and custom have native ones. For the pooled
multi-hop sets, a seeded ~10% of sampled questions has its supporting paragraphs removed from the
corpus and is relabelled unanswerable (the MuSiQue-full construction). QuALITY is multiple choice
and gets none.

Splits are sampled once with a fixed seed and written to apex/eval/splits/<name>.json (committed),
so runs reproduce from a clean checkout after scripts/fetch_benchmarks.py.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from apex.config import REPO_ROOT, resolve_path

SPLITS_DIR = Path(__file__).resolve().parent / "splits"
DATASETS = ("qasper", "quality", "musique", "hotpotqa", "2wiki", "custom")
POOLED = {"musique", "hotpotqa", "2wiki"}
ABSTAIN_GOLD = "Unanswerable"


@dataclass
class EvalItem:
    qid: str
    dataset: str
    question: str
    answers: List[str]
    answerable: bool = True
    evidence_sets: List[List[str]] = field(default_factory=list)  # alternative gold evidence sets (max over sets)
    scope: Optional[str] = None  # source_path to restrict retrieval to
    options: Optional[List[str]] = None
    gold_option: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def prompt(self) -> str:
        """The question as sent to the system (multiple-choice options appended for QuALITY)."""
        if not self.options:
            return self.question
        opts = "\n".join(f"({'ABCD'[i]}) {o}" for i, o in enumerate(self.options))
        return (f"{self.question}\n\nOptions:\n{opts}\n\n"
                "Start your answer with the letter of the correct option, e.g. \"(B) ...\".")


Unit = Dict[str, Any]  # {"section": str|None, "text": str}


def _key(title: str, text: str) -> str:
    return hashlib.sha1(f"{title}\x00{text}".encode("utf-8")).hexdigest()[:16]


def _para(title: str, text: str) -> str:
    return f"{title}\n{text}".strip()


def _data_dir(cfg) -> Path:
    return resolve_path(cfg["eval"]["data_dir"])


def _need(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run: python scripts/fetch_benchmarks.py")
    return path


# =============================================================================
# Raw loaders → (items grouped by source split, documents {name: [units]}, paragraph keys per item)
# =============================================================================
def _load_qasper(d: Path):
    import pandas as pd

    items: Dict[str, List[EvalItem]] = {}
    docs: Dict[str, List[Unit]] = {}
    for src, fname in (("validation", "qasper__validation__0000.parquet"), ("test", "qasper__test__0000.parquet")):
        df = pd.read_parquet(_need(d / "qasper" / fname))
        out = items.setdefault(src, [])
        for _, row in df.iterrows():
            name = f"qasper_{row['id']}.jsonl"
            units: List[Unit] = [{"section": "Title and abstract", "text": f"{row['title']}\n\n{row['abstract']}"}]
            ft = row["full_text"]
            for sec, paras in zip(ft["section_name"], ft["paragraphs"]):
                body = "\n\n".join(p for p in paras if p and p.strip())
                if body:
                    units.append({"section": sec or None, "text": body})
            caps = [c for c in row["figures_and_tables"]["caption"] if c and c.strip()]
            if caps:
                units.append({"section": "Figures and tables", "text": "\n\n".join(caps)})
            docs[name] = units
            qas = row["qas"]
            for q, qid, ans in zip(qas["question"], qas["question_id"], qas["answers"]):
                golds, ev_sets, n_unans = [], [], 0
                for a in ans["answer"]:
                    if a["unanswerable"]:
                        golds.append(ABSTAIN_GOLD)
                        n_unans += 1
                        continue
                    if a["yes_no"] is not None:
                        golds.append("Yes" if a["yes_no"] else "No")
                    elif a["extractive_spans"] is not None and len(a["extractive_spans"]):
                        golds.append(", ".join(s.strip() for s in a["extractive_spans"]))
                    elif a["free_form_answer"]:
                        golds.append(a["free_form_answer"])
                    ev = [e.replace("FLOAT SELECTED:", "").strip() for e in (a["evidence"] if a["evidence"] is not None else [])]
                    ev = [e for e in ev if e]
                    if ev:
                        ev_sets.append(ev)
                if not golds:
                    continue
                out.append(EvalItem(qid=f"qasper:{qid}", dataset="qasper", question=q.strip(), answers=golds,
                                    answerable=n_unans < len(ans["answer"]), evidence_sets=ev_sets, scope=name,
                                    meta={"paper_id": row["id"], "n_annotations": len(ans["answer"]),
                                          "n_unanswerable_annotations": n_unans}))
    return items, docs, {}


def _load_quality(d: Path):
    import pandas as pd

    df = pd.read_parquet(_need(d / "quality" / "data__validation-00000-of-00001-c057a0698ad4d3c4.parquet"))
    items, docs = [], {}
    for _, row in df.iterrows():
        name = f"quality_{row['article_id']}.jsonl"
        docs.setdefault(name, [{"section": row["title"], "text": row["article"]}])
        opts = [str(o) for o in row["options"]]
        gold = int(row["gold_label"]) - 1  # QuALITY labels are 1-based
        items.append(EvalItem(qid=f"quality:{row['question_unique_id']}", dataset="quality",
                              question=row["question"].strip(), answers=[opts[gold]], options=opts, gold_option=gold,
                              scope=name, meta={"article_id": int(row["article_id"]), "hard": bool(row["difficult"])}))
    # one question can appear once per writer set; keep the first occurrence of each id
    seen, uniq = set(), []
    for it in items:
        if it.qid not in seen:
            seen.add(it.qid)
            uniq.append(it)
    return {"validation": uniq}, docs, {}


def _pooled(rows: Iterable[Tuple[str, str, List[str], List[Tuple[str, str]], List[int], Dict[str, Any]]], dataset: str):
    """rows: (qid, question, answers, paragraphs [(title, text)], supporting indices, meta)."""
    items, paras, support = [], {}, {}
    for qid, q, answers, plist, sup_idx, meta in rows:
        keys = []
        for title, text in plist:
            k = _key(title, text)
            paras.setdefault(k, {"section": title, "text": _para(title, text)})
            keys.append(k)
        sup = [keys[i] for i in sup_idx]
        support[qid] = {"all": keys, "support": sup}
        items.append(EvalItem(qid=qid, dataset=dataset, question=q.strip(), answers=answers,
                              evidence_sets=[[paras[k]["text"] for k in sup]], meta=meta))
    return {"validation": items}, {"__paragraphs__": paras}, support


def _load_musique(d: Path):
    rows = []
    for line in open(_need(d / "musique" / "musique_ans_v1.0_dev.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        plist = [(p["title"], p["paragraph_text"]) for p in r["paragraphs"]]
        sup = [i for i, p in enumerate(r["paragraphs"]) if p["is_supporting"]]
        rows.append((f"musique:{r['id']}", r["question"], [r["answer"]] + list(r.get("answer_aliases") or []),
                     plist, sup, {"hops": int(r["id"].split("hop")[0]), "type": r["id"].split("__")[0]}))
    return _pooled(rows, "musique")


def _load_hotpot_like(path: Path, dataset: str, joiner: str):
    import pandas as pd

    df = pd.read_parquet(_need(path))
    rows = []
    for _, r in df.iterrows():
        ctx = r["context"]
        plist = [(t, joiner.join(s for s in sents).strip()) for t, sents in zip(ctx["title"], ctx["sentences"])]
        sup_titles = list(dict.fromkeys(r["supporting_facts"]["title"]))
        sup = [i for i, (t, _) in enumerate(plist) if t in sup_titles]
        if len(sup) < len(sup_titles):
            continue  # a supporting title missing from the context: unusable gold, skip
        meta = {"type": r["type"], "hops": 2}
        if "level" in r:
            meta["level"] = r["level"]
        rows.append((f"{dataset}:{r['id']}", r["question"], [r["answer"]], plist, sup, meta))
    return _pooled(rows, dataset)


def _load_custom(d: Path):
    """The owner's gold set over the 4 repo PDFs. Evidence text is frozen in splits/custom.json."""
    frozen = json.loads((SPLITS_DIR / "custom.json").read_text(encoding="utf-8"))
    items = [EvalItem(**it) for it in frozen["items"]]
    return {"test": items}, {}, {}


LOADERS = {
    "qasper": _load_qasper,
    "quality": _load_quality,
    "musique": _load_musique,
    "hotpotqa": lambda d: _load_hotpot_like(d / "hotpotqa" / "distractor__validation-00000-of-00001.parquet", "hotpotqa", ""),
    "2wiki": lambda d: _load_hotpot_like(d / "2wiki" / "data__validation-00000-of-00001.parquet", "2wiki", " "),
    "custom": _load_custom,
}


# =============================================================================
# Splits
# =============================================================================
def make_splits(name: str, cfg: Dict[str, Any], overwrite: bool = False) -> Path:
    """Sample fixed dev/test question ids (and injected unanswerables) once; write splits/<name>.json."""
    out = SPLITS_DIR / f"{name}.json"
    if name == "custom":
        raise ValueError("custom splits are frozen by scripts/freeze_custom_gold.py")
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; splits are fixed. Pass overwrite=True only to resample deliberately.")
    e = cfg["eval"]
    n_dev, n_test, seed = int(e["n_dev"]), int(e["n_test"]), int(cfg["run"]["seed"])
    groups, _, support = LOADERS[name](_data_dir(cfg))
    rng = random.Random(seed)

    if name == "qasper":  # official validation → dev, official test → test
        dev = rng.sample([it.qid for it in groups["validation"]], n_dev)
        test = rng.sample([it.qid for it in groups["test"]], n_test)
    elif name == "quality":  # split by article so tuning on dev never sees a test article
        pool = groups["validation"]
        arts = sorted({it.meta["article_id"] for it in pool})
        rng.shuffle(arts)
        by_art: Dict[int, List[str]] = {}
        for it in pool:
            by_art.setdefault(it.meta["article_id"], []).append(it.qid)
        dev_arts, dev_q = [], []
        for a in arts:
            if len(dev_q) >= n_dev:
                break
            dev_arts.append(a)
            dev_q += by_art[a]
        dev = rng.sample(dev_q, n_dev)
        test = rng.sample([q for a in arts if a not in dev_arts for q in by_art[a]], n_test)
    else:
        ids = [it.qid for it in groups["validation"]]
        picked = rng.sample(ids, n_dev + n_test)
        dev, test = picked[:n_dev], picked[n_dev:]

    injected: List[str] = []
    if name in POOLED:
        frac = float(e["unanswerable_fraction"])
        sampled = dev + test
        for split_ids in (dev, test):
            k = round(frac * len(split_ids))
            cands = list(split_ids)
            rng.shuffle(cands)
            for qid in cands:
                if len([q for q in injected if q in split_ids]) >= k:
                    break
                mine = set(support[qid]["support"])
                others = {s for q in sampled if q != qid and q not in injected for s in support[q]["support"]}
                if mine & others:
                    continue  # its supporting paragraphs also support another sampled question
                injected.append(qid)

    manifest = json.loads((_data_dir(cfg) / "MANIFEST.json").read_text(encoding="utf-8"))
    payload = {"dataset": name, "seed": seed, "n_dev": n_dev, "n_test": n_test,
               "dev": dev, "test": test, "injected_unanswerable": injected,
               "sources": {k: v["sha256"] for k, v in manifest.items() if k.startswith(name + "/")}}
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return out


def read_splits(name: str) -> Dict[str, Any]:
    p = SPLITS_DIR / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(f"No fixed splits for {name}: run `python -m apex.eval.suite splits {name}`")
    return json.loads(p.read_text(encoding="utf-8"))


def load_split(name: str, split: str, cfg: Dict[str, Any]) -> List[EvalItem]:
    sp = read_splits(name)
    groups, _, _ = LOADERS[name](_data_dir(cfg))
    by_id = {it.qid: it for g in groups.values() for it in g}
    missing = [q for q in sp[split] if q not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} {name}/{split} ids not in the raw data (e.g. {missing[:3]}); data changed?")
    injected = set(sp.get("injected_unanswerable", []))
    out = []
    for qid in sp[split]:
        it = by_id[qid]
        if qid in injected:
            it.meta["injected_unanswerable"] = True
            it.meta["original_answers"] = it.answers
            it.answerable, it.answers, it.evidence_sets = False, [ABSTAIN_GOLD], []
        out.append(it)
    return out


# =============================================================================
# Corpus + index
# =============================================================================
def corpus_dir(name: str, cfg: Dict[str, Any]) -> Path:
    return _data_dir(cfg) / name / "corpus"


def collection_name(name: str) -> str:
    return "nodes" if name == "custom" else f"bench_{name}"


def build_corpus(name: str, cfg: Dict[str, Any], shard_size: int = 200) -> Dict[str, Any]:
    """Write the documents needed by the dev+test items as .jsonl files; returns counts."""
    if name == "custom":
        raise ValueError("custom uses the owner's PDFs in the default store; nothing to build")
    sp = read_splits(name)
    groups, docs, support = LOADERS[name](_data_dir(cfg))
    wanted = set(sp["dev"]) | set(sp["test"])
    out = corpus_dir(name, cfg)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jsonl"):
        old.unlink()

    if name in POOLED:
        paras = docs["__paragraphs__"]
        removed = {k for q in sp.get("injected_unanswerable", []) for k in support[q]["support"]}
        keys = sorted({k for q in wanted for k in support[q]["all"]} - removed)
        for i in range(0, len(keys), shard_size):
            lines = [json.dumps(paras[k], ensure_ascii=False) for k in keys[i:i + shard_size]]
            (out / f"{name}_shard_{i // shard_size:04d}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {"documents": -(-len(keys) // shard_size), "units": len(keys), "removed_supporting": len(removed)}

    by_id = {it.qid: it for g in groups.values() for it in g}
    names = sorted({by_id[q].scope for q in wanted})
    n_units = 0
    for nm in names:
        units = docs[nm]
        n_units += len(units)
        (out / nm).write_text("\n".join(json.dumps(u, ensure_ascii=False) for u in units) + "\n", encoding="utf-8")
    return {"documents": len(names), "units": n_units, "removed_supporting": 0}


def bench_overrides(name: str) -> Dict[str, Any]:
    """Config overrides that point the pipeline at a benchmark's own collection and corpus format."""
    if name == "custom":
        return {}
    return {"store": {"collection": collection_name(name)}, "ingest": {"extensions": [".jsonl"]}}


def item_to_dict(it: EvalItem) -> Dict[str, Any]:
    return asdict(it)
