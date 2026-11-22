"""Citation verifier (spec §2.7, M5): is every answer sentence cited, and entailed by the passages it cites?

Each sentence of the answer is paired with its citation ids (`sentence_citations`). The NLI hypothesis is
the sentence without its citation markers. The premises are each cited passage, plus all cited passages
joined when there are several (a multi-hop sentence can need two). The sentence is supported when the
best entailment probability reaches `verifier.entail_threshold`. An uncited sentence is unsupported.

`verifier.action`:
  flag        record the verdicts only
  remove      drop unsupported sentences from the answer (never all of them: if none is supported the
              answer is kept and flagged `all_unsupported`)
  regenerate  one regeneration that names the unsupported sentences, then `remove` on the new answer

The same NLI model scores the faithfulness metric (apex.eval.nli_eval), so a verifier gain on that metric
is partly by construction; docs/M5_NOTES.md reports it with that caveat.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from apex.generate.citations import sentence_citations
from apex.generate.generator import EvidenceSet

_MARK = re.compile(r"\s*\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\]")


def strip_citations(sentence: str) -> str:
    return _MARK.sub("", sentence).strip()


def score_sentences(nli, answer: str, texts_by_cite: Dict[int, str], threshold: float) -> List[Dict[str, Any]]:
    """Per sentence: text, cite ids, best entailment, supported. Shared by the verifier and the metric."""
    rows, pairs, owners = [], [], []
    for s, ids in sentence_citations(answer):
        hyp = strip_citations(s)
        ids = [i for i in ids if i in texts_by_cite]
        rows.append({"sentence": s, "cites": ids, "entail": None, "supported": False})
        if not hyp or not ids:
            continue
        prem = [texts_by_cite[i] for i in ids]
        if len(prem) > 1:
            prem.append("\n".join(prem))
        for p in prem:
            pairs.append((p, hyp)); owners.append(len(rows) - 1)
    if pairs:
        for o, pr in zip(owners, nli.probs_many(pairs)):
            e = pr["entailment"]
            if rows[o]["entail"] is None or e > rows[o]["entail"]:
                rows[o]["entail"] = e
    for r in rows:
        r["supported"] = r["entail"] is not None and r["entail"] >= threshold
        if r["entail"] is not None:
            r["entail"] = round(r["entail"], 4)
    return rows


class CitationVerifier:
    def __init__(self, cfg: Dict[str, Any], nli):
        v = cfg["verifier"]
        self.nli = nli
        self.threshold = float(v["entail_threshold"])
        self.action = v["action"]
        if self.action not in ("flag", "remove", "regenerate"):
            raise ValueError(f"verifier.action must be flag, remove or regenerate, got {self.action!r}")

    def verify(self, answer: str, evidence: EvidenceSet) -> Dict[str, Any]:
        rows = score_sentences(self.nli, answer, {it.cite_id: it.text for it in evidence.items}, self.threshold)
        n_sup = sum(r["supported"] for r in rows)
        return {"sentences": rows, "n_sentences": len(rows), "n_supported": n_sup,
                "unsupported": [r["sentence"] for r in rows if not r["supported"]]}

    @staticmethod
    def remove_unsupported(answer: str, report: Dict[str, Any]) -> Optional[str]:
        """The answer with unsupported sentences dropped, or None when none would survive."""
        keep = [r["sentence"] for r in report["sentences"] if r["supported"]]
        if not keep:
            return None
        return " ".join(keep)

    @staticmethod
    def feedback(report: Dict[str, Any]) -> str:
        bad = "\n".join(f"- {s}" for s in report["unsupported"])
        return ("A previous answer contained sentences that are not supported by the passages they cite, or carry "
                f"no citation:\n{bad}\nWrite the answer again. State only facts the passages support, and cite the "
                "passage that states each one.")
