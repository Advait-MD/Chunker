"""Citation parsing and resolution. `[n]`, `[n, m]` and `[n][m]` forms are supported."""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

_CITE_RE = re.compile(r"\[(\s*\d+(?:\s*[,;]\s*\d+)*\s*)\]")
_LEAD_RE = re.compile(r"^((?:\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\]\s*)+)(.*)$", re.S)
_SENT_RE =re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def parse_citation_ids(text: str) -> List[int]:
    """All citation numbers in order of first appearance, deduplicated."""
    seen: List[int] = []
    for m in _CITE_RE.finditer(text or ""):
        for part in re.split(r"[,;]", m.group(1)):
            n = int(part.strip())
            if n not in seen:
                seen.append(n)
    return seen


def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def sentence_citations(text: str) -> List[Tuple[str, List[int]]]:
    """Pair every sentence with the citation ids it carries.

    A citation placed right after a sentence's final period (e.g. "X is Y. [2]")
    is attached to that preceding sentence.
    """
    pairs: List[Tuple[str, List[int]]] = []
    for sent in split_sentences(text):
        lead = _LEAD_RE.match(sent)
        if lead and pairs:  # leading citations belong to the previous sentence
            prev_s, prev_ids = pairs[-1]
            lead_ids = parse_citation_ids(lead.group(1))
            pairs[-1] = (prev_s + " " + lead.group(1).strip(), prev_ids + [i for i in lead_ids if i not in prev_ids])
            sent = lead.group(2).strip()
            if not sent:
                continue
        pairs.append((sent, parse_citation_ids(sent)))
    return pairs


def check_citations(answer: str, declared: List[int], n_evidence: int) -> Dict[str, object]:
    """Validate citations against the evidence set [1..n_evidence]."""
    inline = parse_citation_ids(answer)
    all_ids: Set[int] = set(inline) | {int(x) for x in (declared or []) if isinstance(x, (int, float, str)) and str(x).strip().lstrip("-").isdigit()}
    valid = sorted(i for i in all_ids if 1 <= i <= n_evidence)
    invalid = sorted(i for i in all_ids if not 1 <= i <= n_evidence)
    uncited = [s for s, ids in sentence_citations(answer) if not any(1 <= i <= n_evidence for i in ids)]
    return {"inline": inline, "valid": valid, "invalid": invalid,
            "declared_not_inline": sorted(set(valid) - set(inline)), "uncited_sentences": uncited}
