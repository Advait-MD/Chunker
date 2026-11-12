"""Local LLM judge for RAGAS-style and ALCE-style metrics (spec §2.8).

One JSON-schema call per question at temperature 0 through the logged wrapper (caller `judge`),
never bound to the system's trace, so judge cost is excluded from system cost. The judge labels:

  passages[]              relevant: needed to give the reference answer       → context precision (RAGAS, rank-weighted)
  gold_support            the passages state the reference answer             → context recall
  sentences[].supported   some passage supports the answer sentence           → faithfulness
  sentences[].cited[]     each passage the sentence cites supports it alone   → citation precision / recall (ALCE)
  answer_relevance 1–5                                                         → answer relevance

Citations are nested under their sentence (the prompt lists each sentence's cited ids) because a
flat pairs list was observed to be left incomplete by gemma3:4b.

**Quote check.** Every positive support label must come with a short verbatim quote from a passage.
The quote is verified in code; a label whose quote is not found in the relevant passage(s) is
overruled to unsupported and counted in `overruled`, a direct measure of judge reliability. (Small
judges otherwise claim support from outside knowledge, or quote the answer itself; both were
observed with gemma3:4b.) A sentence also counts as faithful when one of its citations is verified.

Missing labels get one repair call that names them; if still missing, `JudgeIncomplete` is raised
and the runner records the row as a judge error (reported, excluded from judge means).

The judge must be from a different model family than the generator (checked at construction).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from apex.eval.metrics import _shingles, normalize_answer
from apex.generate.citations import sentence_citations
from apex.llm import OllamaClient, OllamaError

_QB = {"supported": {"type": "boolean"}, "quote": {"type": "string"}}

SCHEMA = {
    "type": "object",
    "properties": {
        "passages": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "integer"}, "relevant": {"type": "boolean"}}, "required": ["id", "relevant"]}},
        "gold_support": {"type": "object", "properties": _QB, "required": ["supported", "quote"]},
        "sentences": {"type": "array", "items": {"type": "object", "properties": {
            "idx": {"type": "integer"}, **_QB,
            "cited": {"type": "array", "items": {"type": "object", "properties": {
                "passage": {"type": "integer"}, **_QB}, "required": ["passage", "supported", "quote"]}}},
            "required": ["idx", "supported", "quote", "cited"]}},
        "answer_relevance": {"type": "integer"},
    },
    "required": ["passages", "gold_support", "sentences", "answer_relevance"],
}

SYSTEM = """You are a strict, literal fact-checking judge for a question-answering system. Use ONLY the numbered
passages. Do not use outside knowledge: a true fact that is not written in the passages is NOT supported.
Every quote must be copied word for word FROM A PASSAGE (never from the answer or the reference), at most
25 words; use "" when the label is false.

Fill every field:
- passages: one entry for EVERY passage id; relevant = it contains information needed to give the reference answer.
- gold_support: supported = the passages state the reference answer; quote = the passage words that state it.
- sentences: one entry for EVERY answer sentence idx:
    supported = some passage states or directly implies the sentence; quote = those passage words.
    cited = one entry for EVERY passage the sentence cites (shown as "cites: [n]"):
      supported = that one passage supports the sentence; quote = words from that passage.
- answer_relevance: 1 (ignores the question) to 5 (directly and completely answers it); 1 if there is no answer.
Return JSON only."""


class JudgeIncomplete(OllamaError):
    """The judge left labels out even after one repair call. Recorded per row, never silently filled."""


def _family(model: str) -> str:
    name = model.split(":")[0].lower()
    for fam in ("qwen", "llama", "gemma", "mistral", "phi", "deepseek"):
        if name.startswith(fam):
            return fam
    return name


def quote_in(quote: str, texts: Sequence[str], min_cover: float = 0.8) -> bool:
    """A judge quote is genuine if it occurs (after normalisation) in one of the texts.

    Exact normalised substring, or ≥ `min_cover` of its word 3-grams found in a single text
    (tolerates small copy errors). Quotes under 3 words are rejected: too easy to match by chance.
    """
    q = normalize_answer(quote)
    if len(q.split()) < 3:
        return False
    qs = _shingles(q, 3)
    for t in texts:
        nt = normalize_answer(t)
        if q in nt or len(qs & _shingles(nt, 3)) / len(qs) >= min_cover:
            return True
    return False


class Judge:
    def __init__(self, cfg: Dict[str, Any], client: OllamaClient):
        self.client, self.model = client, cfg["models"]["judge"]
        gen = cfg["models"]["generator"]
        if _family(self.model) == _family(gen):
            raise ValueError(f"Judge {self.model} shares a model family with generator {gen} (self-preference bias)")
        j = cfg["judge"]
        self.opts = {"temperature": float(j["temperature"]), "max_tokens": int(j["max_tokens"]),
                     "num_ctx": int(j["num_ctx"])}

    def _ask(self, prompt: str):
        r = self.client.generate(prompt, model=self.model, caller="judge", system=SYSTEM, json_schema=SCHEMA,
                                 state="JUDGE", **self.opts)
        o = r.json()
        if not isinstance(o, dict):
            raise OllamaError(f"judge returned non-object JSON (call {r.call_id})")
        return r, o

    def judge(self, question: str, answer: Optional[str], evidence: Sequence[Dict[str, Any]],
              gold: Optional[str]) -> Optional[Dict[str, Any]]:
        """`answer=None` for an abstention (only context metrics); `gold=None` when there is no reference."""
        if not evidence:
            return None
        text_by_id = {int(e["cite_id"]): e["text"] for e in evidence}
        sents: List[Tuple[str, List[int]]] = [(t, [c for c in ids if c in text_by_id])
                                              for t, ids in (sentence_citations(answer) if answer else [])]
        if not sents and not gold:
            return None

        passages = "\n\n".join(f"[{cid}] {t}" for cid, t in text_by_id.items())
        sent_lines = "\n".join(f"S{i}: {s}\n    cites: {', '.join(f'[{c}]' for c in ids) or 'nothing'}"
                               for i, (s, ids) in enumerate(sents)) or "(no answer: the system abstained)"
        prompt = (f"Question: {question}\n\nReference answer: {gold or '(none)'}\n\nPassages:\n{passages}\n\n"
                  f"Answer sentences:\n{sent_lines}\n\n"
                  f"Label passages {sorted(text_by_id)} and sentences 0..{len(sents) - 1 if sents else -1}.")

        def parse(o):
            rel = {int(x["id"]): bool(x["relevant"]) for x in o.get("passages", []) if isinstance(x, dict) and "id" in x}
            s_by = {int(x["idx"]): x for x in o.get("sentences", []) if isinstance(x, dict) and "idx" in x}
            pairs = {}
            for i, x in s_by.items():
                for c in x.get("cited") or []:
                    if isinstance(c, dict) and "passage" in c:
                        pairs[(i, int(c["passage"]))] = c
            miss = ([f"passage {i}" for i in text_by_id if i not in rel] if gold else []) + \
                   [f"sentence S{i}" for i in range(len(sents)) if i not in s_by] + \
                   [f"citation [{c}] of S{i}" for i, (_, ids) in enumerate(sents) for c in ids if (i, c) not in pairs]
            return rel, s_by, pairs, miss

        res, out = self._ask(prompt)
        calls = [res]
        rel, s_by, pairs, miss = parse(out)
        if miss:  # one repair call naming exactly what was skipped (temp 0: the prompt must change)
            res, out = self._ask(prompt + f"\n\nYour previous answer left out: {', '.join(miss)}. "
                                          "Return the COMPLETE JSON with every label.")
            calls.append(res)
            rel, s_by, pairs, miss = parse(out)
            if miss:
                raise JudgeIncomplete(f"judge skipped {', '.join(miss)} (calls {[c.call_id for c in calls]})")
        call = {"call_ids": [c.call_id for c in calls], "latency_ms": sum(c.latency_ms for c in calls),
                "cache_hit": all(c.cache_hit for c in calls), "n_calls": len(calls),
                "prompt_tokens": sum(c.prompt_tokens for c in calls),
                "completion_tokens": sum(c.completion_tokens for c in calls)}

        overruled = 0

        def checked(label: Dict[str, Any], texts: Sequence[str]) -> bool:
            nonlocal overruled
            if not label.get("supported"):
                return False
            if quote_in(str(label.get("quote", "")), texts):
                return True
            overruled += 1
            return False

        scores: Dict[str, Any] = {}
        all_texts = list(text_by_id.values())
        n_labels = 0
        if gold:
            hits, precs = 0, []
            for k, e in enumerate(evidence, 1):  # packed order = score order
                if rel[int(e["cite_id"])]:
                    hits += 1
                    precs.append(hits / k)
            scores["context_precision"] = (sum(precs) / hits) if hits else 0.0
            scores["context_recall"] = float(checked(out.get("gold_support") or {}, all_texts))
            n_labels += 1
        if sents:
            pair_ok = {(i, c): checked(pairs[(i, c)], [text_by_id[c]]) for i, (_, ids) in enumerate(sents) for c in ids}
            faith = [checked(s_by[i], all_texts) or any(pair_ok[(i, c)] for c in ids) for i, (_, ids) in enumerate(sents)]
            # ALCE citation recall: one of the sentence's own citations supports it; uncited ⇒ not recalled
            recalled = [any(pair_ok[(i, c)] for c in ids) for i, (_, ids) in enumerate(sents)]
            n_labels += len(sents) + len(pair_ok)
            scores.update({
                "faithfulness": sum(faith) / len(faith),
                "citation_recall": sum(recalled) / len(recalled),
                "citation_precision": (sum(pair_ok.values()) / len(pair_ok)) if pair_ok else None,
                "answer_relevance": (min(5, max(1, int(out.get("answer_relevance", 1)))) - 1) / 4,
                "n_sentences": len(sents), "n_pairs": len(pair_ok),
            })
        scores.update({"overruled": overruled, "n_support_labels": n_labels})
        return {"scores": scores, "raw": out, "call": call}
