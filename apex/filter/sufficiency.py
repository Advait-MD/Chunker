"""Sufficiency check and the one-shot query rewrite (spec §2.6 step 5, M5).

"Use an NLI or cheap classifier first, and an LLM judge only if the result is ambiguous." The cheap
classifier is the cross-encoder relevance the reranker already computed: the best reranker score among
the passages that survived the filter.
  best >= sufficient_above              → sufficient
  best <  insufficient_below            → insufficient
  otherwise                             → ambiguous, resolved by `ambiguous`:
      llm        one temperature-0 call: does the evidence contain enough to answer? (JSON yes/no)
      sufficient treat it as sufficient (the generator can still abstain on its own)
Both thresholds are tuned on dev (results/m5_*_dev). On `insufficient` the pipeline rewrites the query
once, retrieves again and re-checks; if it is still insufficient it abstains.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from apex.generate.generator import EvidenceSet, build_user_prompt
from apex.llm import OllamaClient

PROMPT_DIR = Path(__file__).resolve().parents[1] / "generate" / "prompts"

SUFF_SCHEMA = {"type": "object", "properties": {"sufficient": {"type": "boolean"}}, "required": ["sufficient"]}
REWRITE_SCHEMA = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}


class SufficiencyChecker:
    def __init__(self, cfg: Dict[str, Any], client: OllamaClient):
        s = cfg["filter"]["sufficiency"]
        self.enabled = bool(s["enabled"])
        self.hi, self.lo = float(s["sufficient_above"]), float(s["insufficient_below"])
        if self.lo > self.hi:
            raise ValueError("filter.sufficiency.insufficient_below must be <= sufficient_above")
        self.ambiguous = s["ambiguous"]
        if self.ambiguous not in ("llm", "sufficient"):
            raise ValueError(f"filter.sufficiency.ambiguous must be llm or sufficient, got {self.ambiguous!r}")
        self.client = client
        self.model = s.get("model") or cfg["models"]["generator"]
        self.num_ctx = int(cfg["generation"]["num_ctx"])
        self.system = (PROMPT_DIR / s["prompt"]).read_text(encoding="utf-8")

    def check(self, query: str, evidence: EvidenceSet) -> Dict[str, Any]:
        if not self.enabled:
            return {"verdict": "sufficient", "by": "disabled"}
        if not evidence.items:
            return {"verdict": "insufficient", "by": "empty", "best": None}
        best = max(it.score for it in evidence.items)
        if best >= self.hi:
            return {"verdict": "sufficient", "by": "rerank", "best": best}
        if best < self.lo:
            return {"verdict": "insufficient", "by": "rerank", "best": best}
        if self.ambiguous == "sufficient":
            return {"verdict": "sufficient", "by": "ambiguous_default", "best": best}
        res = self.client.generate(build_user_prompt(query, evidence).replace("Answer in JSON:", "Reply in JSON:"),
                                   model=self.model, caller="filter.sufficiency", system=self.system,
                                   temperature=0.0, max_tokens=32, num_ctx=self.num_ctx,
                                   json_schema=SUFF_SCHEMA, state="FILTERING")
        ok = bool(res.json().get("sufficient"))
        return {"verdict": "sufficient" if ok else "insufficient", "by": "llm", "best": best, "call_id": res.call_id}


class QueryRewriter:
    def __init__(self, cfg: Dict[str, Any], client: OllamaClient):
        r = cfg["filter"]["rewrite"]
        self.enabled = bool(r["enabled"])
        self.client = client
        self.model = r.get("model") or cfg["models"]["generator"]
        self.max_tokens = int(r["max_tokens"])
        self.system = (PROMPT_DIR / r["prompt"]).read_text(encoding="utf-8")

    def rewrite(self, query: str) -> Optional[str]:
        res = self.client.generate(f"Question:\n{query}\n\nRewritten search query in JSON:", model=self.model,
                                   caller="filter.rewrite", system=self.system, temperature=0.0,
                                   max_tokens=self.max_tokens, json_schema=REWRITE_SCHEMA, state="FILTERING")
        q = str(res.json().get("query") or "").strip()
        return q or None
