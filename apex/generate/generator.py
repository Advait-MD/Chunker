"""Grounded, cited answer generation through Ollama with JSON-schema output."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from apex.index.chroma_store import Hit
from apex.ingest.chunker import count_tokens
from apex.llm import OllamaClient, OllamaError

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

ANSWER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "short_answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "integer"}},
        "confidence": {"type": "number"},
        "abstained": {"type": "boolean"},
    },
    "required": ["answer", "short_answer", "citations", "confidence", "abstained"],
}


@dataclass
class EvidenceItem:
    cite_id: int
    node_id: str
    text: str
    score: float
    source: str
    page: Any = None
    section: Any = None
    char_span: Any = None

    def label(self) -> str:
        loc = f", p. {self.page}" if self.page is not None else ""
        if self.section:
            loc += f", § {self.section}"
        return f"[{self.cite_id}] (source: {self.source}{loc})"


@dataclass
class EvidenceSet:
    """Spec §2.6 output contract. M1 fills items only; filtering arrives in M5."""
    items: List[EvidenceItem] = field(default_factory=list)
    sufficient: bool = True
    conflicts: List[Any] = field(default_factory=list)


def pack_evidence(hits: List[Hit], token_budget: int) -> EvidenceSet:
    """Order by score, fit the generator budget, assign cite ids [1..n]. Always keeps the top hit."""
    items: List[EvidenceItem] = []
    used = 0
    for h in sorted(hits, key=lambda x: x.score, reverse=True):
        t = h.node.token_count or count_tokens(h.node.text)
        if items and used + t > token_budget:
            continue
        used += t
        items.append(EvidenceItem(cite_id=len(items) + 1, node_id=h.node.id, text=h.node.text, score=h.score,
                                  source=h.node.source_path, page=h.node.page, section=h.node.section,
                                  char_span=h.node.char_span))
    return EvidenceSet(items=items)


def build_user_prompt(query: str, evidence: EvidenceSet) -> str:
    blocks = "\n\n".join(f"{it.label()}\n{it.text}" for it in evidence.items)
    return f"Evidence passages:\n\n{blocks}\n\nQuestion: {query}\n\nAnswer in JSON:"


class GenerationError(OllamaError):
    """The generator produced no usable answer (truncated at max_tokens, invalid JSON, missing keys)."""


class Generator:
    def __init__(self, cfg: Dict[str, Any], client: OllamaClient):
        g = cfg["generation"]
        self.client = client
        self.model = cfg["models"]["generator"]
        self.temperature = float(g["temperature"])
        self.max_tokens = int(g["max_tokens"])
        self.num_ctx = int(g["num_ctx"])
        self.system = (PROMPT_DIR / g["system_prompt"]).read_text(encoding="utf-8")

    def generate(self, query: str, evidence: EvidenceSet) -> Dict[str, Any]:
        res = self.client.generate(build_user_prompt(query, evidence), model=self.model, caller="generate.answer",
                                   system=self.system, temperature=self.temperature, max_tokens=self.max_tokens,
                                   num_ctx=self.num_ctx, json_schema=ANSWER_SCHEMA, state="GENERATING")
        if res.raw.get("done_reason") == "length":
            raise GenerationError(f"Generator output truncated at max_tokens={self.max_tokens} "
                                  f"(call {res.call_id}): {res.text[:200]!r}")
        try:
            out = res.json()
        except OllamaError as e:
            raise GenerationError(str(e)) from e
        missing = [k for k in ANSWER_SCHEMA["required"] if k not in out]
        if missing:
            raise GenerationError(f"Generator output missing keys {missing}: {res.text[:200]!r}")
        out["_call_id"] = res.call_id
        return out
