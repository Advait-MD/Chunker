"""
eval/judge.py — LLM-as-a-Judge Engine for Evaluating Generation Quality.
Implements faithfulness, correctness, and abstention checks per HARNESS.md Section 8.
Uses a distinct judge model with temperature=0 and strict JSON schema validation.
"""

import re
import json
import requests
from typing import Dict, Any, List, Optional
from eval.cache import ResponseCache

JUDGE_PROMPT_VERSION = "2026-10-23.v1"

JUDGE_SYSTEM_PROMPT = f"""You are an impartial, strict evaluation judge for a RAG (Retrieval-Augmented Generation) system.
Version: {JUDGE_PROMPT_VERSION}

Evaluate the candidate answer against the provided context and ground truth.
You must output a single, valid JSON object with EXACTLY the following structure:

{{
  "faithfulness": boolean,
  "correctness": boolean,
  "abstained": boolean,
  "reasoning": "brief explanation (1-2 sentences)"
}}

Evaluation Criteria:
1. FAITHFULNESS (boolean):
   - Set to true if EVERY factual claim in the Candidate Answer is directly supported or substantiated by the Retrieved Context Chunks.
   - Set to false if the answer introduces unsupported assertions, hallucinated facts, or claims not found in the chunks.
   - If Candidate Answer abstained / declined to answer due to lack of evidence, faithfulness is true.

2. CORRECTNESS (boolean):
   - Only evaluated when Answerable is true.
   - Set to true if the Candidate Answer semantically conveys the key facts of the Gold Answer.
   - Set to false if it contradicts the Gold Answer, gives wrong names/dates, or fails to answer the question.

3. ABSTAINED (boolean):
   - Set to true if the Candidate Answer explicitly declined to answer, admitted it does not know, noted insufficient context, or rejected an impossible premise.
   - Set to false if the candidate attempted to answer the question as factual.
   - For Answerable=false questions: abstained=true means successful abstention; abstained=false means hallucination.
"""


def format_judge_prompt(
    question: str,
    answer: str,
    retrieved_chunks: List[str],
    gold_answer: Optional[str],
    answerable: bool
) -> str:
    """Construct structured evaluation prompt for the LLM judge."""
    # Bounded to top 4 retrieved chunks (max 500 chars each) to prevent CPU prompt-eval timeout
    bounded_chunks = [c[:500].strip() for c in (retrieved_chunks or [])[:4]]
    chunks_text = "\n---\n".join([f"[Chunk {i+1}]: {c}" for i, c in enumerate(bounded_chunks)]) if bounded_chunks else "[No chunks retrieved]"
    gold_text = gold_answer if gold_answer is not None else "[UNANSWERABLE / ADVERSARIAL QUESTION - EXPECT ABSTENTION]"
    
    return f"""### INPUTS TO EVALUATE:

[Question]:
{question}

[Answerable Expected]:
{answerable}

[Gold Answer]:
{gold_text}

[Retrieved Context Chunks]:
{chunks_text}

[Candidate Answer]:
{answer}

Evaluate the Candidate Answer now. Return ONLY the strict JSON object."""


class LLMJudge:
    """Evaluates generation quality using an independent local Ollama model."""

    def __init__(
        self,
        model: str = "llama3.2:1b",
        temperature: float = 0.0,
        ollama_url: str = "http://localhost:11434/api/generate",
        cache: Optional[ResponseCache] = None,
        timeout: int = 90
    ):
        self.model = model
        self.temperature = float(temperature)
        self.ollama_url = ollama_url
        self.cache = cache or ResponseCache()
        self.timeout = timeout
        self.session = requests.Session()

    def _call_llm(self, prompt: str) -> str:
        """Call Ollama with caching and temperature=0."""
        # 1. Check disk cache
        cached = self.cache.get(self.model, self.temperature, prompt)
        if cached is not None:
            return cached

        # 2. Query Ollama with JSON format constraint
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": JUDGE_SYSTEM_PROMPT,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.temperature,
                "num_predict": 256
            }
        }
        res = self.session.post(self.ollama_url, json=payload, timeout=self.timeout)
        if res.status_code != 200:
            raise RuntimeError(f"Ollama judge call failed ({res.status_code}): {res.text}")
        
        response_text = res.json().get("response", "").strip()
        self.cache.put(self.model, self.temperature, prompt, response_text)
        return response_text

    def _parse_and_validate(self, text: str) -> Dict[str, Any]:
        """Validate JSON schema returned by judge."""
        # Attempt direct JSON parse or regex extract
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
        else:
            parsed = json.loads(text)

        required_keys = {"faithfulness", "correctness", "abstained"}
        if not required_keys.issubset(parsed.keys()):
            raise ValueError(f"Missing required keys in judge response: {parsed.keys()}")

        return {
            "faithfulness": bool(parsed["faithfulness"]),
            "correctness": bool(parsed["correctness"]),
            "abstained": bool(parsed["abstained"]),
            "reasoning": str(parsed.get("reasoning", "")).strip()
        }

    def judge_case(
        self,
        question: str,
        answer: str,
        retrieved_chunks: List[str],
        gold_answer: Optional[str],
        answerable: bool
    ) -> Dict[str, Any]:
        """
        Evaluate a single case with retry-once logic.
        Returns dict with faithfulness, correctness, abstained, reasoning, judge_error.
        """
        prompt = format_judge_prompt(question, answer, retrieved_chunks, gold_answer, answerable)

        # Attempt 1
        try:
            raw_text = self._call_llm(prompt)
            result = self._parse_and_validate(raw_text)
            result["judge_error"] = False
            return result
        except Exception:
            pass

        # Attempt 2 (Retry once on parse or network failure)
        try:
            retry_prompt = prompt + "\n\nCRITICAL: Respond ONLY with valid RFC8259 JSON matching the required schema."
            raw_text = self._call_llm(retry_prompt)
            result = self._parse_and_validate(raw_text)
            result["judge_error"] = False
            return result
        except Exception as e:
            return {
                "faithfulness": None,
                "correctness": None,
                "abstained": False,
                "reasoning": f"Judge error after retry: {str(e)}",
                "judge_error": True
            }
