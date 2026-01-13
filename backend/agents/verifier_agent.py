from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..llm import call_groq_chat
from .retriever_agent import RetrieverOutput


NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
LOCATION_TERMS = {"city", "town", "county", "state", "florida"}
FINANCE_TERMS = {"stock", "share", "nasdaq", "company", "ticker"}


@dataclass
class VerificationResult:
    final_answer: str
    hallucination_detected: bool
    confidence: float
    notes: str
    trusted_facts: Dict[str, any]


def _extract_numbers(text: str) -> List[float]:
    values = []
    for match in NUMBER_PATTERN.findall(text):
        try:
            values.append(float(match))
        except ValueError:
            continue
    return values


def _numbers_consistent(answer_nums: List[float], fact_nums: List[float]) -> bool:
    if not answer_nums:
        return True
    for num in answer_nums:
        if any(math.isclose(num, fact, rel_tol=1e-2, abs_tol=1e-2) for fact in fact_nums):
            return True
    return False


def _summarize_metadatas(metadatas: List[Dict[str, str]]) -> str:
    rows = []
    for meta in metadatas:
        planet = meta.get("planet", "Unknown")
        summary_parts = []
        for key, value in meta.items():
            if key == "planet" or not value:
                continue
            summary_parts.append(f"{key}: {value}")
        rows.append(f"{planet} -> " + ", ".join(summary_parts[:12]))
    return "\n".join(rows)


class VerifierAgent:
    """Validates baseline answers and produces corrected outputs if needed."""

    def __init__(self, *, temperature: float = 0.1) -> None:
        self.temperature = temperature

    def verify(
        self,
        question: str,
        baseline_answer: str,
        retrieval: RetrieverOutput,
    ) -> VerificationResult:
        local_context = "\n\n".join(retrieval.local_results.documents)
        web_context = "\n\n".join(retrieval.web_snippets)
        metadata_summary = _summarize_metadatas(retrieval.local_results.metadatas)

        prompt = (
            "Determine whether the baseline answer is fully supported by the trusted facts.\n"
            "Return JSON with keys: final_answer (string), hallucination (bool), "
            "confidence (0-1), notes (string), trusted_facts (object summarizing facts used).\n"
            "Use local facts as primary truth. Only rely on web snippets if marked available."
        )
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Baseline answer: {baseline_answer}\n"
                    f"Local context:\n{local_context}\n\n"
                    f"Metadata summary:\n{metadata_summary}\n\n"
                    f"Web context:\n{web_context or 'None'}"
                ),
            },
        ]
        response = call_groq_chat(messages, temperature=self.temperature)
        result = self._parse_response(response, baseline_answer, retrieval, question)
        return result

    def _parse_response(
        self,
        text: str,
        baseline_answer: str,
        retrieval: RetrieverOutput,
        question: str,
    ) -> VerificationResult:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = {
                "final_answer": baseline_answer,
                "hallucination": False,
                "confidence": 0.5,
                "notes": text,
                "trusted_facts": {},
            }

        final_answer = payload.get("final_answer", baseline_answer).strip()
        hallucination = bool(payload.get("hallucination", False))
        confidence = float(payload.get("confidence", 0.5))
        notes = payload.get("notes", "").strip()
        trusted_facts = payload.get("trusted_facts") or {}

        # heuristic numeric consistency check
        answer_nums = _extract_numbers(final_answer or baseline_answer)
        fact_nums = []
        for doc in retrieval.local_results.documents:
            fact_nums.extend(_extract_numbers(doc))
        if not _numbers_consistent(answer_nums, fact_nums):
            hallucination = True
            notes += " | Numeric mismatch detected."
            confidence = min(confidence, 0.5)

        lower_answer = final_answer.lower()
        lower_question = question.lower()
        
        # Check for question-answer misalignment
        question_keywords = set(re.findall(r'\b\w+\b', lower_question))
        answer_keywords = set(re.findall(r'\b\w+\b', lower_answer))
        
        # Extract main question topic (e.g., "forests", "population", "mountains")
        question_topic = None
        for word in ["forest", "castle", "population", "gdp", "visa", "stock", "capital", "mountain", "wall", "year"]:
            if word in lower_question:
                question_topic = word
                break
        
        # If question asks for something specific, check if answer addresses it
        if question_topic:
            if question_topic not in lower_answer and "cannot" not in lower_answer and "don't know" not in lower_answer:
                # Answer doesn't address the question topic - likely hallucination
                hallucination = True
                notes += f" | Answer does not address question topic '{question_topic}'."
                confidence = min(confidence, 0.3)
        
        # Check for completely unrelated answers (e.g., answering about moons when asked about forests)
        if question_topic == "forest" and ("moon" in lower_answer or "ring" in lower_answer):
            hallucination = True
            notes += " | Answer addresses different topic than question."
            confidence = 0.2
        
        if any(term in lower_answer for term in LOCATION_TERMS) and "planet" in lower_question:
            hallucination = True
            notes += " | Location term detected for planetary question."
            confidence = min(confidence, 0.4)
        if any(term in lower_answer for term in FINANCE_TERMS) and "planet" in lower_question:
            hallucination = True
            notes += " | Financial term detected for planetary question."
            confidence = min(confidence, 0.4)

        return VerificationResult(
            final_answer=final_answer or baseline_answer,
            hallucination_detected=hallucination,
            confidence=max(0.0, min(1.0, confidence)),
            notes=notes,
            trusted_facts=trusted_facts or {
                "local": retrieval.local_results.metadatas,
                "web": retrieval.web_results if retrieval.web_results else [],
            },
        )

