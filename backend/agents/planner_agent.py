from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from ..llm import call_groq_chat

RoutingStrategy = Literal["local_only", "local_and_web"]


@dataclass
class PlannerDecision:
    confidence: float
    strategy: RoutingStrategy
    critique: str


def _parse_response(text: str) -> PlannerDecision:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # fallback: naive parsing
        strategy = "local_only"
        confidence = 0.5
        if "web" in text.lower():
            strategy = "local_and_web"
        return PlannerDecision(
            confidence=confidence,
            strategy=strategy,  # type: ignore[arg-type]
            critique=text.strip(),
        )

    strategy = payload.get("strategy", "local_only")
    if strategy not in {"local_only", "local_and_web"}:
        strategy = "local_only"
    confidence = float(payload.get("confidence", 0.5))
    critique = payload.get("critique", "").strip()
    return PlannerDecision(
        confidence=max(0.0, min(1.0, confidence)),
        strategy=strategy,  # type: ignore[arg-type]
        critique=critique,
    )


class PlannerAgent:
    """Runs LLM self-critique on the baseline answer and chooses retrieval strategy."""

    def __init__(self, *, base_temperature: float = 0.3) -> None:
        self.base_temperature = base_temperature

    def plan(
        self,
        question: str,
        baseline_answer: str,
        context_excerpt: str,
    ) -> PlannerDecision:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a planning agent deciding whether a baseline answer is reliable. "
                    "Return JSON with keys: critique (string), confidence (0-1), strategy "
                    "('local_only' or 'local_and_web'). Choose 'local_and_web' when baseline "
                    "lacks certainty, references missing data, or needs external confirmation."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Baseline answer: {baseline_answer}\n"
                    f"Context excerpt: {context_excerpt}\n"
                    "Respond with JSON."
                ),
            },
        ]
        response = call_groq_chat(messages, temperature=self.base_temperature)
        return _parse_response(response)

