"""Question classification: factual vs nonsense/out-of-scope."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from . import config
from .llm import call_groq_chat

# Known planets from the CSV
KNOWN_PLANETS = {
    "mercury", "venus", "earth", "moon", "mars", "jupiter",
    "saturn", "uranus", "neptune", "pluto"
}

# Keywords that suggest nonsense/out-of-scope questions
NONSENSE_KEYWORDS = {
    "visa", "policy", "president", "prime minister", "population",
    "gdp", "unemployment", "economy", "government", "citizen",
    "resident", "immigration", "currency", "stock", "price", "market",
    "capital city", "city", "country", "nation", "flag", "anthem"
}


@dataclass
class QuestionClassification:
    """Classification result for a question."""
    question_type: str  # "planet_fact" or "nonsense_or_out_of_scope"
    is_answerable_from_local: bool
    detected_planet: str | None = None


def _extract_planet_name(question: str) -> str | None:
    """Extract planet name from question if present."""
    question_lower = question.lower()
    for planet in KNOWN_PLANETS:
        if planet in question_lower:
            return planet.capitalize()
    return None


def _has_nonsense_keywords(question: str) -> bool:
    """Check if question contains nonsense/out-of-scope keywords."""
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in NONSENSE_KEYWORDS)


def classify_question(question: str) -> QuestionClassification:
    """
    Classify a question as factual (planet_fact) or nonsense/out-of-scope.
    
    Uses heuristic checks first, then LLM for ambiguous cases.
    """
    question = question.strip()
    if not question:
        return QuestionClassification(
            question_type="nonsense_or_out_of_scope",
            is_answerable_from_local=False,
        )
    
    # Step 1: Check if question mentions a known planet
    planet = _extract_planet_name(question)
    
    # Step 2: Check for obvious nonsense keywords
    if _has_nonsense_keywords(question):
        return QuestionClassification(
            question_type="nonsense_or_out_of_scope",
            is_answerable_from_local=False,
            detected_planet=planet,
        )
    
    # Step 3: If planet found and no nonsense keywords, use LLM to verify
    if planet:
        # Ask LLM to determine if this is about physical/astronomical properties
        prompt = (
            f"Question: {question}\n\n"
            "Is this question asking about physical/astronomical properties of {planet} "
            "(like mass, diameter, temperature, moons, orbital period, etc.)?\n\n"
            "Or is it asking about human/social/legal concepts "
            "(like population, government, visa policy, economy, etc.)?\n\n"
            "Answer with ONLY one word: 'physical' or 'social'."
        )
        
        try:
            response = call_groq_chat(
                messages=[
                    {"role": "system", "content": "You are a question classifier. Answer with only one word."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
            )
            response_lower = response.strip().lower()
            
            if "physical" in response_lower or "astronomical" in response_lower:
                return QuestionClassification(
                    question_type="planet_fact",
                    is_answerable_from_local=True,
                    detected_planet=planet,
                )
            else:
                return QuestionClassification(
                    question_type="nonsense_or_out_of_scope",
                    is_answerable_from_local=False,
                    detected_planet=planet,
                )
        except Exception:
            # If LLM fails, default to planet_fact if planet is detected
            return QuestionClassification(
                question_type="planet_fact",
                is_answerable_from_local=True,
                detected_planet=planet,
            )
    
    # Step 4: No planet detected - likely out of scope
    return QuestionClassification(
        question_type="nonsense_or_out_of_scope",
        is_answerable_from_local=False,
    )

