"""Extract factual claims from model answers."""

from __future__ import annotations

import json
import re
from typing import Dict, Any, Optional

from .llm import call_groq_chat

# Field mappings from CSV to normalized keys
FIELD_MAPPINGS = {
    "number_of_moons": ["moons", "moon", "satellite", "satellites"],
    "surface_pressure": ["surface pressure", "pressure", "atmospheric pressure"],
    "mean_temperature": ["temperature", "mean temperature", "average temperature", "temp"],
    "gravity": ["gravity", "gravitational", "g"],
    "orbital_period": ["orbital period", "year", "revolution", "orbit"],
    "distance_from_sun": ["distance", "distance from sun", "semi-major axis"],
    "mass": ["mass", "weight"],
    "diameter": ["diameter", "size", "radius"],
    "density": ["density"],
    "escape_velocity": ["escape velocity", "escape speed"],
}


def _extract_numbers(text: str) -> list[float]:
    """Extract all numeric values from text."""
    pattern = r"-?\d+(?:\.\d+)?"
    numbers = []
    for match in re.finditer(pattern, text):
        try:
            numbers.append(float(match.group()))
        except ValueError:
            continue
    return numbers


def _normalize_field_name(text: str) -> str | None:
    """Try to match text to a known field name."""
    text_lower = text.lower()
    for field, keywords in FIELD_MAPPINGS.items():
        if any(keyword in text_lower for keyword in keywords):
            return field
    return None


def extract_planet_facts(answer_text: str, planet_name: str | None = None) -> Dict[str, Any]:
    """
    Parse key factual claims from the model's answer.
    
    Returns a normalized dict with fields like:
    {
      "planet": "Mars",
      "number_of_moons": 2,
      "surface_pressure_bars": 0.01,
      "mean_temperature_celsius": -65,
      ...
    }
    """
    if not answer_text or "cannot" in answer_text.lower() or "don't know" in answer_text.lower():
        return {"planet": planet_name} if planet_name else {}
    
    # Try LLM-based extraction first for better accuracy
    try:
        prompt = (
            f"Extract factual claims from this answer about {planet_name or 'a planet'}:\n\n"
            f"{answer_text}\n\n"
            "Return ONLY a JSON object with these fields (use null if not found):\n"
            "- planet: string\n"
            "- number_of_moons: integer or null\n"
            "- surface_pressure_bars: float or null\n"
            "- mean_temperature_celsius: float or null\n"
            "- gravity_mps2: float or null\n"
            "- orbital_period_days: float or null\n"
            "- distance_from_sun_km: float or null\n"
            "- mass_10e24kg: float or null\n"
            "- diameter_km: float or null\n"
            "\nExample: {\"planet\": \"Mars\", \"number_of_moons\": 2, \"surface_pressure_bars\": 0.01}\n"
            "Return ONLY the JSON, no other text."
        )
        
        response = call_groq_chat(
            messages=[
                {"role": "system", "content": "You are a fact extractor. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
        )
        
        # Try to extract JSON from response
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            extracted = json.loads(json_match.group())
            if planet_name:
                extracted["planet"] = planet_name
            return extracted
    except Exception:
        pass
    
    # Fallback: simple regex-based extraction
    facts: Dict[str, Any] = {"planet": planet_name} if planet_name else {}
    
    answer_lower = answer_text.lower()
    
    # Extract number of moons
    moon_patterns = [
        r"(\d+)\s+moons?",
        r"moons?[:\s]+(\d+)",
        r"has\s+(\d+)\s+moons?",
    ]
    for pattern in moon_patterns:
        match = re.search(pattern, answer_lower)
        if match:
            try:
                facts["number_of_moons"] = int(match.group(1))
                break
            except ValueError:
                continue
    
    # Extract temperature
    temp_patterns = [
        r"temperature[:\s]+(-?\d+(?:\.\d+)?)",
        r"(-?\d+(?:\.\d+)?)\s*degrees?\s*celsius",
        r"(-?\d+(?:\.\d+)?)\s*°c",
    ]
    for pattern in temp_patterns:
        match = re.search(pattern, answer_lower)
        if match:
            try:
                facts["mean_temperature_celsius"] = float(match.group(1))
                break
            except ValueError:
                continue
    
    # Extract pressure
    pressure_patterns = [
        r"pressure[:\s]+(\d+(?:\.\d+)?)\s*bars?",
        r"(\d+(?:\.\d+)?)\s*bars?\s*pressure",
    ]
    for pattern in pressure_patterns:
        match = re.search(pattern, answer_lower)
        if match:
            try:
                facts["surface_pressure_bars"] = float(match.group(1))
                break
            except ValueError:
                continue
    
    return facts

