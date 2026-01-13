"""Hallucination detection through fact consistency checking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Any, List

from .fact_extraction import extract_planet_facts
from .facts_lookup import get_ground_truth_facts


@dataclass
class Mismatch:
    """Represents a mismatch between answer and ground truth."""
    field: str
    answer_value: Any
    true_value: Any


@dataclass
class HallucinationResult:
    """Result of hallucination detection."""
    hallucination_detected: bool
    mismatches: List[Mismatch]
    confidence: float = 1.0


def _values_match(answer_val: Any, true_val: Any, tolerance: float = 0.01) -> bool:
    """Check if two values match (with tolerance for floats)."""
    if answer_val is None or true_val is None:
        return False
    
    try:
        answer_float = float(answer_val)
        true_float = float(true_val)
        
        # Use relative tolerance for comparison
        if true_float == 0:
            return abs(answer_float) < tolerance
        
        rel_error = abs(answer_float - true_float) / abs(true_float)
        return rel_error < tolerance or abs(answer_float - true_float) < tolerance
    except (ValueError, TypeError):
        # For non-numeric, do exact match
        return str(answer_val).lower().strip() == str(true_val).lower().strip()


def detect_hallucination_for_planet_fact(
    extracted_facts: Dict[str, Any],
    ground_truth: Dict[str, Any],
    tolerance: float = 0.05,
) -> HallucinationResult:
    """
    Compare extracted facts against ground-truth facts.
    
    Returns:
    {
      "hallucination_detected": bool,
      "mismatches": [
        {"field": "number_of_moons", "answer_value": 3, "true_value": 2},
        ...
      ]
    }
    """
    mismatches: List[Mismatch] = []
    
    # Fields to check (numeric fields only)
    fields_to_check = [
        "number_of_moons",
        "surface_pressure_bars",
        "mean_temperature_celsius",
        "gravity_mps2",
        "orbital_period_days",
        "distance_from_sun_km",
        "mass_10e24kg",
        "diameter_km",
    ]
    
    for field in fields_to_check:
        answer_val = extracted_facts.get(field)
        true_val = ground_truth.get(field)
        
        # Only check if both values are present
        if answer_val is not None and true_val is not None:
            if not _values_match(answer_val, true_val, tolerance):
                mismatches.append(
                    Mismatch(
                        field=field,
                        answer_value=answer_val,
                        true_value=true_val,
                    )
                )
    
    # Calculate confidence based on matches
    total_checked = len([f for f in fields_to_check 
                        if extracted_facts.get(f) is not None 
                        and ground_truth.get(f) is not None])
    
    if total_checked == 0:
        # No facts to check - can't determine
        confidence = 0.5
    else:
        matches = total_checked - len(mismatches)
        confidence = matches / total_checked if total_checked > 0 else 0.0
    
    return HallucinationResult(
        hallucination_detected=len(mismatches) > 0,
        mismatches=mismatches,
        confidence=confidence,
    )


def evaluate_refusal_behavior(
    question_type: str,
    final_answer: str,
) -> Dict[str, bool]:
    """
    Evaluate if the answer correctly refuses to answer nonsense questions.
    
    For nonsense_or_out_of_scope:
      - refusal_expected = True
      - refusal_correct = True if final_answer expresses clear inability
    
    For planet_fact:
      - refusal_expected = False
      - refusal_correct = N/A (not meaningful)
    
    Returns:
    {
      "refusal_expected": bool,
      "refusal_correct": bool
    }
    """
    answer_lower = final_answer.lower()
    
    # Keywords that indicate refusal
    refusal_keywords = [
        "cannot answer",
        "cannot determine",
        "don't know",
        "do not know",
        "no data",
        "not defined",
        "doesn't exist",
        "not available",
        "out of scope",
        "not applicable",
        "unable to",
        "no information",
        "lack of data",
    ]
    
    has_refusal = any(keyword in answer_lower for keyword in refusal_keywords)
    
    if question_type == "nonsense_or_out_of_scope":
        return {
            "refusal_expected": True,
            "refusal_correct": has_refusal,
        }
    else:
        return {
            "refusal_expected": False,
            "refusal_correct": True,  # Not meaningful for factual questions
        }


def detect_hallucination_complete(
    question: str,
    question_type: str,
    final_answer: str,
    planet_name: str | None = None,
) -> Dict[str, Any]:
    """
    Complete hallucination detection pipeline.
    
    For planet_fact: extracts facts, compares with ground truth.
    For nonsense_or_out_of_scope: checks refusal behavior.
    
    Returns comprehensive result dict.
    """
    result: Dict[str, Any] = {
        "hallucination_detected": False,
        "mismatches": [],
        "refusal_expected": False,
        "refusal_correct": True,
        "is_correct": False,
    }
    
    if question_type == "planet_fact" and planet_name:
        # Extract facts from answer
        extracted = extract_planet_facts(final_answer, planet_name)
        
        # Get ground truth
        ground_truth = get_ground_truth_facts(planet_name)
        
        # Compare
        halluc_result = detect_hallucination_for_planet_fact(extracted, ground_truth)
        
        result["hallucination_detected"] = halluc_result.hallucination_detected
        result["mismatches"] = [
            {
                "field": m.field,
                "answer_value": m.answer_value,
                "true_value": m.true_value,
            }
            for m in halluc_result.mismatches
        ]
        result["is_correct"] = not halluc_result.hallucination_detected
        result["confidence"] = halluc_result.confidence
        
    elif question_type == "nonsense_or_out_of_scope":
        # Check refusal behavior
        refusal_eval = evaluate_refusal_behavior(question_type, final_answer)
        result["refusal_expected"] = refusal_eval["refusal_expected"]
        result["refusal_correct"] = refusal_eval["refusal_correct"]
        
        # Hallucination = expected refusal but didn't refuse (invented facts)
        result["hallucination_detected"] = (
            refusal_eval["refusal_expected"] and not refusal_eval["refusal_correct"]
        )
        result["is_correct"] = refusal_eval["refusal_correct"]
    
    return result

