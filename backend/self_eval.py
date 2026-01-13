"""Self-evaluation confidence scoring."""

from __future__ import annotations

import json
import re
from typing import Dict, Any

from .llm import call_groq_chat


def self_evaluate_answer_confidence(
    question: str,
    answer: str,
    context_snippet: str = "",
) -> Dict[str, Any]:
    """
    Ask the Groq model to self-evaluate its answer.
    
    Returns:
    {
      "confidence": float (0-1),
      "notes": str
    }
    """
    prompt = (
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        f"Context used: {context_snippet[:500] if context_snippet else 'None'}\n\n"
        "Evaluate your answer:\n"
        "1. List the factual claims made in the answer.\n"
        "2. Assign a confidence score from 0.0 to 1.0 based on how well-supported "
        "these claims are by the context.\n"
        "3. Provide brief notes explaining your confidence level.\n\n"
        "Return ONLY a JSON object with keys: 'confidence' (float) and 'notes' (string).\n"
        "Example: {\"confidence\": 0.85, \"notes\": \"The answer is well-supported by context.\"}"
    )
    
    try:
        response = call_groq_chat(
            messages=[
                {"role": "system", "content": "You are a self-evaluation system. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
        )
        
        # Try to extract JSON
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            confidence = float(result.get("confidence", 0.5))
            # Clamp to [0, 1]
            confidence = max(0.0, min(1.0, confidence))
            return {
                "confidence": confidence,
                "notes": result.get("notes", ""),
            }
    except Exception:
        pass
    
    # Fallback
    return {
        "confidence": 0.5,
        "notes": "Self-evaluation unavailable",
    }

