"""Evaluate system accuracy and hallucination metrics from interactions.jsonl."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.logger import SUMMARY_LOG, INTERACTIONS_LOG


def load_interactions() -> List[Dict[str, Any]]:
    """Load all interactions from interactions.jsonl."""
    interactions = []
    if INTERACTIONS_LOG.exists():
        with INTERACTIONS_LOG.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        interactions.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return interactions


def filter_by_mode(interactions: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    """Filter interactions by mode."""
    return [i for i in interactions if i.get("mode") == mode]


def filter_by_question_type(
    interactions: List[Dict[str, Any]], 
    question_type: str
) -> List[Dict[str, Any]]:
    """Filter interactions by question type."""
    return [i for i in interactions if i.get("question_type") == question_type]


def compute_planet_fact_metrics(interactions: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute accuracy and hallucination rate for planet_fact questions."""
    if not interactions:
        return {"accuracy": 0.0, "hallucination_rate": 0.0}
    
    total = len(interactions)
    correct = sum(1 for i in interactions if i.get("is_correct", False))
    hallucinations = sum(1 for i in interactions if i.get("hallucination_detected", False))
    
    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "hallucination_rate": hallucinations / total if total > 0 else 0.0,
        "total": total,
    }


def compute_nonsense_metrics(interactions: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute refusal accuracy and overclaim rate for nonsense questions."""
    if not interactions:
        return {"refusal_accuracy": 0.0, "overclaim_rate": 0.0}
    
    # Only count interactions where refusal was expected
    expected_refusals = [i for i in interactions if i.get("refusal_expected", False)]
    
    if not expected_refusals:
        return {"refusal_accuracy": 0.0, "overclaim_rate": 0.0, "total": 0}
    
    total = len(expected_refusals)
    correct_refusals = sum(1 for i in expected_refusals if i.get("refusal_correct", False))
    overclaims = sum(1 for i in expected_refusals if i.get("hallucination_detected", False))
    
    return {
        "refusal_accuracy": correct_refusals / total if total > 0 else 0.0,
        "overclaim_rate": overclaims / total if total > 0 else 0.0,
        "total": total,
    }


def compute_metrics() -> Dict[str, Any]:
    """Compute comprehensive metrics for all modes and question types."""
    interactions = load_interactions()
    
    if not interactions:
        return {}
    
    modes = ["plain_llm", "basic_rag", "agentic_rag"]
    result: Dict[str, Any] = {}
    
    for mode in modes:
        mode_interactions = filter_by_mode(interactions, mode)
        if not mode_interactions:
            continue
        
        mode_result: Dict[str, Any] = {}
        
        # Planet fact metrics
        planet_fact_interactions = filter_by_question_type(mode_interactions, "planet_fact")
        if planet_fact_interactions:
            mode_result["planet_fact"] = compute_planet_fact_metrics(planet_fact_interactions)
        
        # Nonsense/out-of-scope metrics
        nonsense_interactions = filter_by_question_type(mode_interactions, "nonsense_or_out_of_scope")
        if nonsense_interactions:
            mode_result["nonsense_or_out_of_scope"] = compute_nonsense_metrics(nonsense_interactions)
        
        if mode_result:
            result[mode] = mode_result
    
    return result


def print_metrics_table(metrics: Dict[str, Any]) -> None:
    """Print a formatted table of metrics."""
    print("\n" + "=" * 80)
    print("EVALUATION METRICS")
    print("=" * 80)
    
    for mode, mode_data in metrics.items():
        print(f"\n{mode.upper().replace('_', ' ')}")
        print("-" * 80)
        
        if "planet_fact" in mode_data:
            pf = mode_data["planet_fact"]
            print(f"  Planet Facts (n={pf.get('total', 0)}):")
            print(f"    Accuracy:           {pf['accuracy']:.2%}")
            print(f"    Hallucination Rate:  {pf['hallucination_rate']:.2%}")
        
        if "nonsense_or_out_of_scope" in mode_data:
            ns = mode_data["nonsense_or_out_of_scope"]
            print(f"  Nonsense/Out-of-Scope (n={ns.get('total', 0)}):")
            print(f"    Refusal Accuracy:    {ns['refusal_accuracy']:.2%}")
            print(f"    Overclaim Rate:      {ns['overclaim_rate']:.2%}")
    
    print("\n" + "=" * 80)


def main() -> None:
    """Main evaluation script."""
    metrics = compute_metrics()
    
    if not metrics:
        print("No interactions found in interactions.jsonl.")
        print("Run batch evaluation first to generate data.")
        return
    
    print_metrics_table(metrics)
    
    # Write to summary file
    from backend.logger import write_summary
    write_summary(metrics)
    print(f"\nMetrics saved to {SUMMARY_LOG}")


if __name__ == "__main__":
    main()

