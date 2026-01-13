from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from . import config

LOG_DIR = config.PROJECT_ROOT / "logs"
INTERACTIONS_LOG = LOG_DIR / "interactions.jsonl"
SUMMARY_LOG = LOG_DIR / "summary_metrics.json"


def _ensure_logs_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_interaction(
    question: str,
    initial_answer: str,
    final_answer: str,
    mode: str,  # "plain_llm" | "basic_rag" | "agentic_rag"
    question_type: str,  # "planet_fact" | "nonsense_or_out_of_scope"
    is_answerable_from_local: bool,
    hallucination_detected: bool,
    is_correct: bool,
    refusal_expected: bool = False,
    refusal_correct: bool = True,
    mismatches: list[Dict[str, Any]] | None = None,
    self_eval_confidence: float | None = None,
    trusted_facts: Dict[str, Any] | None = None,
    sources_used: list[str] | None = None,
    timestamp: float | None = None,
    **kwargs: Any,
) -> None:
    """
    Log a complete interaction with all new fields.
    
    This function ensures backward compatibility - older entries may miss some fields.
    """
    import time
    
    record: Dict[str, Any] = {
        "question": question,
        "initial_answer": initial_answer,
        "final_answer": final_answer,
        "mode": mode,
        "question_type": question_type,
        "is_answerable_from_local": is_answerable_from_local,
        "hallucination_detected": hallucination_detected,
        "is_correct": is_correct,
        "refusal_expected": refusal_expected,
        "refusal_correct": refusal_correct,
        "mismatches": mismatches or [],
        "timestamp": timestamp or time.time(),
    }
    
    if self_eval_confidence is not None:
        record["self_eval_confidence"] = self_eval_confidence
    
    if trusted_facts:
        # Normalize trusted_facts structure
        if isinstance(trusted_facts, dict):
            record["trusted_facts"] = {
                "local": trusted_facts.get("local", []),
                "web": trusted_facts.get("web", []),
            }
        else:
            record["trusted_facts"] = {"local": [], "web": []}
    else:
        record["trusted_facts"] = {"local": [], "web": []}
    
    if sources_used:
        record["sources_used"] = sources_used
    else:
        record["sources_used"] = []
    
    # Add any extra kwargs
    record.update(kwargs)
    
    _ensure_logs_dir()
    with INTERACTIONS_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_interaction(record: Dict[str, Any]) -> None:
    """
    Append a single JSON record to interactions.jsonl.
    
    For backward compatibility - use log_interaction() for new code.
    """
    _ensure_logs_dir()
    with INTERACTIONS_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary(summary: Dict[str, Any]) -> None:
    """Overwrite the summary metrics JSON."""

    _ensure_logs_dir()
    with SUMMARY_LOG.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

