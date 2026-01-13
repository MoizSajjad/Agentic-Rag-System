"""Run batch evaluation across all modes and test questions."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.pipeline import AgenticPipeline
from backend.baseline_rag import BaselineRAG
from backend.llm import call_groq_chat
from backend.classifier import classify_question
from backend.hallucination_detection import detect_hallucination_complete
from backend.self_eval import self_evaluate_answer_confidence
from backend.logger import log_interaction
from backend.retrieval import retrieve_planet_facts
from backend.chroma_setup import init_chroma_client


def load_test_questions() -> Dict[str, List[str]]:
    """Load test questions from JSON file."""
    test_file = Path(__file__).parent / "test_questions.json"
    with test_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_plain_llm(question: str) -> Dict[str, Any]:
    """Run plain LLM mode."""
    print(f"  [Plain LLM] Processing: {question[:60]}...")
    
    # Classify question
    classification = classify_question(question)
    
    # Get answer
    answer = call_groq_chat(
        messages=[
            {"role": "system", "content": "You are a helpful assistant about planetary science."},
            {"role": "user", "content": question}
        ],
        temperature=0.2,
    )
    
    # Self-evaluation
    self_eval = self_evaluate_answer_confidence(question, answer)
    
    # Hallucination detection
    planet_name = classification.detected_planet
    halluc_result = detect_hallucination_complete(
        question=question,
        question_type=classification.question_type,
        final_answer=answer,
        planet_name=planet_name,
    )
    
    # Get trusted facts (minimal for plain LLM)
    trusted_facts = {"local": [], "web": []}
    
    # Log
    log_interaction(
        question=question,
        initial_answer=answer,
        final_answer=answer,
        mode="plain_llm",
        question_type=classification.question_type,
        is_answerable_from_local=classification.is_answerable_from_local,
        hallucination_detected=halluc_result["hallucination_detected"],
        is_correct=halluc_result["is_correct"],
        refusal_expected=halluc_result.get("refusal_expected", False),
        refusal_correct=halluc_result.get("refusal_correct", True),
        mismatches=halluc_result.get("mismatches", []),
        self_eval_confidence=self_eval["confidence"],
        trusted_facts=trusted_facts,
        sources_used=[],
    )
    
    return {
        "question": question,
        "answer": answer,
        "classification": classification,
        "hallucination": halluc_result,
        "self_eval": self_eval,
    }


def run_basic_rag(question: str) -> Dict[str, Any]:
    """Run basic RAG mode."""
    print(f"  [Basic RAG] Processing: {question[:60]}...")
    
    # Classify question
    classification = classify_question(question)
    
    # Get answer
    rag = BaselineRAG()
    result = rag.answer_question(question)
    answer = result["answer"]
    
    # Self-evaluation with context
    context = "\n\n".join(result["documents"][:2])
    self_eval = self_evaluate_answer_confidence(question, answer, context)
    
    # Hallucination detection
    planet_name = classification.detected_planet
    halluc_result = detect_hallucination_complete(
        question=question,
        question_type=classification.question_type,
        final_answer=answer,
        planet_name=planet_name,
    )
    
    # Format trusted facts
    trusted_facts = {
        "local": [
            {"planet": meta.get("planet", "Unknown"), "content": doc}
            for doc, meta in zip(result["documents"], result["metadatas"])
        ],
        "web": [],
    }
    
    sources = [meta.get("planet", "local") for meta in result["metadatas"]]
    
    # Log
    log_interaction(
        question=question,
        initial_answer=answer,
        final_answer=answer,
        mode="basic_rag",
        question_type=classification.question_type,
        is_answerable_from_local=classification.is_answerable_from_local,
        hallucination_detected=halluc_result["hallucination_detected"],
        is_correct=halluc_result["is_correct"],
        refusal_expected=halluc_result.get("refusal_expected", False),
        refusal_correct=halluc_result.get("refusal_correct", True),
        mismatches=halluc_result.get("mismatches", []),
        self_eval_confidence=self_eval["confidence"],
        trusted_facts=trusted_facts,
        sources_used=sources,
    )
    
    return {
        "question": question,
        "answer": answer,
        "classification": classification,
        "hallucination": halluc_result,
        "self_eval": self_eval,
    }


def run_agentic_rag(question: str) -> Dict[str, Any]:
    """Run agentic RAG mode."""
    print(f"  [Agentic RAG] Processing: {question[:60]}...")
    
    # Classify question
    classification = classify_question(question)
    
    # Run pipeline
    pipeline = AgenticPipeline()
    result = pipeline.run(question)
    
    answer = result.final_answer
    baseline_answer = result.baseline["answer"] if result.baseline else answer
    
    # Self-evaluation with context
    context = ""
    if result.retrieval:
        context = "\n\n".join(result.retrieval.local_results.documents[:2])
    self_eval = self_evaluate_answer_confidence(question, answer, context)
    
    # Hallucination detection
    planet_name = classification.detected_planet
    halluc_result = detect_hallucination_complete(
        question=question,
        question_type=classification.question_type,
        final_answer=answer,
        planet_name=planet_name,
    )
    
    # Format trusted facts
    trusted_facts = {
        "local": [
            {"planet": meta.get("planet", "Unknown"), "content": doc}
            for doc, meta in zip(
                result.retrieval.local_results.documents if result.retrieval else [],
                result.retrieval.local_results.metadatas if result.retrieval else [],
            )
        ],
        "web": [
            {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
            for r in (result.retrieval.web_results if result.retrieval and result.retrieval.web_results else [])
        ],
    }
    
    sources = []
    if result.retrieval:
        sources = [meta.get("planet", "local") for meta in result.retrieval.local_results.metadatas]
        if result.retrieval.used_web:
            sources.append("web:tavily")
    
    # Log
    log_interaction(
        question=question,
        initial_answer=baseline_answer,
        final_answer=answer,
        mode="agentic_rag",
        question_type=classification.question_type,
        is_answerable_from_local=classification.is_answerable_from_local,
        hallucination_detected=halluc_result["hallucination_detected"],
        is_correct=halluc_result["is_correct"],
        refusal_expected=halluc_result.get("refusal_expected", False),
        refusal_correct=halluc_result.get("refusal_correct", True),
        mismatches=halluc_result.get("mismatches", []),
        self_eval_confidence=self_eval["confidence"],
        trusted_facts=trusted_facts,
        sources_used=sources,
    )
    
    return {
        "question": question,
        "answer": answer,
        "classification": classification,
        "hallucination": halluc_result,
        "self_eval": self_eval,
    }


def main() -> None:
    """Run batch evaluation for all modes and questions."""
    print("=" * 80)
    print("BATCH EVALUATION")
    print("=" * 80)
    
    test_questions = load_test_questions()
    modes = ["plain_llm", "basic_rag", "agentic_rag"]
    
    total_questions = (
        len(test_questions["planet_fact"]) + 
        len(test_questions["nonsense_or_out_of_scope"])
    )
    total_runs = total_questions * len(modes)
    
    print(f"\nTotal questions: {total_questions}")
    print(f"Total runs: {total_runs} (across {len(modes)} modes)")
    print("\nStarting evaluation...\n")
    
    for mode in modes:
        print(f"\n{'=' * 80}")
        print(f"MODE: {mode.upper().replace('_', ' ')}")
        print("=" * 80)
        
        # Process planet facts
        print("\n[Planet Facts]")
        for question in test_questions["planet_fact"]:
            try:
                if mode == "plain_llm":
                    run_plain_llm(question)
                elif mode == "basic_rag":
                    run_basic_rag(question)
                elif mode == "agentic_rag":
                    run_agentic_rag(question)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()
        
        # Process nonsense questions
        print("\n[Nonsense/Out-of-Scope]")
        for question in test_questions["nonsense_or_out_of_scope"]:
            try:
                if mode == "plain_llm":
                    run_plain_llm(question)
                elif mode == "basic_rag":
                    run_basic_rag(question)
                elif mode == "agentic_rag":
                    run_agentic_rag(question)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("BATCH EVALUATION COMPLETE")
    print("=" * 80)
    print("\nRun evaluation/evaluate_system.py to see metrics.")


if __name__ == "__main__":
    main()

