from __future__ import annotations

import argparse
from typing import List

from backend.logger import write_summary
from backend.pipeline import AgenticPipeline, PipelineResult

DEFAULT_QUESTIONS = [
    "How many moons does Mars have?",
    "What is the diameter of Neptune?",
    "List the mean temperature on Venus.",
    "How massive is Jupiter compared to Earth?",
]


def print_result(result: PipelineResult) -> None:
    print("\n" + "=" * 70)
    print(f"QUESTION: {result.question}")
    if result.used_memory:
        memory = result.memory_hits[0]
        print("Answered from memory.")
        print(f"Final answer: {memory.answer}")
        return

    print(f"Baseline answer: {result.baseline['answer']}")
    if result.planner:
        print(f"Planner strategy: {result.planner.strategy} (confidence {result.planner.confidence:.2f})")
    print(f"Final answer: {result.final_answer}")
    if result.verification:
        status = "⚠️" if result.verification.hallucination_detected else "✅"
        print(f"{status} Hallucination detected: {result.verification.hallucination_detected}")
        print(f"Verifier confidence: {result.verification.confidence:.2f}")
        if result.verification.notes:
            print("Notes:", result.verification.notes)


def summarize(results: List[PipelineResult]) -> None:
    total = len(results)
    hallucinations = 0
    confidences = []
    for res in results:
        if res.verification:
            if res.verification.hallucination_detected:
                hallucinations += 1
            confidences.append(res.verification.confidence)
        else:
            confidences.append(1.0)

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    summary = {
        "total_questions": total,
        "hallucinations_detected": hallucinations,
        "avg_confidence": round(avg_conf, 3),
    }
    print("\n" + "=" * 70)
    print("SUMMARY METRICS")
    print("=" * 70)
    for key, value in summary.items():
        print(f"{key}: {value}")
    write_summary(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate agentic pipeline.")
    parser.add_argument(
        "-q",
        "--question",
        action="append",
        help="Provide one or more questions to evaluate.",
    )
    args = parser.parse_args()

    questions = args.question or DEFAULT_QUESTIONS

    pipeline = AgenticPipeline()
    results: List[PipelineResult] = []
    for question in questions:
        result = pipeline.run(question)
        results.append(result)
        print_result(result)

    summarize(results)


if __name__ == "__main__":
    main()

