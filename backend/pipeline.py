from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import chromadb

from .baseline_rag import baseline_answer
from .chroma_setup import init_chroma_client
from .agents.planner_agent import PlannerAgent, PlannerDecision
from .agents.retriever_agent import RetrieverAgent, RetrieverOutput
from .agents.verifier_agent import VerifierAgent, VerificationResult
from .agents.memory_agent import MemoryAgent, MemoryRecord
from .classifier import QuestionClassification, classify_question
from .hallucination_detection import detect_hallucination_complete
from .self_eval import self_evaluate_answer_confidence
from .logger import log_interaction


@dataclass
class PipelineResult:
    question: str
    final_answer: str
    baseline: Optional[Dict[str, Any]]
    planner: Optional[PlannerDecision]
    retrieval: Optional[RetrieverOutput]
    verification: Optional[VerificationResult]
    memory_hits: List[MemoryRecord]
    used_memory: bool
    classification: Optional[QuestionClassification] = None
    hallucination_result: Optional[Dict[str, Any]] = None
    self_eval: Optional[Dict[str, Any]] = None


class AgenticPipeline:
    """End-to-end pipeline orchestrating baseline + multi-agent refinement."""

    def __init__(
        self,
        *,
        planner: Optional[PlannerAgent] = None,
        retriever: Optional[RetrieverAgent] = None,
        verifier: Optional[VerifierAgent] = None,
        memory: Optional[MemoryAgent] = None,
        chroma_client: Optional[chromadb.PersistentClient] = None,
    ) -> None:
        self.planner = planner or PlannerAgent()
        self.retriever = retriever or RetrieverAgent()
        self.verifier = verifier or VerifierAgent()
        self.memory = memory or MemoryAgent()
        self._client = chroma_client or init_chroma_client()

    @property
    def client(self) -> chromadb.PersistentClient:
        return self._client

    def run(self, question: str) -> PipelineResult:
        """Execute the pipeline for a single user question."""

        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        memory_hits = self.memory.search(question)
        # Only use memory if the cached answer was NOT a hallucination
        if memory_hits:
            top_hit = memory_hits[0]
            # Check if the cached answer was a hallucination
            if top_hit.metadata:
                hallucination = top_hit.metadata.get("hallucination_detected", False)
                # Convert string "True"/"False" to bool if needed
                if isinstance(hallucination, str):
                    hallucination = hallucination.lower() == "true"
                if not hallucination:
                    return PipelineResult(
                        question=question,
                        final_answer=top_hit.answer,
                        baseline=None,
                        planner=None,
                        retrieval=None,
                        verification=None,
                        memory_hits=memory_hits,
                        used_memory=True,
                    )
            else:
                # If metadata not available, use it anyway (backward compatibility)
                return PipelineResult(
                    question=question,
                    final_answer=top_hit.answer,
                    baseline=None,
                    planner=None,
                    retrieval=None,
                    verification=None,
                    memory_hits=memory_hits,
                    used_memory=True,
                )

        baseline = baseline_answer(question, client=self.client)
        context_excerpt = "\n\n".join(baseline["documents"][:2])
        planner_decision = self.planner.plan(
            question=question,
            baseline_answer=baseline["answer"],
            context_excerpt=context_excerpt,
        )
        use_web = planner_decision.strategy == "local_and_web"
        retrieval = self.retriever.retrieve(question, use_web=use_web)
        verification = self.verifier.verify(
            question=question,
            baseline_answer=baseline["answer"],
            retrieval=retrieval,
        )

        sources = [
            meta.get("planet", meta.get("source", "local"))
            for meta in retrieval.local_results.metadatas
        ]
        if retrieval.used_web:
            sources.append("web:tavily")

        self.memory.save(
            question=question,
            initial_answer=baseline["answer"],
            final_answer=verification.final_answer,
            trusted_facts=verification.trusted_facts,
            hallucination_detected=verification.hallucination_detected,
            sources=sources,
        )

        return PipelineResult(
            question=question,
            final_answer=verification.final_answer,
            baseline=baseline,
            planner=planner_decision,
            retrieval=retrieval,
            verification=verification,
            memory_hits=[],
            used_memory=False,
        )
    
    def run_with_evaluation(self, question: str, mode: str = "agentic_rag") -> PipelineResult:
        """
        Enhanced run method with classification, hallucination detection, and logging.
        
        This method extends the basic run() method with:
        - Question classification
        - Enhanced hallucination detection
        - Self-evaluation
        - Comprehensive logging
        """
        # Classify question
        classification = classify_question(question)
        
        # Run the standard pipeline
        result = self.run(question)
        
        # Add classification to result
        result.classification = classification
        
        # Enhanced hallucination detection
        planet_name = classification.detected_planet
        halluc_result = detect_hallucination_complete(
            question=question,
            question_type=classification.question_type,
            final_answer=result.final_answer,
            planet_name=planet_name,
        )
        result.hallucination_result = halluc_result
        
        # Self-evaluation
        context = ""
        if result.retrieval:
            context = "\n\n".join(result.retrieval.local_results.documents[:2])
        self_eval = self_evaluate_answer_confidence(question, result.final_answer, context)
        result.self_eval = self_eval
        
        # Format trusted facts
        trusted_facts = {
            "local": [],
            "web": [],
        }
        
        if result.retrieval:
            trusted_facts["local"] = [
                {"planet": meta.get("planet", "Unknown"), "content": doc}
                for doc, meta in zip(
                    result.retrieval.local_results.documents,
                    result.retrieval.local_results.metadatas,
                )
            ]
            if result.retrieval.web_results:
                trusted_facts["web"] = [
                    {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
                    for r in result.retrieval.web_results
                ]
        
        sources = []
        if result.retrieval:
            sources = [meta.get("planet", "local") for meta in result.retrieval.local_results.metadatas]
            if result.retrieval.used_web:
                sources.append("web:tavily")
        
        # Log with enhanced schema
        initial_answer = result.baseline["answer"] if result.baseline else result.final_answer
        log_interaction(
            question=question,
            initial_answer=initial_answer,
            final_answer=result.final_answer,
            mode=mode,
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
        
        return result

