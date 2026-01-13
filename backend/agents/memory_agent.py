from __future__ import annotations

import time
import uuid
import json
from dataclasses import dataclass
from typing import Any, Dict, List

import chromadb

from .. import config
from ..chroma_setup import ensure_corrections_collection, init_chroma_client
from ..embeddings import embed_texts
from ..logger import append_interaction


@dataclass
class MemoryRecord:
    question: str
    answer: str
    trusted_facts: Dict[str, Any]
    sources: List[str]
    timestamp: float
    metadata: Dict[str, Any] = None


class MemoryAgent:
    """Persists corrected answers and surfaces them for future queries."""

    def __init__(self, *, match_threshold: float = 0.3) -> None:
        self.client = init_chroma_client()
        ensure_corrections_collection(self.client)
        self.collection = self.client.get_collection(
            config.CHROMA_COLLECTIONS.corrections_memory
        )
        self.match_threshold = match_threshold

    def search(self, question: str, n_results: int = 2) -> List[MemoryRecord]:
        if not question.strip():
            return []
        response = self.collection.query(
            query_texts=[question],
            n_results=n_results,
            include=["metadatas", "distances"],
        )
        records: List[MemoryRecord] = []
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]
        for metadata, distance in zip(metadatas, distances):
            if distance is None or distance > self.match_threshold:
                continue
            trusted_facts = metadata.get("trusted_facts")
            if isinstance(trusted_facts, str):
                try:
                    trusted_facts = json.loads(trusted_facts)
                except json.JSONDecodeError:
                    trusted_facts = {"raw": trusted_facts}
            sources = metadata.get("sources", [])
            if isinstance(sources, str):
                try:
                    sources = json.loads(sources)
                except json.JSONDecodeError:
                    sources = [sources]
            records.append(
                MemoryRecord(
                    question=metadata.get("question", ""),
                    answer=metadata.get("answer", ""),
                    trusted_facts=trusted_facts or {},
                    sources=sources,
                    timestamp=metadata.get("timestamp", 0.0),
                    metadata=metadata,
                )
            )
        return records

    def save(
        self,
        question: str,
        initial_answer: str,
        final_answer: str,
        trusted_facts: Dict[str, Any],
        hallucination_detected: bool,
        sources: List[str],
    ) -> None:
        embedding = embed_texts([question])[0]
        metadata = {
            "question": question,
            "answer": final_answer,
            "initial_answer": initial_answer,
            "trusted_facts": json.dumps(trusted_facts, ensure_ascii=False),
            "sources": json.dumps(sources, ensure_ascii=False),
            "timestamp": time.time(),
            "hallucination_detected": hallucination_detected,
        }
        document = (
            f"Question: {question}\nAnswer: {final_answer}\nFacts: {trusted_facts}"
        )
        record_id = f"memory-{uuid.uuid4()}"
        self.collection.upsert(
            ids=[record_id],
            documents=[document],
            embeddings=[embedding],
            metadatas=[metadata],
        )
        append_interaction(
            {
                "question": question,
                "initial_answer": initial_answer,
                "final_answer": final_answer,
                "trusted_facts": trusted_facts,
                "hallucination_detected": hallucination_detected,
                "sources_used": sources,
                "timestamp": metadata["timestamp"],
            }
        )

