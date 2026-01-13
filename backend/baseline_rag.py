from __future__ import annotations

from typing import Any, Dict, List, Optional

import chromadb
import requests

from . import config
from .llm import call_groq_chat
from .retrieval import RetrievalChunk, retrieve_planet_facts
from .chroma_setup import init_chroma_client


def _build_prompt(question: str, contexts: List[str]) -> List[Dict[str, str]]:
    context_block = "\n\n".join(contexts) if contexts else "No context found."
    return [
        {
            "role": "system",
            "content": (
                "You are a planetary science assistant. Answer strictly using the provided "
                "context. If the answer is unknowable, explicitly say you cannot answer."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context_block}\n\nQuestion: {question}",
        },
    ]


def baseline_answer(
    question: str,
    *,
    n_results: int = 4,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    client: Optional[chromadb.PersistentClient] = None,
) -> Dict[str, Any]:
    client = client or init_chroma_client()
    retrieval = retrieve_planet_facts(question, n_results=n_results, client=client)
    messages = _build_prompt(question, retrieval.documents)
    answer = call_groq_chat(messages, api_key=api_key, model=model)
    return {
        "question": question,
        "answer": answer,
        "documents": retrieval.documents,
        "metadatas": retrieval.metadatas,
        "ids": retrieval.ids,
    }


class BaselineRAG:
    """Simple Groq + Chroma baseline RAG for experimental comparison."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        n_results: int = 4,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.n_results = n_results
        self._client: Optional[chromadb.PersistentClient] = None

    @property
    def client(self) -> chromadb.PersistentClient:
        if self._client is None:
            self._client = init_chroma_client()
        return self._client

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question using baseline RAG pipeline."""
        return baseline_answer(
            question,
            n_results=self.n_results,
            api_key=self.api_key,
            model=self.model,
            client=self.client,
        )


def run_cli_loop() -> None:
    print("Baseline RAG ready. Type 'exit' to quit.")
    client = init_chroma_client()
    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting baseline RAG loop.")
            break
        if not question or question.lower() in {"exit", "quit"}:
            break
        retrieval = retrieve_planet_facts(question, client=client)
        messages = _build_prompt(question, retrieval.documents)
        try:
            answer = call_groq_chat(messages)
        except requests.HTTPError as exc:
            print(f"Groq API error: {exc}")
            continue
        print("\nAnswer:\n", answer)
        print("\nTop contexts:")
        for idx, meta in enumerate(retrieval.metadatas, start=1):
            planet = meta.get("planet", "Unknown")
            print(f"{idx}. {planet}")
        print("-" * 60)


if __name__ == "__main__":
    run_cli_loop()

