from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import chromadb

from . import config
from .chroma_setup import init_chroma_client


@dataclass
class RetrievalChunk:
    documents: List[str]
    metadatas: List[Dict[str, Any]]
    ids: List[str]


def retrieve_planet_facts(
    question: str,
    *,
    n_results: int = 4,
    client: Optional[chromadb.PersistentClient] = None,
) -> RetrievalChunk:
    client = client or init_chroma_client()
    collection = client.get_collection(config.CHROMA_COLLECTIONS.planets_facts)
    response = collection.query(query_texts=[question], n_results=n_results)
    return RetrievalChunk(
        documents=response["documents"][0],
        metadatas=response["metadatas"][0],
        ids=response["ids"][0],
    )

