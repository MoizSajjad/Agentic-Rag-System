from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Sequence

from sentence_transformers import SentenceTransformer

from . import config


@lru_cache(maxsize=1)
def _get_model(model_name: str | None = None) -> SentenceTransformer:
    """Lazy-load the SentenceTransformer model once per process."""

    name = model_name or config.EMBED_MODEL_NAME
    return SentenceTransformer(name)


def embed_texts(
    texts: Sequence[str],
    *,
    model_name: str | None = None,
) -> List[List[float]]:
    """Generate embeddings for a batch of texts."""

    if not texts:
        return []

    model = _get_model(model_name)
    embeddings = model.encode(
        texts,
        convert_to_numpy=False,
        show_progress_bar=False,
    )
    if hasattr(embeddings, "tolist"):
        embeddings = embeddings.tolist()

    normalized: List[List[float]] = []
    for emb in embeddings:
        if hasattr(emb, "tolist"):
            normalized.append(emb.tolist())
        elif isinstance(emb, (list, tuple)):
            normalized.append(list(emb))
        else:
            raise ValueError("Embedding must be list-like.")
    return normalized

