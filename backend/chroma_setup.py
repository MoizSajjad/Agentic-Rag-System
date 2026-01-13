from __future__ import annotations

import os
from typing import Callable, Dict, Iterable, List

import chromadb
from chromadb.config import Settings
import pandas as pd

from . import config
from .embeddings import embed_texts


def _normalize_key(raw_key: str) -> str:
    """Convert CSV parameter names into snake_case feature keys."""

    sanitized = (
        raw_key.strip()
        .lower()
        .replace("?", "")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )
    return "_".join(filter(None, sanitized.split("_")))


def load_planet_dataframe(csv_path: str | None = None) -> pd.DataFrame:
    """Load the NASA CSV and pivot it into one record per planet."""

    path = csv_path or config.PLANETS_CSV_PATH
    wide_df = pd.read_csv(path)
    if wide_df.shape[1] < 3:
        raise ValueError("CSV must include Parameter, Unit, and planet columns.")

    parameter_col, unit_col = wide_df.columns[:2]
    planet_columns = wide_df.columns[2:]

    records: List[Dict[str, str]] = []
    for planet in planet_columns:
        record: Dict[str, str] = {"planet": planet}
        for _, row in wide_df.iterrows():
            parameter_name = str(row[parameter_col])
            unit_value = str(row[unit_col]) if not pd.isna(row[unit_col]) else ""
            measurement = row[planet]

            key = _normalize_key(parameter_name)
            record[key] = "" if pd.isna(measurement) else str(measurement)
            if unit_value and key not in ("planet", ""):
                record[f"{key}_unit"] = unit_value
        records.append(record)

    tidy_df = pd.DataFrame(records)
    return tidy_df.fillna("")


def init_chroma_client(
    persist_directory: str | None = None,
) -> chromadb.PersistentClient:
    """Initialize a persistent Chroma client."""

    config.ensure_directories()
    directory = persist_directory or str(config.CHROMA_PERSIST_DIR)
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "0")
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")
    try:  # Make posthog telemetry a silent no-op
        import posthog  # type: ignore

        posthog.disabled = True
        posthog.capture = lambda *args, **kwargs: None  # type: ignore
    except Exception:
        pass
    settings = Settings(allow_reset=True, anonymized_telemetry=False)
    return chromadb.PersistentClient(path=directory, settings=settings)


def get_or_create_collection(
    client: chromadb.api.client.ClientAPI,
    name: str,
    metadata: dict | None = None,
):
    """Helper to create a collection if it does not yet exist."""

    try:
        return client.get_collection(name)
    except (chromadb.errors.InvalidCollectionException, ValueError):
        return client.create_collection(name=name, metadata=metadata)


def _format_planet_document(row: Dict[str, str]) -> str:
    """Convert a row into a concise natural language description."""

    attrs = ", ".join(
        f"{key.replace('_', ' ')}: {value}" for key, value in row.items() if value != ""
    )
    return f"Planet record — {attrs}"


def build_planet_payloads(df: pd.DataFrame) -> Dict[str, List]:
    """Prepare documents, metadata, and ids for Chroma ingestion."""

    records = df.to_dict(orient="records")
    documents = [_format_planet_document(rec) for rec in records]
    metadatas = records
    ids = [f"planet-{idx}" for idx in range(len(records))]
    return {"ids": ids, "documents": documents, "metadatas": metadatas}


def embed_and_upsert_planets(
    client: chromadb.api.client.ClientAPI,
    df: pd.DataFrame,
    *,
    embed_fn: Callable[[Iterable[str]], List[List[float]]] = embed_texts,
) -> None:
    """Embed the dataset and upsert into the planets_facts collection."""

    payloads = build_planet_payloads(df)
    embeddings = embed_fn(payloads["documents"])
    collection = get_or_create_collection(
        client,
        config.CHROMA_COLLECTIONS.planets_facts,
        metadata={"source": "local_csv"},
    )
    collection.upsert(
        ids=payloads["ids"],
        documents=payloads["documents"],
        metadatas=payloads["metadatas"],
        embeddings=embeddings,
    )


def ensure_corrections_collection(
    client: chromadb.api.client.ClientAPI,
) -> None:
    """Create the corrections memory collection if needed."""

    get_or_create_collection(
        client,
        config.CHROMA_COLLECTIONS.corrections_memory,
        metadata={"source": "self_corrections"},
    )

