from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PLANETS_CSV_PATH = DATA_DIR / "planets_full_clean.csv"
CHROMA_PERSIST_DIR = PROJECT_ROOT / "chroma_store"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY environment variable is required. "
        "Please set it using: export GROQ_API_KEY='your-api-key'"
    )
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# Tavily API key is optional - web search will be skipped if not provided
TAVILY_ENDPOINT = "https://api.tavily.com/search"
TAVILY_MAX_RESULTS = 5


@dataclass(frozen=True)
class ChromaCollections:
    """Canonical collection names used across the system."""

    planets_facts: str = "planets_facts"
    corrections_memory: str = "corrections_memory"


CHROMA_COLLECTIONS = ChromaCollections()


def ensure_directories() -> None:
    """Make sure commonly used directories exist before runtime."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

