from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..retrieval import RetrievalChunk, retrieve_planet_facts
from ..websearch import format_web_results, tavily_search, WebSearchError

CELESTIAL_BODIES = {
    "mercury",
    "venus",
    "earth",
    "moon",
    "mars",
    "jupiter",
    "saturn",
    "uranus",
    "neptune",
    "pluto",
    "haumea",
    "ceres",
    "eris",
    "makemake",
}

SPACE_CONTEXT_TERMS = {
    "planet",
    "dwarf",
    "ring",
    "moon",
    "space",
    "nasa",
    "astronomy",
    "orbital",
    "km",
    "gravity",
    "temperature",
    "surface",
}

DISALLOWED_TERMS = {
    "florida",
    "city",
    "town",
    "population",
    "county",
    "residents",
    "stock",
    "shares",
    "nasdaq",
    "inc.",
    "company",
    "finance",
}


def _extract_target(question: str) -> Optional[str]:
    lower_q = question.lower()
    for body in CELESTIAL_BODIES:
        if body in lower_q:
            return body
    return None


def _filter_web_results(
    target: Optional[str],
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not results:
        return []
    filtered: List[Dict[str, Any]] = []
    for item in results:
        text = f"{item.get('title', '')} {item.get('content') or item.get('snippet','')}".lower()
        if target and target not in text:
            continue
        if any(term in text for term in DISALLOWED_TERMS):
            continue
        if not any(term in text for term in SPACE_CONTEXT_TERMS):
            continue
        filtered.append(item)
    return filtered if filtered else []


@dataclass
class RetrieverOutput:
    local_results: RetrievalChunk
    web_results: List[Dict[str, Any]]
    web_snippets: List[str]
    used_web: bool


class RetrieverAgent:
    """Fetches local Chroma facts and optional web search evidence."""

    def __init__(self, *, n_local_results: int = 4) -> None:
        self.n_local_results = n_local_results

    def retrieve(self, question: str, *, use_web: bool) -> RetrieverOutput:
        local_results = retrieve_planet_facts(question, n_results=self.n_local_results)
        web_raw: List[Dict[str, Any]] = []
        web_snippets: List[str] = []
        if use_web:
            try:
                raw_results = tavily_search(question)
                target = _extract_target(question)
                web_raw = _filter_web_results(target, raw_results)
                web_snippets = format_web_results(web_raw)
            except WebSearchError as exc:
                web_snippets = [f"Web search error: {exc}"]
        return RetrieverOutput(
            local_results=local_results,
            web_results=web_raw,
            web_snippets=web_snippets,
            used_web=use_web and bool(web_snippets),
        )

