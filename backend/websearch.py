from __future__ import annotations

from typing import List, Dict, Any

import requests

from . import config


class WebSearchError(RuntimeError):
    """Raised when the external web search API fails."""


def tavily_search(
    query: str,
    *,
    max_results: int | None = None,
    include_domains: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """Query the Tavily search API and return structured results."""

    payload = {
        "api_key": config.TAVILY_API_KEY,
        "query": query,
        "max_results": max_results or config.TAVILY_MAX_RESULTS,
        "search_depth": "advanced",
    }
    if include_domains:
        payload["include_domains"] = include_domains

    response = requests.post(config.TAVILY_ENDPOINT, json=payload, timeout=30)
    if response.status_code != 200:
        raise WebSearchError(response.text)

    data = response.json()
    return data.get("results", [])


def format_web_results(results: List[Dict[str, Any]]) -> List[str]:
    """Convert Tavily results into plain text snippets suitable for LLM context."""

    formatted = []
    for item in results:
        title = item.get("title", "Unknown")
        url = item.get("url", "")
        snippet = item.get("content") or item.get("snippet") or ""
        formatted.append(f"{title} ({url}): {snippet}")
    return formatted

