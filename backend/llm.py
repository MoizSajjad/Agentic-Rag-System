from __future__ import annotations

import random
import time
from typing import List, Dict, Optional

import requests

from . import config


def call_groq_chat(
    messages: List[Dict[str, str]],
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
    timeout: int = 45,
    max_retries: int = 5,
    retry_delay: float = 2.0,
) -> str:
    """Generic Groq chat completion helper."""

    api_key = api_key or config.GROQ_API_KEY
    model = model or config.GROQ_MODEL_NAME
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": messages,
    }
    attempt = 0
    while True:
        response = requests.post(
            config.GROQ_ENDPOINT, headers=headers, json=payload, timeout=timeout
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as err:
            attempt += 1
            if (
                response.status_code == 429
                and attempt <= max_retries
            ):
                jitter = random.uniform(0, retry_delay)
                sleep_for = retry_delay * attempt + jitter
                time.sleep(sleep_for)
                continue
            raise
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

