"""Minimal OpenAI-compatible chat client (standard library only).

Used to generate and check training prompts through a local free-model gateway such as freellmapi.
The API key is read from the environment or from ~/.packlight/gateway_key. It is never stored in
the repository and never printed.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

KEY_ENV = "PACKLIGHT_LLM_KEY"
URL_ENV = "PACKLIGHT_LLM_URL"
KEY_FILE = Path.home() / ".packlight" / "gateway_key"
DEFAULT_URL = "http://127.0.0.1:3001/v1"
DEFAULT_MODEL = "auto:smart"
REQUEST_TIMEOUT_SECONDS = 180
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 3


class MissingKeyError(RuntimeError):
    pass


def load_key() -> str:
    key = os.environ.get(KEY_ENV, "").strip()
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text(encoding="utf-8").strip()
    if not key:
        raise MissingKeyError(f"Set {KEY_ENV} or put the key in {KEY_FILE}")
    return key


class Gateway:
    def __init__(self, model: str = DEFAULT_MODEL, base_url: str | None = None, key: str | None = None):
        self.model = model
        self.base_url = (base_url or os.environ.get(URL_ENV) or DEFAULT_URL).rstrip("/")
        self._key = key or load_key()

    def chat(self, messages: list[dict], temperature: float = 0.8, max_tokens: int = 2500) -> str:
        body = json.dumps({
            "model": self.model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens,
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"},
        )
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return payload["choices"][0]["message"]["content"] or ""
            except urllib.error.HTTPError as error:
                last_error = error
                if error.code not in RETRYABLE_STATUS:
                    raise
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as error:
                last_error = error
            time.sleep(BACKOFF_BASE_SECONDS * (attempt + 1))
        raise RuntimeError(f"gateway failed after {MAX_RETRIES} attempts: {last_error}")
