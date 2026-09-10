"""Groq client with retries, backoff, and simple rate limiting for free tier."""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv()

try:
    from groq import Groq
except ImportError:  # pragma: no cover
    Groq = None  # type: ignore


CLASSIFY_MODEL = os.getenv("GROQ_CLASSIFY_MODEL", "llama-3.1-8b-instant")
DRAFT_MODEL = os.getenv("GROQ_DRAFT_MODEL", "llama-3.3-70b-versatile")
JUDGE_MODEL = os.getenv("GROQ_JUDGE_MODEL", "llama-3.3-70b-versatile")

# Stay under free-tier ~30 RPM
MIN_INTERVAL_SEC = float(os.getenv("GROQ_MIN_INTERVAL_SEC", "2.2"))


class GroqClient:
    def __init__(self) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key, "
                "or use --use-cache evaluation paths that do not call Groq."
            )
        if Groq is None:
            raise RuntimeError("groq package is not installed. pip install groq")
        self.client = Groq(api_key=api_key)
        self._last_call = 0.0

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < MIN_INTERVAL_SEC:
            time.sleep(MIN_INTERVAL_SEC - elapsed)

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        retries: int = 5,
    ) -> str:
        last_err: Exception | None = None
        for attempt in range(retries):
            self._throttle()
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                self._last_call = time.time()
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                msg = str(e).lower()
                wait = 2 ** attempt
                if "429" in msg or "rate" in msg:
                    wait = max(wait, 8 + attempt * 4)
                time.sleep(wait)
        raise RuntimeError(f"Groq request failed after {retries} retries: {last_err}")


def extract_json(text: str) -> dict[str, Any]:
    """Parse JSON from model output, tolerating markdown fences."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    return json.loads(text)