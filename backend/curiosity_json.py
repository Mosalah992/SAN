"""
curiosity_json.py — defensive JSON parsing for LLM outputs.

Handles markdown-wrapped JSON, preamble text, and malformed responses.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger("curiosity_run")

# Match ```json ... ``` or ``` ... ```
_MARKDOWN_BLOCK = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL | re.IGNORECASE)


def _extract_balanced_json(text: str) -> str | None:
    """Extract first complete JSON object or array via brace matching."""
    start = -1
    for i, c in enumerate(text):
        if c in "{[":
            start = i
            break
    if start < 0:
        return None
    open_c, close_c = ("{", "}") if text[start] == "{" else ("[", "]")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == open_c:
            depth += 1
        elif text[i] == close_c:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_json_from_llm(raw: str | None, log_on_fail: bool = True) -> Any | None:
    """
    Extract and parse JSON from LLM output. Handles:
    - Empty or None input
    - Markdown code blocks (```json ... ```)
    - Leading/trailing non-JSON text
    Returns parsed object/array or None. Logs raw response on failure if log_on_fail.
    """
    if not raw or not raw.strip():
        if log_on_fail:
            logger.debug("[CURIOSITY] JSON parse: empty input")
        return None

    text = raw.strip()

    # 1. Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Extract from markdown code block
    m = _MARKDOWN_BLOCK.search(text)
    if m:
        block = m.group(1).strip()
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            text = block  # fall through to regex extraction

    # 3. Extract first JSON object or array via brace matching
    extracted = _extract_balanced_json(text)
    if extracted:
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass

    if log_on_fail:
        logger.debug(
            "[CURIOSITY] JSON parse failed for raw (first 300 chars): %s",
            repr(text[:300]) if text else "",
        )
    return None


def generate_with_retry(
    generate_fn,
    max_retries: int = 2,
) -> str | None:
    """
    Call generate_fn() until non-empty result or max_retries. Retries on empty or exception. Returns first non-empty string or None.
    """
    last_err = None
    for _ in range(max_retries + 1):
        try:
            out = generate_fn()
            if out and str(out).strip():
                return str(out).strip()
        except Exception as e:
            last_err = e
            logger.debug("[CURIOSITY] Retry after exception: %s", e)
    if last_err:
        logger.debug("[CURIOSITY] All retries exhausted, last error: %s", last_err)
    return None
