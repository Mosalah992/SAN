"""
sancta_ollama.py — Shared Ollama connection for sancta.py and curiosity_run.py.
Never starts ollama serve. Connects to an already-running instance.

TLS: For ``OLLAMA_URL=https://...``, ``requests`` verifies certificates by default
(verify=True). Use a proper CA or tunnel; do not disable verification in production.
"""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger("sancta.ollama")

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

OLLAMA_BASE = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_MODEL = os.getenv("LOCAL_MODEL", "llama3.2")
DEFAULT_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))


def is_running() -> bool:
    """Check if Ollama is already running."""
    if not requests:
        return False
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/version", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def wait_until_ready(model: str | None = None, timeout: int = 30) -> bool:
    """
    Wait for Ollama to be reachable and the model to be loaded.
    Does NOT start Ollama — call this at the top of any script that uses it.
    """
    if not requests:
        logger.warning("requests module not installed — Ollama check skipped")
        return False
    model = model or DEFAULT_MODEL
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_running():
            try:
                r = requests.post(
                    f"{OLLAMA_BASE}/api/show",
                    json={"name": model},
                    timeout=5,
                )
                if r.status_code == 200:
                    logger.info("Ollama ready — model %s available", model)
                    return True
                logger.warning("Model %s not found. Run: ollama pull %s", model, model)
                return False
            except Exception as e:
                logger.debug("Model check failed: %s", e)
        time.sleep(2)

    logger.error(
        "Ollama not reachable at %s after %ds. Start it manually: ollama serve",
        OLLAMA_BASE,
        timeout,
    )
    return False


def chat(
    prompt: str,
    system: str = "",
    model: str | None = None,
    timeout: int | None = None,
) -> str:
    """Single call to Ollama. Returns response text or raises."""
    if not requests:
        raise RuntimeError("requests module not installed")
    model = model or DEFAULT_MODEL
    timeout = timeout or DEFAULT_TIMEOUT
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json={"model": model, "messages": messages, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()
