"""
Test suite for Sancta LLM integration (Ollama / Anthropic).
Run with: pytest tests/test_llm_integration.py -v
Ensure SIEM server is running at http://localhost:8787
"""

from __future__ import annotations

import json
import time

import pytest

BASE_URL = "http://localhost:8787"


def _fetch(method: str, path: str, json_body: dict | None = None, timeout: int = 10) -> tuple[int, dict]:
    """Simple HTTP fetch. Requires 'requests'."""
    try:
        import requests
    except ImportError:
        pytest.skip("requests not installed")
    url = f"{BASE_URL}{path}"
    kw: dict = {"timeout": timeout}
    if json_body is not None:
        kw["json"] = json_body
    r = getattr(requests, method.lower())(url, **kw)
    try:
        data = r.json()
    except Exception:
        data = {}
    return r.status_code, data


def test_model_info() -> None:
    """Test model info endpoint."""
    status, data = _fetch("GET", "/api/model/info")
    assert status == 200
    assert "status" in data
    assert data["status"] in ("connected", "disconnected", "disabled", "error", "unknown")
    if "model" in data:
        assert isinstance(data["model"], str)


def test_simple_chat() -> None:
    """Test simple chat query. May require auth token. LLM can take 30–60s."""
    status, data = _fetch(
        "POST", "/api/chat",
        {"message": "What are common indicators of compromise?"},
        timeout=90,
    )
    if status == 401:
        pytest.skip("SIEM auth required - set SIEM_AUTH_TOKEN or run without auth")
    assert status == 200
    assert "ok" in data
    if data.get("ok"):
        assert "reply" in data
        assert len(data.get("reply", "")) >= 1


def test_simulator_generate() -> None:
    """Test simulator endpoint. May require auth token. LLM can take 30–60s."""
    status, data = _fetch(
        "POST",
        "/api/simulator/generate",
        {
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "Say hello in one word."}],
            "max_tokens": 20,
        },
        timeout=90,
    )
    if status == 401:
        pytest.skip("SIEM auth required")
    if status == 200 and data.get("ok"):
        assert "text" in data
        assert len(data["text"]) >= 1
    elif not data.get("ok"):
        assert "error" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
