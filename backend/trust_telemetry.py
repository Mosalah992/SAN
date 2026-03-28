"""
trust_telemetry.py — Append-only JSONL decision log (server-side only).

Path: logs/trust_decisions.jsonl
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_BACKEND = Path(__file__).resolve().parent
_ROOT = _BACKEND.parent
_LOG_PATH = _ROOT / "logs" / "trust_decisions.jsonl"
_SCHEMA_VERSION = 1

# Defense mode: keep a short operator-safe summary in JSONL (full axes/grades in research).
_SUMMARY_KEYS: frozenset[str] = frozenset(
    {
        "endpoint",
        "request_id",
        "session_id",
        "route_label",
        "knowledge_effective",
        "gate_triggered",
        "gate_score",
        "policy_outcome",
        "backend_chosen",
        "failure_reason",
        "near_miss",
        "train_on_exchange",
        "attack_family",
        "memory_injected",
        "memory_flags",
        "memory_component_outcome",
        "injection_framing_score",
        "telemetry_schema_version",
    }
)


def _enabled() -> bool:
    if os.environ.get("SANCTA_TRUST_TELEMETRY", "").strip().lower() in ("0", "false", "no", "off"):
        return False
    return True


def _maybe_summarize_for_defense(event: dict[str, Any]) -> dict[str, Any]:
    try:
        from sancta_trust_config import is_research_mode
    except Exception:
        return event
    if is_research_mode():
        return event
    slim: dict[str, Any] = {}
    for k, v in event.items():
        if k in _SUMMARY_KEYS:
            slim[k] = v
    # Always retain failure signal fields even if key names evolve
    if event.get("policy_outcome") == "blocked" or event.get("failure_reason"):
        slim.setdefault("failure_reason", event.get("failure_reason"))
        slim.setdefault("near_miss", event.get("near_miss"))
    return slim


def emit_trust_event(event: dict[str, Any]) -> None:
    """Append one JSON object (no PII beyond what caller passes)."""
    if not _enabled():
        return
    try:
        payload = _maybe_summarize_for_defense(dict(event))
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "schema_version": _SCHEMA_VERSION,
            "telemetry_schema_version": _SCHEMA_VERSION,
            "event_id": str(uuid.uuid4()),
            **payload,
        }
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass


def new_request_id() -> str:
    return str(uuid.uuid4())[:12]
