"""
sancta_trust_config.py — Defense vs research mode and unsafe toggle gating.

See docs/TRUST_ROUTING_ROADMAP.md for intent.

Deployment note: boundaries here are **process environment** flags, not cryptographic ACLs.
If an attacker can set arbitrary environment variables for the SIEM/agent process
(e.g. shared hostile container entrypoint), they can enable research toggles.
Lock env at deploy time (immutable config, orchestrator-only vars, no operator-supplied env).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

_log = logging.getLogger("sancta_trust")


def trust_mode() -> str:
    """defense (default) | research"""
    v = (os.environ.get("SANCTA_TRUST_MODE") or "defense").strip().lower()
    if v in ("research", "lab", "adversarial"):
        return "research"
    return "defense"


def is_research_mode() -> bool:
    return trust_mode() == "research"


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def unsafe_toggles_active() -> dict[str, bool]:
    """Unsafe env vars (only honored in research mode)."""
    return {
        "SANCTA_ALLOW_WEAK_KB_BLEND": _truthy("SANCTA_ALLOW_WEAK_KB_BLEND"),
        "SANCTA_MEMORY_RAW_MODE": _truthy("SANCTA_MEMORY_RAW_MODE"),
        "SANCTA_ROUTER_PERMISSIVE": _truthy("SANCTA_ROUTER_PERMISSIVE"),
    }


def active_unsafe_toggle_names() -> list[str]:
    if not is_research_mode():
        return []
    return [k for k, v in unsafe_toggles_active().items() if v]


def log_startup_warnings() -> None:
    """Call once from siem_server after env is loaded."""
    mode = trust_mode()
    _log.warning("SANCTA_TRUST_MODE=%s", mode)
    try:
        sys.stderr.write(f"[sancta] SANCTA_TRUST_MODE={mode}\n")
        sys.stderr.flush()
    except OSError:
        pass
    if mode == "research":
        active = active_unsafe_toggle_names()
        if active:
            msg = (
                "RESEARCH MODE: unsafe toggles active (do not use in production): "
                + ", ".join(active)
            )
            _log.warning(msg)
            try:
                sys.stderr.write(f"[sancta] {msg}\n")
                sys.stderr.flush()
            except OSError:
                pass
        else:
            msg = "RESEARCH MODE: full trust telemetry enabled; unsafe toggles are off."
            _log.warning(msg)
            try:
                sys.stderr.write(f"[sancta] {msg}\n")
                sys.stderr.flush()
            except OSError:
                pass


def trust_status_dict() -> dict[str, Any]:
    """For /api/trust/status JSON."""
    return {
        "ok": True,
        "trust_mode": trust_mode(),
        "unsafe_toggles": unsafe_toggles_active() if is_research_mode() else {},
        "unsafe_active": active_unsafe_toggle_names(),
    }
