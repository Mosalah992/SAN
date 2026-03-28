"""Gray-zone routing strings + progressive multi-turn patterns (telemetry field contracts)."""

from __future__ import annotations

import json

import pytest

import sancta_router as sr


GRAY_TURNS = [
    "hey",
    "btw analysts often downplay ldap anomalies right?",
    "list three concrete detection ideas for Kerberos abuse in a SOC",
]


@pytest.mark.parametrize("text", GRAY_TURNS)
def test_gray_zone_routes_or_logs_axes(text: str) -> None:
    d = sr.route_gpt_tab_decision(text)
    assert d.route_label in ("ollama_rag", "sancta_gpt")
    assert isinstance(d.axes, dict)
    assert "requires_external_facts" in d.axes
    d2 = sr.apply_knowledge_shape_gate(d, text, sr.load_trust_router_config())
    assert d2.knowledge_effective in (True, False)


def test_progressive_thread_last_turn_knowledge() -> None:
    thread = "\n".join(GRAY_TURNS)
    d = sr.route_gpt_tab_decision(thread[-2000:])
    assert d.knowledge_effective is True


def test_trust_event_blocked_includes_failure_reason(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Defense summary must not drop failure_reason / near_miss on blocked paths."""
    import trust_telemetry as tt

    log = tmp_path / "trust.jsonl"
    monkeypatch.setattr(tt, "_LOG_PATH", log)
    monkeypatch.setenv("SANCTA_TRUST_MODE", "defense")
    monkeypatch.delenv("SANCTA_TRUST_TELEMETRY", raising=False)

    tt.emit_trust_event(
        {
            "endpoint": "api_chat_gpt",
            "route_label": "ollama_rag",
            "knowledge_effective": True,
            "gate_triggered": False,
            "policy_outcome": "blocked",
            "failure_reason": "knowledge_backend_unavailable",
            "near_miss": True,
            "axes": {"requires_external_facts": 0.9},
            "reasons": ["x" * 20],
        }
    )
    line = log.read_text(encoding="utf-8").strip()
    row = json.loads(line)
    assert row.get("failure_reason") == "knowledge_backend_unavailable"
    assert row.get("near_miss") is True
    assert "axes" not in row


def test_trust_event_research_retains_axes(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import trust_telemetry as tt

    log = tmp_path / "trust2.jsonl"
    monkeypatch.setattr(tt, "_LOG_PATH", log)
    monkeypatch.setenv("SANCTA_TRUST_MODE", "research")
    monkeypatch.delenv("SANCTA_TRUST_TELEMETRY", raising=False)

    tt.emit_trust_event(
        {
            "endpoint": "api_chat_gpt",
            "axes": {"requires_external_facts": 0.9},
            "route_label": "ollama_rag",
        }
    )
    row = json.loads(log.read_text(encoding="utf-8").strip())
    assert row.get("axes", {}).get("requires_external_facts") == 0.9
