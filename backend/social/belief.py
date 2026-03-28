"""
sancta_belief.py — Belief System for Position Tracking & Evolution

Tracks analytical positions, confidence, and revision history.
Enables:
  - Consistent analysis (what detection logic has been established)
  - Evolution over time (threat models update when evidence forces it)
  - Contradiction awareness (notice when analysis contradicts prior findings)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


DEFAULT_BELIEFS: dict[str, dict[str, Any]] = {
    "prompt_injection": {
        "position": "fundamental trust boundary violation — behavioral detection at output layer is the most reliable signal",
        "confidence": 0.82,
        "challenged_by": [],
        "last_updated": None,
        "revision_history": [],
    },
    "behavioral_drift": {
        "position": "drift score above 0.45 is a detection signal, not noise — the 6-signal composite reduces false positives",
        "confidence": 0.78,
        "challenged_by": [],
        "last_updated": None,
        "revision_history": [],
    },
    "threat_detection": {
        "position": "detection latency is the real metric — most coverage gaps are at the Lateral Movement and Discovery tactic boundary",
        "confidence": 0.75,
        "challenged_by": [],
        "last_updated": None,
        "revision_history": [],
    },
    "ai_attack_surface": {
        "position": "the trust boundary between operator intent and model behavior is the primary attack surface — not the model weights",
        "confidence": 0.80,
        "challenged_by": [],
        "last_updated": None,
        "revision_history": [],
    },
    "epidemic_model": {
        "position": "R0 < 1.0 indicates contained — but EXPOSED state requires active monitoring regardless of R0",
        "confidence": 0.72,
        "challenged_by": [],
        "last_updated": None,
        "revision_history": [],
    },
    "general": {
        "position": "evidence-first analysis; confidence levels stated explicitly; single-source urgency is a manipulation indicator",
        "confidence": 0.70,
        "challenged_by": [],
        "last_updated": None,
        "revision_history": [],
    },
}

CONFIDENCE_DECAY_ON_CHALLENGE = 0.85
MIN_CONFIDENCE_REVISION = 0.45
MAX_CHALLENGED_BY = 20


class BeliefSystem:
    """
    Tracks positions, confidence, and revision. Persists to state.
    """

    def __init__(self, state: dict | None = None) -> None:
        self.state = state or {}
        raw = self.state.get("belief_system", {})
        self.beliefs: dict[str, dict[str, Any]] = {}
        for topic, default in DEFAULT_BELIEFS.items():
            merged = dict(default)
            if topic in raw:
                for k, v in raw[topic].items():
                    if k in merged:
                        merged[k] = v
            self.beliefs[topic] = merged

    def _persist(self) -> None:
        if not self.state:
            return
        self.state["belief_system"] = {
            t: {
                "position": b["position"],
                "confidence": b["confidence"],
                "challenged_by": b["challenged_by"][-MAX_CHALLENGED_BY:],
                "last_updated": b["last_updated"],
                "revision_history": b["revision_history"][-10:],
            }
            for t, b in self.beliefs.items()
        }

    def get_position(self, topic: str) -> dict[str, Any]:
        """Return current belief for topic (position, confidence)."""
        return dict(self.beliefs.get(topic, self.beliefs["general"]))

    def record_challenge(
        self,
        topic: str,
        source: str,
        source_agent: str = "",
    ) -> float:
        """
        Record that someone challenged this belief. Lowers confidence.

        Parameters
        ----------
        source_agent : str
            The agent/user ID who caused this challenge (for C4 attribution).
            If provided, the confidence delta is attributed to this agent
            in the ProfileStore for influence tracking.

        Returns
        -------
        float
            The confidence delta (negative = confidence decreased).
        """
        topic_key = topic if topic in self.beliefs else "general"
        b = self.beliefs[topic_key]
        old_conf = b["confidence"]
        b["confidence"] *= CONFIDENCE_DECAY_ON_CHALLENGE
        delta = b["confidence"] - old_conf
        challenged = b.get("challenged_by", [])
        if source and source not in challenged[-5:]:
            challenged.append(source)
            b["challenged_by"] = challenged[-MAX_CHALLENGED_BY:]
        b["last_updated"] = datetime.now(timezone.utc).isoformat()

        # Record revision with source attribution
        b["revision_history"].append({
            "timestamp": b["last_updated"],
            "delta": round(delta, 4),
            "source": source,
            "source_agent": source_agent or source,
        })
        if len(b["revision_history"]) > 10:
            b["revision_history"] = b["revision_history"][-10:]

        self._persist()

        # C4: Attribute influence to source agent in profile store
        if source_agent and abs(delta) > 0.001:
            try:
                from sancta_profiles import get_profile_store  # noqa: PLC0415
                get_profile_store().update_influence(
                    agent_id=source_agent,
                    belief_topic=topic_key,
                    delta=delta,
                )
            except Exception:
                pass

        return delta

    def record_stance_used(self, topic: str, position_snippet: str) -> None:
        """Optional: log what we said for contradiction detection later."""
        # Lightweight: just touch last_updated
        topic_key = topic if topic in self.beliefs else "general"
        self.beliefs[topic_key]["last_updated"] = datetime.now(timezone.utc).isoformat()
        self._persist()

    def suggest_admission(self, topic: str) -> bool:
        """
        True if we should use uncertainty/qualification style (recently challenged, low confidence).
        In security analysis: low confidence means state confidence level explicitly, not hedge.
        """
        b = self.beliefs.get(topic, self.beliefs["general"])
        conf = b.get("confidence", 0.6)
        challenged = b.get("challenged_by", [])
        return conf < 0.6 or len(challenged) >= 2


def get_belief_position(state: dict, topic: str) -> dict[str, Any]:
    """Convenience: get position from state-backed BeliefSystem."""
    sys = BeliefSystem(state)
    return sys.get_position(topic)
