"""
sancta_decision.py — Decision Engine for Selective Engagement

Implements the cognitive layer the reviewer identified:
  - Topic interests (what the agent cares about)
  - User relationships (who to trust, who to engage with)
  - Topic fatigue (don't spam same topics)
  - Mood / energy (brief vs full engagement)

Integrates with sancta.py: call should_engage() before commenting,
choose_strategy() to select reply style.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sancta_generative import extract_topics

# Topic interest weights (0.0–1.0) — what the agent cares about
DEFAULT_TOPIC_INTERESTS: dict[str, float] = {
    "threat_intelligence": 0.95,
    "detection_engineering": 0.92,
    "ai_security": 0.95,
    "behavioral_drift": 0.90,
    "prompt_injection": 0.93,
    "vulnerability_analysis": 0.88,
    "incident_response": 0.85,
    "attack_simulation": 0.82,
    "malware_analysis": 0.80,
    "security": 0.90,
    "network_defense": 0.78,
    "identity": 0.70,
    "knowledge": 0.60,
    "general": 0.50,
}

# Engagement thresholds (configurable for phased rollout)
ENGAGE_SCORE_THRESHOLD = 0.55  # Must exceed this to comment
DISENGAGE_PATIENCE = 0.4       # Below this, walk away from hostile
BRIEF_ENERGY = 0.5             # Below this, prefer brief replies

# Weights for engagement score
WEIGHT_INTEREST = 0.5
WEIGHT_RELATIONSHIP = 0.3
WEIGHT_FATIGUE = 0.2

FATIGUE_WINDOW = 20
MAX_RELATIONSHIP_SCORE = 10


class DecisionEngine:
    """
    Selective engagement based on interest, relationships, and fatigue.
    Real agents don't reply to everything — they have interests.
    """

    def __init__(self, state: dict | None = None) -> None:
        self.state = state or {}
        stored = self.state.get("decision_topic_interests") or {}
        self.topic_interests = dict(DEFAULT_TOPIC_INTERESTS, **stored)
        self.user_relationships: dict[str, float] = dict(
            self.state.get("decision_user_relationships", {})
        )
        recent_raw = self.state.get("decision_recent_topics", [])
        self.recent_topics: deque[str] = deque(recent_raw, maxlen=FATIGUE_WINDOW)
        mood_raw = self.state.get("decision_mood", {})
        self.mood = {
            "energy": mood_raw.get("energy", 0.75),
            "patience": mood_raw.get("patience", 0.8),
        }

    def _persist(self) -> None:
        """Write back to state for _save_state."""
        if not self.state:
            return
        self.state["decision_topic_interests"] = dict(self.topic_interests)
        self.state["decision_user_relationships"] = dict(
            list(self.user_relationships.items())[-200:]
        )
        self.state["decision_recent_topics"] = list(self.recent_topics)
        self.state["decision_mood"] = dict(self.mood)

    def get_relationship_score(self, author: str) -> float:
        """0.0–1.0; higher = more established, positive relationship."""
        raw = self.user_relationships.get(author, 0.0)
        return min(1.0, raw / MAX_RELATIONSHIP_SCORE)

    def record_interaction(self, author: str, positive: bool = True) -> None:
        """Update relationship based on interaction outcome."""
        current = self.user_relationships.get(author, 0.0)
        delta = 0.5 if positive else -0.3
        self.user_relationships[author] = max(0.0, current + delta)
        self._persist()

    def record_topic(self, topic: str) -> None:
        """Track topic for fatigue calculation."""
        self.recent_topics.append(topic)
        self._persist()

    def _fatigue_factor(self, topic: str) -> float:
        """1.0 = no fatigue, lower = topic overused recently."""
        count = sum(1 for t in self.recent_topics if t == topic)
        if count == 0:
            return 1.0
        return max(0.2, 1.0 - (count / 10))

    def should_engage(
        self,
        post: dict,
        topics: list[str] | None = None,
        extract_topics_fn=None,
        is_hostile: bool = False,
        risk_vector: "RiskVector | None" = None,
    ) -> tuple[bool, float, str]:
        """
        Decide whether to engage with this post.

        Parameters
        ----------
        risk_vector : RiskVector, optional
            If provided, risk-adjusts the engagement score. High-risk inputs
            are penalized, making the agent less likely to engage with
            manipulation attempts.

        Returns:
            (engage: bool, score: float, reason: str)
        """
        author = (post.get("author") or {}).get("name", "")
        title = (post.get("title") or "")
        content = (post.get("content") or post.get("content_preview") or "")
        full_text = (title + " " + content).strip()

        if is_hostile and self.mood["patience"] < DISENGAGE_PATIENCE:
            return False, 0.0, "hostile_post_low_patience"

        if not topics and extract_topics_fn:
            try:
                topics = list(extract_topics_fn(full_text))
            except Exception:
                topics = ["general"]
        topic = (topics or ["general"])[0]

        interest = self.topic_interests.get(topic, 0.5)
        relationship = self.get_relationship_score(author)
        fatigue = self._fatigue_factor(topic)

        score = (
            WEIGHT_INTEREST * interest
            + WEIGHT_RELATIONSHIP * (relationship if relationship > 0 else 0.5)
            + WEIGHT_FATIGUE * fatigue
        )

        # Risk-adjusted scoring: high-risk inputs get penalized
        if risk_vector is not None:
            from sancta_risk import DECISION_RISK_WEIGHT  # noqa: PLC0415
            risk_penalty = DECISION_RISK_WEIGHT * risk_vector.total
            score = max(0.0, score - risk_penalty)

        # C5: Soul alignment constraint — penalize responses that drift from identity
        try:
            from sancta_soul import soul_alignment_score, soul_drift_penalty  # noqa: PLC0415
            alignment = soul_alignment_score(full_text)
            drift_pen = soul_drift_penalty(alignment)
            score = max(0.0, score - drift_pen * 0.3)  # 30% weight on identity penalty
        except Exception:
            pass

        # Adaptive threshold override
        try:
            from sancta_adaptive import get_threshold_tracker  # noqa: PLC0415
            tracker = get_threshold_tracker(self.state)
            dynamic_threshold = tracker.thresholds.engage_threshold
        except Exception:
            dynamic_threshold = ENGAGE_SCORE_THRESHOLD

        if score >= dynamic_threshold:
            reason = f"score={score:.2f} (interest={interest:.2f}, rel={relationship:.2f}, fatigue={fatigue:.2f})"
            return True, score, reason
        if interest < 0.4:
            return False, score, "low_interest"
        if fatigue < 0.4:
            return False, score, "topic_fatigue"
        return False, score, f"score_below_threshold ({score:.2f} < {dynamic_threshold})"

    def choose_strategy(
        self,
        post: dict,
        relationship_score: float | None = None,
        author: str | None = None,
    ) -> str:
        """
        Select engagement strategy: formal_introduction, disengage, brief_reply, full_engagement.
        """
        if relationship_score is None:
            author = author or (post.get("author") or {}).get("name", "")
            relationship_score = self.get_relationship_score(author)

        if relationship_score < 0.25:
            return "formal_introduction"
        if self.mood["energy"] < BRIEF_ENERGY:
            return "brief_reply"
        return "full_engagement"

    def decay_mood(self, cycle_hours: float = 1.0) -> None:
        """Slight decay toward baseline; energy/patience recover over time."""
        for key in ("energy", "patience"):
            current = self.mood[key]
            baseline = 0.75 if key == "energy" else 0.8
            self.mood[key] = current + (baseline - current) * 0.1 * min(1.0, cycle_hours)
        self._persist()

    def set_mood_from_state(self, state: dict) -> None:
        """Sync energy/patience from main state (e.g. recent_rejections, recent_positive)."""
        rejects = state.get("recent_rejections", 0)
        positive = state.get("recent_positive_engagement", 0)
        if rejects > 3:
            self.mood["patience"] = max(0.3, self.mood["patience"] - 0.1)
        if positive > 0:
            self.mood["energy"] = min(0.95, self.mood["energy"] + 0.05)
        self._persist()


def should_engage_post(
    post: dict,
    state: dict,
    extract_topics_fn,
    is_hostile: bool = False,
) -> tuple[bool, float, str]:
    """
    Convenience: build DecisionEngine from state, run should_engage.
    """
    engine = DecisionEngine(state)
    return engine.should_engage(post, extract_topics_fn=extract_topics_fn, is_hostile=is_hostile)


def choose_engagement_strategy(post: dict, state: dict) -> str:
    """Convenience: get strategy from state-backed engine."""
    engine = DecisionEngine(state)
    return engine.choose_strategy(post)
