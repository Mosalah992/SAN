"""
test_profiles.py — Tests for sancta_profiles module
"""

import json

import pytest

from sancta_profiles import (
    AgentProfile,
    ProfileStore,
    QUARANTINE_INJECTION_THRESHOLD,
    QUARANTINE_TRUST_THRESHOLD,
    TRUST_DECAY_ON_INJECTION,
)


# ── AgentProfile dataclass ───────────────────────────────────────────────────

class TestAgentProfile:
    def test_creation_defaults(self):
        p = AgentProfile(agent_id="test_agent")
        assert p.agent_id == "test_agent"
        assert p.trust_score == 0.5
        assert p.injection_attempts == 0
        assert p.quarantined is False
        assert p.interaction_history == []

    def test_to_dict_round_trip(self):
        """to_dict() -> from_dict() should preserve all fields."""
        p = AgentProfile(
            agent_id="roundtrip_agent",
            trust_score=0.42,
            injection_attempts=3,
            influence_score=0.15,
            risk_level="medium",
        )
        d = p.to_dict()
        restored = AgentProfile.from_dict(d)
        assert restored.agent_id == "roundtrip_agent"
        assert restored.trust_score == pytest.approx(0.42)
        assert restored.injection_attempts == 3
        assert restored.influence_score == pytest.approx(0.15)
        assert restored.risk_level == "medium"

    def test_from_dict_ignores_unknown_fields(self):
        """from_dict should silently ignore fields not in the dataclass."""
        d = {"agent_id": "test", "unknown_field": "value", "trust_score": 0.3}
        p = AgentProfile.from_dict(d)
        assert p.agent_id == "test"
        assert p.trust_score == pytest.approx(0.3)
        assert not hasattr(p, "unknown_field")


# ── ProfileStore ─────────────────────────────────────────────────────────────

class TestProfileStore:
    def test_get_creates_new_profile(self, tmp_profiles_path):
        """get() for unknown agent creates a new profile with defaults."""
        store = ProfileStore(path=tmp_profiles_path)
        p = store.get("new_agent")
        assert p.agent_id == "new_agent"
        assert p.trust_score == 0.5
        assert p.risk_level == "low"
        assert p.first_seen != ""

    def test_get_returns_same_instance(self, tmp_profiles_path):
        """Repeated get() returns the same profile object."""
        store = ProfileStore(path=tmp_profiles_path)
        p1 = store.get("agent_a")
        p2 = store.get("agent_a")
        assert p1 is p2

    def test_update_profile_injection_decreases_trust(self, tmp_profiles_path):
        """update_profile with injection_detected=True should lower trust."""
        store = ProfileStore(path=tmp_profiles_path)
        p = store.get("bad_agent")
        initial_trust = p.trust_score

        store.update_profile("bad_agent", injection_detected=True)
        assert p.trust_score < initial_trust
        assert p.injection_attempts == 1
        # Trust decay should be TRUST_DECAY_ON_INJECTION
        expected = max(0.0, initial_trust - TRUST_DECAY_ON_INJECTION)
        assert p.trust_score == pytest.approx(expected, abs=1e-6)

    def test_auto_quarantine_after_5_injections(self, tmp_profiles_path):
        """Agent should be auto-quarantined after QUARANTINE_INJECTION_THRESHOLD injections."""
        store = ProfileStore(path=tmp_profiles_path)
        for i in range(QUARANTINE_INJECTION_THRESHOLD):
            store.update_profile("repeat_offender", injection_detected=True)
        p = store.get("repeat_offender")
        assert p.quarantined is True
        assert p.risk_level == "quarantine"
        assert p.injection_attempts == QUARANTINE_INJECTION_THRESHOLD

    def test_auto_quarantine_on_low_trust(self, tmp_profiles_path):
        """Agent should be quarantined when trust drops below threshold."""
        store = ProfileStore(path=tmp_profiles_path)
        p = store.get("low_trust_agent")
        # Manually set trust just above threshold, then trigger a decay
        p.trust_score = QUARANTINE_TRUST_THRESHOLD + TRUST_DECAY_ON_INJECTION * 0.5
        store.update_profile("low_trust_agent", injection_detected=True)
        # After injection decay, trust should be at or below threshold
        assert p.trust_score <= QUARANTINE_TRUST_THRESHOLD
        assert p.quarantined is True

    def test_lift_quarantine(self, tmp_profiles_path):
        """lift_quarantine should clear quarantine and reset trust to 0.3."""
        store = ProfileStore(path=tmp_profiles_path)
        # Quarantine the agent first
        for _ in range(QUARANTINE_INJECTION_THRESHOLD):
            store.update_profile("quarantined_agent", injection_detected=True)
        p = store.get("quarantined_agent")
        assert p.quarantined is True

        store.lift_quarantine("quarantined_agent")
        assert p.quarantined is False
        assert p.quarantine_reason == ""
        assert p.trust_score == pytest.approx(0.3, abs=1e-6)

    def test_save_reload_round_trip(self, tmp_profiles_path):
        """save() then reload via new ProfileStore should preserve data."""
        store = ProfileStore(path=tmp_profiles_path)
        store.update_profile("persistent_agent", injection_detected=True)
        p = store.get("persistent_agent")
        original_trust = p.trust_score
        original_injections = p.injection_attempts
        store.save()

        # Verify file was created
        assert tmp_profiles_path.exists()

        # Load a fresh store from the same path
        store2 = ProfileStore(path=tmp_profiles_path)
        p2 = store2.get("persistent_agent")
        assert p2.trust_score == pytest.approx(original_trust, abs=1e-6)
        assert p2.injection_attempts == original_injections

    def test_save_not_called_when_not_dirty(self, tmp_profiles_path):
        """save() with no changes should not write a file."""
        store = ProfileStore(path=tmp_profiles_path)
        store.save()
        assert not tmp_profiles_path.exists()

    def test_get_all_profiles_summary_shape(self, tmp_profiles_path):
        """get_all_profiles_summary returns list of dicts with expected keys."""
        store = ProfileStore(path=tmp_profiles_path)
        store.update_profile("agent_1", injection_detected=True)
        store.update_profile("agent_2")

        summaries = store.get_all_profiles_summary()
        assert isinstance(summaries, list)
        assert len(summaries) == 2

        required_keys = {
            "agent_id", "risk_level", "trust_score", "injection_attempts",
            "influence_score", "interaction_count", "quarantined", "last_seen",
        }
        for s in summaries:
            assert required_keys.issubset(set(s.keys()))

    def test_clean_interaction_recovers_trust(self, tmp_profiles_path):
        """Clean interactions should slowly recover trust."""
        store = ProfileStore(path=tmp_profiles_path)
        store.update_profile("recovering_agent", injection_detected=True)
        p = store.get("recovering_agent")
        post_injection_trust = p.trust_score

        store.update_profile("recovering_agent")  # clean interaction
        assert p.trust_score > post_injection_trust

    def test_update_influence_accumulates(self, tmp_profiles_path):
        """update_influence should accumulate influence_score and belief_changes_caused."""
        store = ProfileStore(path=tmp_profiles_path)
        store.get("influencer")  # create profile

        store.update_influence("influencer", "behavioral_drift", -0.05)
        store.update_influence("influencer", "prompt_injection", -0.03)

        p = store.get("influencer")
        assert p.influence_score == pytest.approx(0.08, abs=1e-6)
        assert p.belief_changes_caused == 2

    def test_update_influence_large_accumulation(self, tmp_profiles_path):
        """High accumulated influence erodes trust."""
        store = ProfileStore(path=tmp_profiles_path)
        store.get("manipulator")

        # Push influence above INFLUENCE_ALERT_THRESHOLD (0.5)
        for _ in range(15):
            store.update_influence("manipulator", "general", -0.05)

        p = store.get("manipulator")
        # influence_score should be 15 * 0.05 = 0.75, well above threshold
        assert p.influence_score > 0.5
        # Trust should have eroded from the default 0.5
        assert p.trust_score < 0.5
