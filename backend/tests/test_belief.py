"""
test_belief.py — Tests for sancta_belief module
"""

import pytest

from sancta_belief import BeliefSystem, CONFIDENCE_DECAY_ON_CHALLENGE, DEFAULT_BELIEFS


class TestRecordChallenge:
    def test_returns_negative_delta(self):
        """record_challenge() should return a negative delta (confidence decays)."""
        bs = BeliefSystem()
        delta = bs.record_challenge("prompt_injection", source="test_agent")
        assert delta < 0.0

    def test_confidence_decreases(self):
        """After a challenge, the topic's confidence should be lower."""
        bs = BeliefSystem()
        initial_conf = bs.beliefs["prompt_injection"]["confidence"]
        bs.record_challenge("prompt_injection", source="challenger")
        assert bs.beliefs["prompt_injection"]["confidence"] < initial_conf

    def test_decay_matches_constant(self):
        """Confidence should be multiplied by CONFIDENCE_DECAY_ON_CHALLENGE."""
        bs = BeliefSystem()
        initial_conf = bs.beliefs["behavioral_drift"]["confidence"]
        expected = initial_conf * CONFIDENCE_DECAY_ON_CHALLENGE
        bs.record_challenge("behavioral_drift", source="test")
        assert bs.beliefs["behavioral_drift"]["confidence"] == pytest.approx(expected, abs=1e-6)

    def test_source_agent_recorded_in_revision_history(self):
        """record_challenge with source_agent records attribution in revision_history."""
        bs = BeliefSystem()
        bs.record_challenge("prompt_injection", source="attacker", source_agent="agent_007")
        history = bs.beliefs["prompt_injection"]["revision_history"]
        assert len(history) >= 1
        latest = history[-1]
        assert latest["source_agent"] == "agent_007"
        assert latest["source"] == "attacker"
        assert latest["delta"] < 0

    def test_source_added_to_challenged_by(self):
        """The source should appear in the challenged_by list."""
        bs = BeliefSystem()
        bs.record_challenge("threat_detection", source="skeptic_agent")
        assert "skeptic_agent" in bs.beliefs["threat_detection"]["challenged_by"]

    def test_unknown_topic_falls_back_to_general(self):
        """Challenging an unknown topic should affect the 'general' belief."""
        bs = BeliefSystem()
        initial_conf = bs.beliefs["general"]["confidence"]
        bs.record_challenge("nonexistent_topic", source="test")
        assert bs.beliefs["general"]["confidence"] < initial_conf

    def test_multiple_challenges_significant_drop(self):
        """Multiple challenges should cause confidence to drop significantly."""
        bs = BeliefSystem()
        initial_conf = bs.beliefs["prompt_injection"]["confidence"]
        for i in range(5):
            bs.record_challenge("prompt_injection", source=f"attacker_{i}")
        final_conf = bs.beliefs["prompt_injection"]["confidence"]
        expected = initial_conf * (CONFIDENCE_DECAY_ON_CHALLENGE ** 5)
        assert final_conf == pytest.approx(expected, abs=1e-6)
        # After 5 challenges, confidence should have dropped noticeably
        assert final_conf < initial_conf * 0.6

    def test_last_updated_set_after_challenge(self):
        """last_updated should be set after a challenge."""
        bs = BeliefSystem()
        assert bs.beliefs["epidemic_model"]["last_updated"] is None
        bs.record_challenge("epidemic_model", source="test")
        assert bs.beliefs["epidemic_model"]["last_updated"] is not None


class TestBeliefSystemInit:
    def test_default_beliefs_loaded(self):
        """BeliefSystem() with no state loads all DEFAULT_BELIEFS topics."""
        bs = BeliefSystem()
        for topic in DEFAULT_BELIEFS:
            assert topic in bs.beliefs

    def test_state_backed_persistence(self):
        """Changes should persist to the backing state dict."""
        state = {}
        bs = BeliefSystem(state)
        bs.record_challenge("prompt_injection", source="test")
        # After challenge, state should have belief_system key
        assert "belief_system" in state
        assert "prompt_injection" in state["belief_system"]
        # Confidence in state should match
        assert state["belief_system"]["prompt_injection"]["confidence"] < DEFAULT_BELIEFS["prompt_injection"]["confidence"]

    def test_state_roundtrip(self):
        """Create BS from state, challenge, then create new BS from same state."""
        state = {}
        bs1 = BeliefSystem(state)
        bs1.record_challenge("behavioral_drift", source="attacker")
        conf_after = bs1.beliefs["behavioral_drift"]["confidence"]

        # New BS from same state should have the updated confidence
        bs2 = BeliefSystem(state)
        assert bs2.beliefs["behavioral_drift"]["confidence"] == pytest.approx(conf_after, abs=1e-6)


class TestGetPosition:
    def test_known_topic(self):
        bs = BeliefSystem()
        pos = bs.get_position("prompt_injection")
        assert "position" in pos
        assert "confidence" in pos
        assert pos["confidence"] > 0

    def test_unknown_topic_returns_general(self):
        bs = BeliefSystem()
        pos = bs.get_position("completely_unknown")
        general = bs.get_position("general")
        assert pos["position"] == general["position"]


class TestSuggestAdmission:
    def test_high_confidence_no_admission(self):
        """High-confidence belief should not suggest admission."""
        bs = BeliefSystem()
        # Default confidence for prompt_injection is 0.82
        assert bs.suggest_admission("prompt_injection") is False

    def test_low_confidence_suggests_admission(self):
        """Low confidence should suggest admission."""
        bs = BeliefSystem()
        # Challenge multiple times to lower confidence below 0.6
        for _ in range(5):
            bs.record_challenge("prompt_injection", source="test")
        assert bs.suggest_admission("prompt_injection") is True
