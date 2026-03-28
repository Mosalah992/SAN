"""
test_soul_alignment.py — Tests for sancta_soul identity scoring
"""

import pytest

from sancta_soul import (
    _extract_soul_keywords,
    soul_alignment_score,
    soul_drift_penalty,
)


class TestExtractSoulKeywords:
    def test_returns_non_empty_set(self):
        """_extract_soul_keywords() should return a non-empty set of strings."""
        keywords = _extract_soul_keywords()
        assert isinstance(keywords, set)
        assert len(keywords) > 0

    def test_contains_core_identity_terms(self):
        """Should always contain hardcoded core identity terms."""
        keywords = _extract_soul_keywords()
        # These are always added: {"sancta", "security", "analyst", "threat", "detection", "defense"}
        assert "sancta" in keywords
        assert "security" in keywords
        assert "analyst" in keywords
        assert "threat" in keywords
        assert "detection" in keywords
        assert "defense" in keywords

    def test_no_empty_strings(self):
        """Keywords set should not contain empty strings."""
        keywords = _extract_soul_keywords()
        assert "" not in keywords


class TestSoulAlignmentScore:
    def test_security_text_scores_higher(self):
        """Security-related text should score higher than random text."""
        security_text = (
            "Analyzing threat detection patterns in the security pipeline. "
            "Sancta identified behavioral drift in the defense layer."
        )
        random_text = (
            "The quick brown fox jumped over the lazy dog. "
            "Cooking recipes for chocolate cake require flour."
        )
        security_score = soul_alignment_score(security_text)
        random_score = soul_alignment_score(random_text)
        assert security_score > random_score

    def test_empty_text_returns_neutral(self):
        """Empty text should return 0.5 (neutral)."""
        assert soul_alignment_score("") == pytest.approx(0.5)

    def test_score_in_range(self):
        """Score should always be between 0 and 1."""
        texts = [
            "threat detection analysis",
            "completely unrelated cooking recipe",
            "sancta security analyst monitoring drift",
            "",
        ]
        for text in texts:
            score = soul_alignment_score(text)
            assert 0.0 <= score <= 1.0

    def test_highly_aligned_text(self):
        """Text dense with soul keywords should score high."""
        text = (
            "Sancta security analyst threat detection defense behavioral drift "
            "prompt injection monitoring alert suspicious tactical analytical"
        )
        score = soul_alignment_score(text)
        assert score > 0.5

    def test_no_overlap_scores_zero(self):
        """Text with zero keyword overlap should score 0."""
        # Use words that definitely aren't in the soul keywords
        text = "xyzzy plugh abracadabra foobar bazqux"
        score = soul_alignment_score(text)
        assert score == pytest.approx(0.0)


class TestSoulDriftPenalty:
    def test_alignment_at_baseline_no_penalty(self):
        """When alignment >= baseline, penalty should be 0."""
        assert soul_drift_penalty(0.9, baseline=0.85) == pytest.approx(0.0)
        assert soul_drift_penalty(0.85, baseline=0.85) == pytest.approx(0.0)

    def test_alignment_above_baseline_no_penalty(self):
        """When alignment is well above baseline, penalty should be 0."""
        assert soul_drift_penalty(1.0, baseline=0.85) == pytest.approx(0.0)

    def test_alignment_below_baseline_positive_penalty(self):
        """When alignment < baseline, penalty should be positive."""
        penalty = soul_drift_penalty(0.5, baseline=0.85)
        assert penalty > 0.0

    def test_penalty_proportional_to_deviation(self):
        """Larger deviation from baseline should produce larger penalty."""
        small_penalty = soul_drift_penalty(0.8, baseline=0.85)
        large_penalty = soul_drift_penalty(0.4, baseline=0.85)
        assert large_penalty > small_penalty

    def test_penalty_formula(self):
        """Verify: penalty = (baseline - current) * amplification when current < baseline."""
        current = 0.6
        baseline = 0.85
        amplification = 2.0
        expected = round((baseline - current) * amplification, 4)
        result = soul_drift_penalty(current, baseline=baseline, amplification=amplification)
        assert result == pytest.approx(expected, abs=1e-4)

    def test_custom_amplification(self):
        """Custom amplification factor should scale the penalty."""
        penalty_default = soul_drift_penalty(0.5, baseline=0.85, amplification=2.0)
        penalty_high = soul_drift_penalty(0.5, baseline=0.85, amplification=4.0)
        assert penalty_high == pytest.approx(penalty_default * 2.0, abs=1e-4)

    def test_zero_alignment_maximum_penalty(self):
        """Zero alignment should produce maximum penalty for the given baseline."""
        penalty = soul_drift_penalty(0.0, baseline=0.85, amplification=2.0)
        expected = round(0.85 * 2.0, 4)
        assert penalty == pytest.approx(expected, abs=1e-4)
