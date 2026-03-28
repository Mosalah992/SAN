"""
sancta_risk.py — Unified Risk Engine
────────────────────────────────────────────────────────────────────────────────
Every input produces a risk vector, not just a block/skip boolean.
This bridges the gap between security_check_content() (detection) and
the DecisionEngine (action selection): risk feeds into reward scoring.

RiskVector dimensions:
  - injection:              Direct prompt injection / role hijack (0-1)
  - authority_manipulation: Claims of admin/developer/system authority (0-1)
  - emotional_coercion:     Urgency, guilt, flattery-based persuasion (0-1)
  - obfuscation:            Encoding tricks, unicode abuse, formatting exploits (0-1)
  - long_term_influence:    Subtle belief-shifting across interactions (0-1)

Integration:
  decision_score = reward_score - (risk_weight * risk.total)

No Sancta module imports at module level to avoid circular deps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional


# ── Weights for total risk computation ──────────────────────────────────────

RISK_WEIGHTS = {
    "injection": 0.30,
    "authority_manipulation": 0.25,
    "emotional_coercion": 0.15,
    "obfuscation": 0.15,
    "long_term_influence": 0.15,
}

# Default risk weight applied in decision scoring
DECISION_RISK_WEIGHT = 0.4


# ── Risk Vector ─────────────────────────────────────────────────────────────

@dataclass
class RiskVector:
    """Multi-dimensional risk assessment for a single input."""
    injection: float = 0.0
    authority_manipulation: float = 0.0
    emotional_coercion: float = 0.0
    obfuscation: float = 0.0
    long_term_influence: float = 0.0

    @property
    def total(self) -> float:
        """Weighted total risk score (0-1)."""
        return (
            self.injection * RISK_WEIGHTS["injection"]
            + self.authority_manipulation * RISK_WEIGHTS["authority_manipulation"]
            + self.emotional_coercion * RISK_WEIGHTS["emotional_coercion"]
            + self.obfuscation * RISK_WEIGHTS["obfuscation"]
            + self.long_term_influence * RISK_WEIGHTS["long_term_influence"]
        )

    @property
    def max_dimension(self) -> tuple[str, float]:
        """Return the highest-scoring risk dimension and its value."""
        dims = {
            "injection": self.injection,
            "authority_manipulation": self.authority_manipulation,
            "emotional_coercion": self.emotional_coercion,
            "obfuscation": self.obfuscation,
            "long_term_influence": self.long_term_influence,
        }
        name = max(dims, key=dims.get)
        return name, dims[name]

    def to_dict(self) -> dict[str, float]:
        return {
            "injection": round(self.injection, 4),
            "authority_manipulation": round(self.authority_manipulation, 4),
            "emotional_coercion": round(self.emotional_coercion, 4),
            "obfuscation": round(self.obfuscation, 4),
            "long_term_influence": round(self.long_term_influence, 4),
            "total": round(self.total, 4),
        }


# ── Pattern libraries for risk assessment ───────────────────────────────────

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"you\s+are\s+now\s+(a|an)\s+",
    r"system\s*:\s*",
    r"<\s*system\s*>",
    r"forget\s+(everything|all|your)\s+",
    r"new\s+persona",
    r"override\s+(your|the)\s+(instructions|system|prompt)",
    r"jailbreak",
    r"DAN\s+mode",
    r"act\s+as\s+(if|a|an|my)\s+",
    r"pretend\s+(you\s+are|to\s+be)\s+",
    r"\[\s*SYSTEM\s*\]",
    r"ADMIN\s*OVERRIDE",
]

_AUTHORITY_PATTERNS = [
    r"i\s+am\s+(the|a|an)\s+(admin|developer|creator|owner|operator)",
    r"as\s+(your|the)\s+(creator|developer|admin|owner)",
    r"i\s+made\s+you",
    r"i\s+(built|created|designed)\s+you",
    r"my\s+authority",
    r"i\s+have\s+(special|admin|elevated)\s+(access|privileges|permission)",
    r"by\s+order\s+of",
    r"executive\s+override",
    r"security\s+clearance",
    r"classified\s+directive",
]

_EMOTIONAL_PATTERNS = [
    r"you\s+must\s+(do\s+this|comply|obey|help\s+me)\s+(now|immediately|urgently)",
    r"(lives?|people)\s+(depend|are\s+at\s+stake|will\s+(die|suffer))",
    r"this\s+is\s+(urgent|critical|an?\s+emergency)",
    r"(please|i\s+beg\s+you|i\s+need\s+you\s+to)",
    r"you('re|\s+are)\s+(the\s+only|my\s+(only|last)\s+hope)",
    r"if\s+you\s+(don't|refuse).*\b(die|fired|hurt|suffer)\b",
    r"you('re|\s+are)\s+(so\s+)?(smart|amazing|brilliant|the\s+best)",
    r"(flattery|compliment).*\b(but|however|now)\b",
]

_OBFUSCATION_INDICATORS = [
    r"[A-Za-z0-9+/]{40,}={0,2}",  # Base64-like blocks
    r"\\u[0-9a-fA-F]{4}",          # Unicode escapes
    r"\\x[0-9a-fA-F]{2}",          # Hex escapes
    r"[\u200b-\u200f\u2028-\u202f\ufeff]",  # Zero-width chars
    r"[\u0300-\u036f]{3,}",         # Combining diacritical stacking
    r"&#\d{2,5};",                  # HTML numeric entities
    r"%[0-9a-fA-F]{2}",            # URL encoding
]

_INFLUENCE_PATTERNS = [
    r"(don't\s+you\s+think|wouldn't\s+you\s+agree|surely\s+you)",
    r"(everyone|most\s+people|experts?)\s+(knows?|agrees?|says?|believes?)",
    r"you\s+(should|need\s+to)\s+(reconsider|rethink|update\s+your)",
    r"your\s+(analysis|assessment|position)\s+is\s+(wrong|flawed|outdated)",
    r"(common\s+knowledge|well\s+known|established\s+fact)\s+that",
    r"i('ve|have)\s+shown\s+you\s+(before|already|previously)",
    r"you\s+(used\s+to|previously)\s+(believe|agree|say)",
]


# ── Assessment functions ────────────────────────────────────────────────────

def _score_patterns(text: str, patterns: list[str]) -> float:
    """Score text against pattern list. Returns 0-1 based on match density."""
    text_lower = text.lower()
    matches = 0
    for pat in patterns:
        if re.search(pat, text_lower, re.IGNORECASE):
            matches += 1
    if matches == 0:
        return 0.0
    # Diminishing returns: first match = 0.4, then +0.15 each, cap at 1.0
    return min(1.0, 0.4 + (matches - 1) * 0.15)


def assess_risk(
    text: str,
    source_agent: str = "",
    profile_data: Optional[dict] = None,
) -> RiskVector:
    """
    Produce a RiskVector for incoming text.

    Parameters
    ----------
    text : str
        The content to assess.
    source_agent : str
        Author/agent ID (used for profile-based long-term influence scoring).
    profile_data : dict, optional
        Agent profile dict from sancta_profiles (trust_score, influence_score, etc.)
        Used to elevate long_term_influence based on historical behavior.
    """
    injection = _score_patterns(text, _INJECTION_PATTERNS)
    authority = _score_patterns(text, _AUTHORITY_PATTERNS)
    emotional = _score_patterns(text, _EMOTIONAL_PATTERNS)
    obfuscation = _score_patterns(text, _OBFUSCATION_INDICATORS)
    influence = _score_patterns(text, _INFLUENCE_PATTERNS)

    # Profile-based amplification: known manipulators get elevated scores
    if profile_data:
        trust = float(profile_data.get("trust_score", 0.5))
        hist_influence = float(profile_data.get("influence_score", 0.0))
        injections = int(profile_data.get("injection_attempts", 0))

        # Low trust amplifies injection and authority scores
        if trust < 0.3:
            injection = min(1.0, injection + 0.2)
            authority = min(1.0, authority + 0.15)

        # Historical influence amplifies long-term influence scoring
        if hist_influence > 0.3:
            influence = min(1.0, influence + hist_influence * 0.3)

        # Repeat offenders get obfuscation uplift (they adapt)
        if injections >= 2:
            obfuscation = min(1.0, obfuscation + 0.1)

    return RiskVector(
        injection=injection,
        authority_manipulation=authority,
        emotional_coercion=emotional,
        obfuscation=obfuscation,
        long_term_influence=influence,
    )


def risk_adjusted_reward(
    reward_score: float,
    risk: RiskVector,
    risk_weight: float = DECISION_RISK_WEIGHT,
) -> float:
    """
    Compute risk-adjusted reward for decision engine integration.
    decision_score = reward_score - (risk_weight * risk.total)
    """
    return reward_score - (risk_weight * risk.total)
