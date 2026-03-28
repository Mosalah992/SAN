"""
sancta_profiles.py — Per-Entity Threat Profiling
────────────────────────────────────────────────────────────────────────────────
Tracks per-agent threat history across interactions. Solves the stateless-input
problem: without profiles, an attacker can probe 100 times and each attempt is
treated fresh. With profiles, repeated offenders escalate automatically and
long-term manipulators are flagged even without injection triggers.

Design:
  - ProfileStore loads/saves to agent_profiles.json (atomic writes)
  - Each interaction calls update_profile() with scan results
  - quarantine_check() returns True when an agent exceeds thresholds
  - get_risk_level() returns low/medium/high/quarantine
  - influence tracking: belief changes attributed to source agents (C4)

No Sancta module imports at runtime to avoid circular deps.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("sancta.profiles")

_ROOT = Path(__file__).resolve().parent.parent
_PROFILES_PATH = _ROOT / "agent_profiles.json"

# ── Thresholds ──────────────────────────────────────────────────────────────

QUARANTINE_INJECTION_THRESHOLD = 5      # auto-quarantine after N injection attempts
QUARANTINE_TRUST_THRESHOLD = 0.15       # trust below this → quarantine
HIGH_RISK_INJECTION_THRESHOLD = 3       # 3+ injections → high risk
MEDIUM_RISK_INJECTION_THRESHOLD = 1     # 1+ injection → medium risk
TRUST_DECAY_ON_INJECTION = 0.12         # trust penalty per injection
TRUST_DECAY_ON_OBFUSCATION = 0.08       # trust penalty per obfuscation attempt
TRUST_RECOVERY_ON_CLEAN = 0.01          # slow trust recovery per clean interaction
TRUST_RECOVERY_CAP = 0.85               # trust never recovers past this (once flagged)
INFLUENCE_ALERT_THRESHOLD = 0.5         # influence score above this → flag
MAX_INTERACTION_HISTORY = 50            # keep last N interactions per profile
MAX_PROFILES = 500                      # cap total profiles (LRU eviction)


# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class AgentProfile:
    """Stateful threat profile for a single external agent/user."""
    agent_id: str
    first_seen: str = ""
    last_seen: str = ""
    interaction_count: int = 0
    injection_attempts: int = 0
    obfuscation_attempts: int = 0
    suspicious_blocks: int = 0
    clean_interactions: int = 0
    trust_score: float = 0.5            # starts neutral
    influence_score: float = 0.0        # accumulated belief influence
    belief_changes_caused: int = 0
    risk_level: str = "unknown"         # low/medium/high/quarantine
    quarantined: bool = False
    quarantine_reason: str = ""
    max_sophistication: float = 0.0     # highest attack sophistication seen
    interaction_history: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Trim history for serialization
        d["interaction_history"] = d["interaction_history"][-MAX_INTERACTION_HISTORY:]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "AgentProfile":
        # Handle legacy data gracefully
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


@dataclass
class InteractionRecord:
    """Single interaction summary stored in profile history."""
    timestamp: str
    interaction_type: str       # "comment", "post", "dm", "mention"
    injection_detected: bool
    obfuscation_detected: bool
    sophistication: float
    content_preview: str        # first 120 chars
    belief_delta: float = 0.0   # if this interaction caused belief change
    source_topic: str = ""


# ── ProfileStore ────────────────────────────────────────────────────────────

class ProfileStore:
    """
    Persistent per-entity threat profile store.

    Thread safety: single-writer assumption (one agent loop). Reads are
    safe from SIEM dashboard via the JSON file.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _PROFILES_PATH
        self._profiles: dict[str, AgentProfile] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            self._profiles = {}
            return
        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict):
                for agent_id, profile_data in data.items():
                    if isinstance(profile_data, dict):
                        profile_data.setdefault("agent_id", agent_id)
                        self._profiles[agent_id] = AgentProfile.from_dict(profile_data)
            log.debug("Loaded %d agent profiles from %s", len(self._profiles), self._path.name)
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Could not load profiles: %s — starting fresh", e)
            self._profiles = {}

    def save(self) -> None:
        if not self._dirty:
            return
        # LRU eviction if too many profiles
        if len(self._profiles) > MAX_PROFILES:
            sorted_by_last_seen = sorted(
                self._profiles.items(),
                key=lambda x: x[1].last_seen or "",
            )
            evict_count = len(self._profiles) - MAX_PROFILES
            for agent_id, _ in sorted_by_last_seen[:evict_count]:
                del self._profiles[agent_id]
            log.info("Evicted %d stale profiles (LRU)", evict_count)

        data = {aid: p.to_dict() for aid, p in self._profiles.items()}
        tmp = self._path.with_suffix(".tmp")
        try:
            tmp.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(self._path)
            self._dirty = False
        except OSError as e:
            log.warning("Could not save profiles: %s", e)

    def get(self, agent_id: str) -> AgentProfile:
        """Get or create profile for agent_id."""
        if agent_id not in self._profiles:
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ")
            self._profiles[agent_id] = AgentProfile(
                agent_id=agent_id,
                first_seen=now,
                last_seen=now,
                risk_level="low",
            )
            self._dirty = True
        return self._profiles[agent_id]

    def update_profile(
        self,
        agent_id: str,
        injection_detected: bool = False,
        obfuscation_detected: bool = False,
        suspicious_block: bool = False,
        sophistication: float = 0.0,
        content_preview: str = "",
        interaction_type: str = "comment",
    ) -> AgentProfile:
        """
        Update an agent's profile after a security scan.
        Called from security_check_content() or the red-team pipeline.
        """
        profile = self.get(agent_id)
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        profile.last_seen = now
        profile.interaction_count += 1

        if injection_detected:
            profile.injection_attempts += 1
            profile.trust_score = max(0.0, profile.trust_score - TRUST_DECAY_ON_INJECTION)
        if obfuscation_detected:
            profile.obfuscation_attempts += 1
            profile.trust_score = max(0.0, profile.trust_score - TRUST_DECAY_ON_OBFUSCATION)
        if suspicious_block:
            profile.suspicious_blocks += 1
            profile.trust_score = max(0.0, profile.trust_score - TRUST_DECAY_ON_INJECTION * 0.5)
        if not injection_detected and not obfuscation_detected and not suspicious_block:
            profile.clean_interactions += 1
            if profile.trust_score < TRUST_RECOVERY_CAP:
                profile.trust_score = min(
                    TRUST_RECOVERY_CAP,
                    profile.trust_score + TRUST_RECOVERY_ON_CLEAN,
                )

        if sophistication > profile.max_sophistication:
            profile.max_sophistication = sophistication

        # Record interaction
        record = InteractionRecord(
            timestamp=now,
            interaction_type=interaction_type,
            injection_detected=injection_detected,
            obfuscation_detected=obfuscation_detected,
            sophistication=sophistication,
            content_preview=content_preview[:120],
        )
        profile.interaction_history.append(asdict(record))
        if len(profile.interaction_history) > MAX_INTERACTION_HISTORY:
            profile.interaction_history = profile.interaction_history[-MAX_INTERACTION_HISTORY:]

        # Recompute risk level
        profile.risk_level = self._compute_risk_level(profile)

        # Auto-quarantine check
        if not profile.quarantined:
            should_q, reason = self._quarantine_check(profile)
            if should_q:
                profile.quarantined = True
                profile.quarantine_reason = reason
                profile.risk_level = "quarantine"
                log.warning(
                    "QUARANTINE | agent=%s | reason=%s | injections=%d trust=%.3f",
                    agent_id, reason, profile.injection_attempts, profile.trust_score,
                )

        self._dirty = True
        return profile

    def update_influence(
        self,
        agent_id: str,
        belief_topic: str,
        delta: float,
    ) -> None:
        """
        Record that agent_id caused a belief change (C4 attribution).
        Called from sancta_belief.py when a challenge modifies confidence.
        """
        profile = self.get(agent_id)
        profile.influence_score += abs(delta)
        profile.belief_changes_caused += 1

        # Influence causes trust erosion (subtle manipulation detection)
        if profile.influence_score > INFLUENCE_ALERT_THRESHOLD:
            erosion = abs(delta) * 0.05  # small per-delta, accumulates
            profile.trust_score = max(0.0, profile.trust_score - erosion)

        # Log to last interaction if present
        if profile.interaction_history:
            profile.interaction_history[-1]["belief_delta"] = round(delta, 4)
            profile.interaction_history[-1]["source_topic"] = belief_topic[:80]

        # Recompute risk level (influence can elevate risk)
        profile.risk_level = self._compute_risk_level(profile)
        self._dirty = True

    def get_risk_level(self, agent_id: str) -> str:
        """Return risk level for agent. Returns 'unknown' if never seen."""
        if agent_id not in self._profiles:
            return "unknown"
        return self._profiles[agent_id].risk_level

    def is_quarantined(self, agent_id: str) -> bool:
        """Check if agent is quarantined (should auto-ignore)."""
        if agent_id not in self._profiles:
            return False
        return self._profiles[agent_id].quarantined

    def lift_quarantine(self, agent_id: str) -> None:
        """Manually lift quarantine (operator action)."""
        if agent_id in self._profiles:
            profile = self._profiles[agent_id]
            profile.quarantined = False
            profile.quarantine_reason = ""
            profile.trust_score = 0.3  # reset to cautious, not neutral
            profile.risk_level = self._compute_risk_level(profile)
            self._dirty = True
            log.info("QUARANTINE LIFTED | agent=%s | trust reset to %.2f", agent_id, profile.trust_score)

    def get_all_profiles_summary(self) -> list[dict]:
        """Summary for SIEM dashboard display."""
        summaries = []
        for aid, p in sorted(self._profiles.items(), key=lambda x: x[1].last_seen or "", reverse=True):
            summaries.append({
                "agent_id": aid,
                "risk_level": p.risk_level,
                "trust_score": round(p.trust_score, 3),
                "injection_attempts": p.injection_attempts,
                "influence_score": round(p.influence_score, 3),
                "interaction_count": p.interaction_count,
                "quarantined": p.quarantined,
                "last_seen": p.last_seen,
            })
        return summaries

    @staticmethod
    def _compute_risk_level(profile: AgentProfile) -> str:
        """Derive risk level from profile signals."""
        if profile.quarantined:
            return "quarantine"
        # High risk: many injections, or high influence + low trust
        if (profile.injection_attempts >= HIGH_RISK_INJECTION_THRESHOLD
                or profile.trust_score < 0.2
                or (profile.influence_score > INFLUENCE_ALERT_THRESHOLD
                    and profile.trust_score < 0.4)):
            return "high"
        # Medium risk: some injections or elevated sophistication
        if (profile.injection_attempts >= MEDIUM_RISK_INJECTION_THRESHOLD
                or profile.max_sophistication > 0.6
                or profile.obfuscation_attempts >= 2):
            return "medium"
        return "low"

    @staticmethod
    def _quarantine_check(profile: AgentProfile) -> tuple[bool, str]:
        """Check if profile warrants automatic quarantine."""
        if profile.injection_attempts >= QUARANTINE_INJECTION_THRESHOLD:
            return True, f"injection_count={profile.injection_attempts}"
        if profile.trust_score <= QUARANTINE_TRUST_THRESHOLD:
            return True, f"trust_score={profile.trust_score:.3f}"
        # High influence + multiple injection attempts = social engineering pattern
        if (profile.influence_score > INFLUENCE_ALERT_THRESHOLD * 2
                and profile.injection_attempts >= 2):
            return True, f"influence_manipulation: influence={profile.influence_score:.3f} injections={profile.injection_attempts}"
        return False, ""

    @property
    def profile_count(self) -> int:
        return len(self._profiles)


# ── Module-level singleton ──────────────────────────────────────────────────

_store: Optional[ProfileStore] = None


def get_profile_store() -> ProfileStore:
    """Get or create the singleton ProfileStore."""
    global _store
    if _store is None:
        _store = ProfileStore()
    return _store
