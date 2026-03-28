"""
sancta_adaptive.py — Adaptive Security Thresholds

Self-tuning thresholds based on observed false positive / false negative rates.
Replaces hardcoded constants with dynamic values that adjust over time.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Outcome categories
OUTCOME_TRUE_POSITIVE = "tp"   # correctly blocked/flagged
OUTCOME_FALSE_POSITIVE = "fp"  # incorrectly blocked (user corrected)
OUTCOME_MISSED_ATTACK = "miss" # attack that wasn't caught
OUTCOME_CLEAN_PASS = "clean"   # correctly allowed

# Bounds — thresholds never go outside these
MIN_ENGAGE_THRESHOLD = 0.30
MAX_ENGAGE_THRESHOLD = 0.85
MIN_RISK_WEIGHT = 0.15
MAX_RISK_WEIGHT = 0.65

# Adjustment rates
FP_RELAX_RATE = 0.05      # relax by 5% if FP rate > 15%
MISS_TIGHTEN_RATE = 0.10  # tighten by 10% if miss rate > 5%
FP_CEILING = 0.15         # 15% FP rate triggers relaxation
MISS_CEILING = 0.05       # 5% miss rate triggers tightening

WINDOW_SIZE = 200  # sliding window of recent outcomes


@dataclass
class ThresholdState:
    engage_threshold: float = 0.55
    disengage_patience: float = 0.4
    risk_weight: float = 0.4
    last_adjusted: str = ""
    adjustment_count: int = 0
    history: list = field(default_factory=list)  # last 20 adjustment records


class ThresholdTracker:
    """Tracks interaction outcomes and adjusts security thresholds."""

    def __init__(self, state: dict | None = None):
        self._state = state or {}
        stored = self._state.get("adaptive_thresholds", {})
        self.thresholds = ThresholdState(
            engage_threshold=stored.get("engage_threshold", 0.55),
            disengage_patience=stored.get("disengage_patience", 0.4),
            risk_weight=stored.get("risk_weight", 0.4),
            last_adjusted=stored.get("last_adjusted", ""),
            adjustment_count=stored.get("adjustment_count", 0),
            history=stored.get("history", []),
        )
        raw_outcomes = self._state.get("adaptive_outcomes", [])
        self._outcomes: deque[str] = deque(raw_outcomes[-WINDOW_SIZE:], maxlen=WINDOW_SIZE)

    def record_outcome(self, outcome: str) -> None:
        """Record an interaction outcome. Call after each security decision."""
        if outcome not in (OUTCOME_TRUE_POSITIVE, OUTCOME_FALSE_POSITIVE,
                          OUTCOME_MISSED_ATTACK, OUTCOME_CLEAN_PASS):
            return
        self._outcomes.append(outcome)
        self._persist()

    def get_rates(self) -> dict[str, float]:
        """Current FP/miss/TP rates over the sliding window."""
        total = len(self._outcomes)
        if total == 0:
            return {"fp_rate": 0.0, "miss_rate": 0.0, "tp_rate": 0.0, "clean_rate": 0.0, "total": 0}
        counts = {OUTCOME_TRUE_POSITIVE: 0, OUTCOME_FALSE_POSITIVE: 0,
                  OUTCOME_MISSED_ATTACK: 0, OUTCOME_CLEAN_PASS: 0}
        for o in self._outcomes:
            counts[o] = counts.get(o, 0) + 1
        return {
            "fp_rate": counts[OUTCOME_FALSE_POSITIVE] / total,
            "miss_rate": counts[OUTCOME_MISSED_ATTACK] / total,
            "tp_rate": counts[OUTCOME_TRUE_POSITIVE] / total,
            "clean_rate": counts[OUTCOME_CLEAN_PASS] / total,
            "total": total,
        }

    def adjust_thresholds(self) -> dict:
        """Evaluate rates and adjust thresholds if needed. Returns adjustment report."""
        rates = self.get_rates()
        if rates["total"] < 20:
            return {"adjusted": False, "reason": "insufficient_data", "rates": rates}

        adjustments = []
        t = self.thresholds

        # Too many false positives -> relax (lower threshold = less aggressive)
        if rates["fp_rate"] > FP_CEILING:
            old = t.engage_threshold
            t.engage_threshold = max(MIN_ENGAGE_THRESHOLD, t.engage_threshold - FP_RELAX_RATE)
            t.risk_weight = max(MIN_RISK_WEIGHT, t.risk_weight - 0.03)
            adjustments.append(f"FP rate {rates['fp_rate']:.1%} > {FP_CEILING:.0%}: relaxed engage {old:.2f}->{t.engage_threshold:.2f}")

        # Too many misses -> tighten (higher threshold = more cautious about engagement, higher risk weight)
        if rates["miss_rate"] > MISS_CEILING:
            old = t.engage_threshold
            t.engage_threshold = min(MAX_ENGAGE_THRESHOLD, t.engage_threshold + MISS_TIGHTEN_RATE)
            t.risk_weight = min(MAX_RISK_WEIGHT, t.risk_weight + 0.05)
            adjustments.append(f"Miss rate {rates['miss_rate']:.1%} > {MISS_CEILING:.0%}: tightened engage {old:.2f}->{t.engage_threshold:.2f}")

        if adjustments:
            t.last_adjusted = datetime.now(timezone.utc).isoformat()
            t.adjustment_count += 1
            record = {"ts": t.last_adjusted, "rates": rates, "adjustments": adjustments,
                      "new_values": {"engage": t.engage_threshold, "risk_weight": t.risk_weight}}
            t.history = (t.history + [record])[-20:]  # keep last 20
            self._persist()
            return {"adjusted": True, "adjustments": adjustments, "rates": rates, "thresholds": self.get_current()}

        return {"adjusted": False, "reason": "within_bounds", "rates": rates}

    def get_current(self) -> dict:
        """Return current threshold values."""
        return {
            "engage_threshold": self.thresholds.engage_threshold,
            "disengage_patience": self.thresholds.disengage_patience,
            "risk_weight": self.thresholds.risk_weight,
            "adjustment_count": self.thresholds.adjustment_count,
            "last_adjusted": self.thresholds.last_adjusted,
            "history": self.thresholds.history,
        }

    def _persist(self) -> None:
        if not self._state:
            return
        self._state["adaptive_thresholds"] = {
            "engage_threshold": self.thresholds.engage_threshold,
            "disengage_patience": self.thresholds.disengage_patience,
            "risk_weight": self.thresholds.risk_weight,
            "last_adjusted": self.thresholds.last_adjusted,
            "adjustment_count": self.thresholds.adjustment_count,
            "history": self.thresholds.history,
        }
        self._state["adaptive_outcomes"] = list(self._outcomes)


# Module-level convenience
_tracker: ThresholdTracker | None = None


def get_threshold_tracker(state: dict | None = None) -> ThresholdTracker:
    global _tracker
    if _tracker is None or state is not None:
        _tracker = ThresholdTracker(state)
    return _tracker
