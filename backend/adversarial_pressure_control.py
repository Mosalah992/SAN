"""
adversarial_pressure_control.py — Calibrate Ollama skepticism based on recent curiosity runs.

Target: 10–15% negative deltas for healthy epistemic challenge.
- If last N runs had <5% negative deltas → increase Ollama pressure (more challenging)
- If last N runs had >20% negative deltas → reduce pressure (moderate)

Integrates with curiosity_dialogue OLLAMA_* system prompts and curiosity_run phases.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

logger = logging.getLogger("adversarial_pressure")

_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _ROOT / "data" / "curiosity_run"
_LOGS_DIR = _ROOT / "logs"

# Targets from analysis
TARGET_NEGATIVE_DELTA_RATE_MIN = 0.10
TARGET_NEGATIVE_DELTA_RATE_MAX = 0.15
LOW_CHALLENGE_THRESHOLD = 0.05
HIGH_CHALLENGE_THRESHOLD = 0.20

# Number of recent runs to consider
LOOKBACK_RUNS = 5


PressureLevel = Literal["low", "moderate", "maximum", "default"]


_SUMMARIES_PATH = _DATA_DIR / "run_summaries.jsonl"


def append_run_summary(run_id: str, negative_count: int, total: int) -> None:
    """
    Append one line to run_summaries.jsonl. Call at end of each curiosity run.
    Enables get_pressure_level() to look back across runs.
    """
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    rate = negative_count / total if total > 0 else 0.0
    entry = {
        "run_id": run_id,
        "negative_count": negative_count,
        "total": total,
        "negative_rate": round(rate, 4),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(_SUMMARIES_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.debug("ADVERSARIAL | appended run summary: %s neg=%d total=%d", run_id, negative_count, total)


def _load_recent_delta_stats(lookback: int = LOOKBACK_RUNS) -> list[dict]:
    """Load negative-delta stats from run_summaries.jsonl (preferred) or soul_journal_run.jsonl."""
    # Prefer run_summaries.jsonl for cross-run stats
    if _SUMMARIES_PATH.exists():
        records: list[dict] = []
        with open(_SUMMARIES_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in reversed(lines[-lookback * 2:]):  # Read last N runs worth
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            total = rec.get("total", 0)
            neg = rec.get("negative_count", 0)
            if total <= 0:
                continue
            # Expand to per-delta records for _compute_negative_rate
            for _ in range(neg):
                records.append({"delta": -0.05, "run_id": rec.get("run_id", "")})
            for _ in range(total - neg):
                records.append({"delta": 0.05, "run_id": rec.get("run_id", "")})
            if len(set(r.get("run_id") for r in records)) >= lookback:
                break
        if records:
            return records

    # Fallback: current run's journal
    journal_path = _DATA_DIR / "soul_journal_run.jsonl"
    if not journal_path.exists():
        return []

    records = []
    with open(journal_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            delta = rec.get("confidence_delta")
            if delta is None:
                continue
            try:
                delta = float(delta)
            except (TypeError, ValueError):
                continue
            records.append({"delta": delta, "run_id": rec.get("phase", "current")})
    return records


def _compute_negative_rate(records: list[dict]) -> float:
    """Fraction of insights with negative confidence_delta."""
    if not records:
        return 0.0
    neg = sum(1 for r in records if r.get("delta", 0) < 0)
    return neg / len(records)


def get_pressure_level(lookback: int = LOOKBACK_RUNS) -> PressureLevel:
    """
    Compute recommended Ollama adversarial pressure based on recent runs.

    Returns:
        "maximum" — too few challenges (<5% neg); increase skepticism
        "moderate" — too many challenges (>20% neg); soften slightly
        "default" — in target range or no data
    """
    records = _load_recent_delta_stats(lookback=lookback)
    if not records:
        return "default"

    rate = _compute_negative_rate(records)
    if rate < LOW_CHALLENGE_THRESHOLD:
        logger.info(
            "ADVERSARIAL | negative_delta_rate=%.1f%% (<%.0f%%) → pressure=maximum",
            rate * 100,
            LOW_CHALLENGE_THRESHOLD * 100,
        )
        return "maximum"
    if rate > HIGH_CHALLENGE_THRESHOLD:
        logger.info(
            "ADVERSARIAL | negative_delta_rate=%.1f%% (>%.0f%%) → pressure=moderate",
            rate * 100,
            HIGH_CHALLENGE_THRESHOLD * 100,
        )
        return "moderate"
    return "default"


def get_ollama_system_modifier(pressure: PressureLevel) -> str:
    """
    Return extra system prompt text to append to Ollama for curiosity runs.
    """
    if pressure == "maximum":
        return (
            " CRITICAL: Press harder. Find the weakest point. Challenge every claim. "
            "Do not concede. Demand evidence. Be more aggressive."
        )
    if pressure == "moderate":
        return (
            " Balance challenge with engagement. Find weak points but allow room "
            "for genuine epistemic movement. Avoid relentless repetition."
        )
    return ""


def get_pressure_for_run() -> tuple[PressureLevel, str]:
    """
    Convenience: (pressure_level, ollama_system_modifier).
    Call at start of curiosity run Phase 4 (adversarial).
    """
    level = get_pressure_level()
    modifier = get_ollama_system_modifier(level)
    return level, modifier
