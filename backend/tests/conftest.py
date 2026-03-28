"""
conftest.py — Shared pytest fixtures for Sancta backend tests
"""

import sys
from pathlib import Path

import pytest

# Ensure backend/ is on sys.path so `from sancta_risk import ...` etc. work
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


@pytest.fixture()
def tmp_state():
    """Return a fresh dict with typical agent state keys."""
    return {
        "cycle_count": 10,
        "belief_system": {},
        "hot_topics": ["prompt_injection", "behavioral_drift"],
        "decision_mood": {"energy": 0.8, "patience": 0.7},
        "recent_positive_engagement": [],
        "recent_rejections": [],
        "last_ingested_trust_level": "high",
    }


@pytest.fixture()
def tmp_profiles_path(tmp_path):
    """Return a temp path for agent_profiles.json (file does not yet exist)."""
    return tmp_path / "agent_profiles.json"
