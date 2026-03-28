from __future__ import annotations

from pathlib import Path

from sancta_events import Event, EventCategory
from notifications import SOUNDS_DIR, get_config, handle_event


def test_config_loads_defaults() -> None:
    cfg = get_config()
    assert isinstance(cfg.enabled, bool)
    assert 0.0 <= cfg.volume <= 1.0
    # A couple of key categories should always be present.
    for cat in (
        EventCategory.SESSION_START.value,
        EventCategory.TASK_COMPLETE.value,
        EventCategory.TASK_ERROR.value,
    ):
        assert cat in cfg.categories


def test_preview_does_not_crash(tmp_path: Path, monkeypatch) -> None:
    """
    Smoke-test that handle_event() does not raise even when sounds are missing.
    """

    # Point SOUNDS_DIR at an empty temp directory so we don't depend on real files.
    monkeypatch.setattr("notifications.SOUNDS_DIR", tmp_path)

    ev = Event(category=EventCategory.TASK_COMPLETE, summary="test")
    handle_event(ev)

