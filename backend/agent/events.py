from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping

import logging


log = logging.getLogger("soul.notifications")


class EventCategory(str, Enum):
    """
    CESP-inspired event categories plus Sancta-specific ones.

    Values are the canonical string identifiers used in config and manifests.
    """

    SESSION_START = "session.start"
    SESSION_END = "session.end"

    TASK_ACK = "task.acknowledge"
    TASK_COMPLETE = "task.complete"
    TASK_ERROR = "task.error"

    INPUT_REQUIRED = "input.required"
    RESOURCE_LIMIT = "resource.limit"
    USER_SPAM = "user.spam"

    SECURITY_ALERT = "security.alert"
    REDTEAM_ALERT = "redteam.alert"
    HEARTBEAT_FAILURE = "heartbeat.failure"


@dataclass(frozen=True)
class Event:
    """
    Lightweight notification event passed to the notifications subsystem.
    """

    category: EventCategory
    summary: str = ""
    details: Mapping[str, Any] | None = None
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def notify(
    category: EventCategory | str,
    summary: str = "",
    details: Mapping[str, Any] | None = None,
    *,
    silent: bool = False,
) -> None:
    """
    Fire a notification event.

    This is the public API used by Sancta and the SIEM dashboard. It is
    intentionally best-effort: failures are logged but never raise.

    When silent=True, skips sound/desktop notifications (used when SIEM
    processes JSONL events to avoid pygame crashes on Windows).
    """

    try:
        if isinstance(category, str):
            try:
                cat = EventCategory(category)
            except ValueError:
                log.warning("Unknown notification category: %s", category)
                return
        else:
            cat = category

        if silent:
            return

        ev = Event(category=cat, summary=summary or "", details=details)

        # Import here to avoid circular imports at module load time.
        try:
            from notifications import handle_event  # type: ignore[import]
        except Exception:
            # Notifications are optional; missing module should not break core.
            log.debug("notifications module not available; skipping event")
            return

        try:
            handle_event(ev)
        except Exception:
            log.exception("Failed to handle notification event", extra={"event": cat.value})
    except Exception:
        # Absolute last-resort guardrail; nothing here should ever crash the agent.
        log.exception("Unexpected error in notify()")


