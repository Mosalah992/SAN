from __future__ import annotations

import json
import logging
import os
import random
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping

from sancta_events import Event, EventCategory


log = logging.getLogger("soul.notifications")


_ROOT = Path(__file__).resolve().parent.parent
ROOT = _ROOT
HOME_CONFIG_DIR = Path.home() / ".sancta"
HOME_CONFIG_PATH = HOME_CONFIG_DIR / "notifications.json"
LOCAL_CONFIG_PATH = _ROOT / "notifications.json"

def _sounds_dir() -> Path:
    """Sound pack directory. Override via SANCTA_SOUNDS_DIR env."""
    override = os.environ.get("SANCTA_SOUNDS_DIR", "").strip()
    if override and Path(override).is_dir():
        return Path(override).resolve()
    return _ROOT / "frontend" / "sounds"


SOUNDS_DIR = _ROOT / "frontend" / "sounds"  # Fallback for manifest path
MANIFEST_PATH = SOUNDS_DIR / "manifest.json"


@dataclass
class NotificationConfig:
    enabled: bool = True
    desktop_notifications: bool = True
    volume: float = 0.7
    categories: Dict[str, bool] = field(
        default_factory=lambda: {
            EventCategory.SESSION_START.value: True,
            EventCategory.SESSION_END.value: True,
            EventCategory.TASK_ACK.value: False,
            EventCategory.TASK_COMPLETE.value: True,
            EventCategory.TASK_ERROR.value: True,
            EventCategory.INPUT_REQUIRED.value: True,
            EventCategory.RESOURCE_LIMIT.value: True,
            EventCategory.USER_SPAM.value: True,
            EventCategory.SECURITY_ALERT.value: True,
            EventCategory.REDTEAM_ALERT.value: True,
            EventCategory.HEARTBEAT_FAILURE.value: True,
        }
    )
    default_pack: str = "office_peon"


_CONFIG: NotificationConfig | None = None
_MANIFEST: Dict[str, Any] | None = None
_LAST_PLAYED: Dict[tuple[str, str], str] = {}
_LOCK = threading.Lock()


def _load_json(path: Path) -> Dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        log.exception("Failed to read JSON config: %s", path)
        return None


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def get_config() -> NotificationConfig:
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG

    data = _load_json(HOME_CONFIG_PATH) or _load_json(LOCAL_CONFIG_PATH) or {}

    cfg = NotificationConfig()
    cfg.enabled = bool(data.get("enabled", cfg.enabled))
    cfg.desktop_notifications = bool(
        data.get("desktop_notifications", cfg.desktop_notifications)
    )
    cfg.volume = float(data.get("volume", cfg.volume))

    categories = dict(cfg.categories)
    for key, val in data.get("categories", {}).items():
        try:
            # Only accept known categories to avoid silent typos.
            EventCategory(key)
            categories[key] = bool(val)
        except ValueError:
            continue
    cfg.categories = categories

    cfg.default_pack = str(data.get("default_pack", cfg.default_pack))

    # Environment overrides (simple but handy).
    cfg.enabled = _env_bool("SANCTA_NOTIFY_ENABLED", cfg.enabled)
    cfg.desktop_notifications = _env_bool(
        "SANCTA_NOTIFY_DESKTOP", cfg.desktop_notifications
    )
    cfg.volume = _env_float("SANCTA_NOTIFY_VOLUME", cfg.volume)

    _CONFIG = cfg
    return cfg


def _load_manifest() -> Dict[str, Any]:
    global _MANIFEST
    if _MANIFEST is not None:
        return _MANIFEST

    data = _load_json(MANIFEST_PATH) or {}
    packs = data.get("packs") or {}
    default_pack = data.get("default_pack") or "office_peon"
    if not isinstance(packs, dict):
        packs = {}
    _MANIFEST = {"packs": packs, "default_pack": str(default_pack)}
    return _MANIFEST


def _select_sound(event: Event, cfg: NotificationConfig) -> Path | None:
    manifest = _load_manifest()
    packs = manifest.get("packs", {})
    if not packs:
        return None

    pack_name = cfg.default_pack
    if pack_name not in packs:
        # Fallback to first available pack.
        pack_name = next(iter(packs.keys()))

    cat_key = event.category.value
    pack = packs.get(pack_name, {})
    sounds = pack.get(cat_key) or []
    if not sounds:
        return None

    # Avoid repeating the same sound consecutively per (pack, category).
    key = (pack_name, cat_key)
    last = _LAST_PLAYED.get(key)
    candidates = [s for s in sounds if s != last] or sounds

    choice = random.choice(candidates)
    _LAST_PLAYED[key] = choice

    path = _sounds_dir() / choice
    return path


_pygame_mixer_ready = False


def _ensure_pygame_mixer() -> bool:
    global _pygame_mixer_ready
    if _pygame_mixer_ready:
        return True
    try:
        import pygame  # type: ignore[import]
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
        _pygame_mixer_ready = True
        return True
    except Exception:
        log.debug("pygame.mixer init failed", exc_info=True)
        return False


def _play_sound(path: Path, volume: float) -> None:
    """Play optional sound. Skip silently if SANCTA_SOUNDS_ENABLED=false or file missing."""
    if not _env_bool("SANCTA_SOUNDS_ENABLED", True):
        return
    try:
        if not path.exists():
            # Log once per path per session to avoid spam (optional asset)
            if path not in getattr(_play_sound, "_logged_missing", set()):
                _logged = getattr(_play_sound, "_logged_missing", set())
                _logged.add(path)
                _play_sound._logged_missing = _logged  # type: ignore[attr-defined]
                log.debug("Sound file not found (optional): %s", path)
            return

        # Primary: pygame.mixer handles MP3, WAV, OGG on all platforms.
        if _ensure_pygame_mixer():
            try:
                import pygame  # type: ignore[import]
                pygame.mixer.music.load(str(path))
                pygame.mixer.music.set_volume(max(0.0, min(1.0, volume)))
                pygame.mixer.music.play()
                return
            except Exception:
                log.debug("pygame playback failed, trying fallbacks", exc_info=True)

        # Windows fallback: winsound only supports PCM WAV.
        if sys.platform.startswith("win") and path.suffix.lower() == ".wav":
            try:
                import winsound  # type: ignore[import]

                winsound.PlaySound(
                    str(path),
                    winsound.SND_FILENAME | winsound.SND_ASYNC,
                )
                return
            except Exception:
                log.debug("winsound playback failed", exc_info=True)

        # CLI fallback: try common external players.
        candidates: list[tuple[str, list[str]]] = [
            ("afplay", ["afplay", str(path)]),
            ("pw-play", ["pw-play", str(path)]),
            ("paplay", ["paplay", str(path)]),
            ("ffplay", ["ffplay", "-nodisp", "-autoexit", str(path)]),
            ("mpv", ["mpv", "--no-video", "--really-quiet", str(path)]),
            ("aplay", ["aplay", str(path)]),
            ("play", ["play", "-q", str(path)]),
        ]

        for name, cmd in candidates:
            if shutil.which(name):
                try:
                    subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return
                except Exception:
                    continue

        log.debug("No suitable audio player found; skipping sound")
    except Exception:
        log.exception("Error while trying to play sound: %s", path)


def _show_desktop_notification(title: str, message: str) -> None:
    try:
        # Prefer plyer if available.
        try:
            from plyer import notification  # type: ignore[import]

            notification.notify(
                title=title,
                message=message,
                app_name="Sancta",
                timeout=5,
            )
            return
        except Exception:
            pass

        # Fallbacks per platform, best-effort only.
        if sys.platform == "darwin":
            script = f'display notification "{message}" with title "{title}"'
            subprocess.Popen(
                ["osascript", "-e", script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return

        if sys.platform.startswith("linux"):
            if shutil.which("notify-send"):
                subprocess.Popen(
                    ["notify-send", title, message],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            return

        # Windows: no built-in CLI notification mechanism; rely on plyer if present.
    except Exception:
        log.exception("Failed to show desktop notification")


def _format_notification_text(event: Event) -> tuple[str, str]:
    cat = event.category.value
    title = "Sancta"
    if cat == EventCategory.SESSION_START.value:
        title = "Sancta ready"
    elif cat == EventCategory.SESSION_END.value:
        title = "Sancta session ended"
    elif cat in {
        EventCategory.TASK_ERROR.value,
        EventCategory.SECURITY_ALERT.value,
        EventCategory.REDTEAM_ALERT.value,
        EventCategory.HEARTBEAT_FAILURE.value,
    }:
        title = "Sancta alert"
    elif cat == EventCategory.TASK_COMPLETE.value:
        title = "Sancta task complete"

    message = event.summary or cat
    return title, message


def handle_event(event: Event) -> None:
    """
    Entry point called by sancta_events.notify().

    This function is intentionally non-blocking: playback and notifications
    are dispatched on a background thread so they never slow down agent logic.
    """

    cfg = get_config()
    if not cfg.enabled:
        return

    if not cfg.categories.get(event.category.value, True):
        return

    def _worker() -> None:
        try:
            sound_path = _select_sound(event, cfg)
            if sound_path is not None:
                _play_sound(sound_path, cfg.volume)

            if cfg.desktop_notifications:
                title, message = _format_notification_text(event)
                _show_desktop_notification(title, message)
        except Exception:
            log.exception("Error in notification worker for %s", event.category.value)

    with _LOCK:
        t = threading.Thread(target=_worker, name="sancta-notify", daemon=True)
        t.start()


def preview_all() -> None:
    """
    Best-effort preview: play one sound per configured category.
    """

    cfg = get_config()
    manifest = _load_manifest()
    packs = manifest.get("packs", {})
    if not packs:
        print("No sound packs configured under", SOUNDS_DIR)
        return

    print("Sancta notification preview:")
    for key, enabled in sorted(cfg.categories.items()):
        if not enabled:
            continue
        try:
            cat = EventCategory(key)
        except ValueError:
            continue

        ev = Event(category=cat, summary=f"Preview for {key}")

        # Only preview categories that actually have a sound configured
        # for the current/default pack.
        if _select_sound(ev, cfg) is None:
            continue

        print(" -", key)
        handle_event(ev)


