from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from notifications import HOME_CONFIG_DIR, HOME_CONFIG_PATH, LOCAL_CONFIG_PATH, NotificationConfig, get_config


def _save_config(cfg: NotificationConfig, target: Path | None = None) -> None:
    data: dict[str, Any] = {
        "enabled": cfg.enabled,
        "desktop_notifications": cfg.desktop_notifications,
        "volume": float(cfg.volume),
        "categories": dict(cfg.categories),
        "default_pack": cfg.default_pack,
    }
    target = target or HOME_CONFIG_PATH
    if target == HOME_CONFIG_PATH:
        HOME_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Control Sancta notification settings (peon-ping style).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="Show current notification settings")

    sub.add_parser("pause", help="Disable all sounds (enabled = false)")
    sub.add_parser("resume", help="Enable all sounds (enabled = true)")

    sub.add_parser("notifications-on", help="Enable desktop notifications")
    sub.add_parser("notifications-off", help="Disable desktop notifications")

    p_vol = sub.add_parser("volume", help="Get or set volume (0.0–1.0)")
    p_vol.add_argument("value", type=float, nargs="?", help="New volume (0.0–1.0)")

    sub.add_parser("preview", help="Play one sound per enabled category")

    args = parser.parse_args()
    cfg = get_config()

    # Prefer home config by default; fall back to local project config only if it exists.
    target = HOME_CONFIG_PATH if HOME_CONFIG_PATH.exists() or not LOCAL_CONFIG_PATH.exists() else LOCAL_CONFIG_PATH

    if args.cmd == "status":
        print("Notifications config:", target)
        print("  enabled:", cfg.enabled)
        print("  desktop_notifications:", cfg.desktop_notifications)
        print("  volume:", cfg.volume)
        print("  default_pack:", cfg.default_pack)
        print("  categories:")
        for k, v in sorted(cfg.categories.items()):
            print(f"    {k}: {v}")
        return

    if args.cmd == "pause":
        cfg.enabled = False
        _save_config(cfg, target)
        print("Notifications paused (enabled = false)")
        return

    if args.cmd == "resume":
        cfg.enabled = True
        _save_config(cfg, target)
        print("Notifications resumed (enabled = true)")
        return

    if args.cmd == "notifications-on":
        cfg.desktop_notifications = True
        _save_config(cfg, target)
        print("Desktop notifications enabled")
        return

    if args.cmd == "notifications-off":
        cfg.desktop_notifications = False
        _save_config(cfg, target)
        print("Desktop notifications disabled")
        return

    if args.cmd == "volume":
        if args.value is None:
            print("Current volume:", cfg.volume)
            return
        v = max(0.0, min(1.0, float(args.value)))
        cfg.volume = v
        _save_config(cfg, target)
        print("Volume set to", v)
        return

    if args.cmd == "preview":
        from notifications import preview_all

        preview_all()
        return


if __name__ == "__main__":
    main()

