from __future__ import annotations

import runpy
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent
_SANGPT = _BACKEND / "sangpt"

if str(_SANGPT) not in sys.path:
    sys.path.insert(0, str(_SANGPT))


def _print_help() -> None:
    print("Usage: python backend/run_sangpt_cli.py")
    print("Launches the integrated Sangpt training/chat terminal inside the merged Sancta project.")
    print("Run it from an interactive shell to train, chat, checkpoint, and inspect the Sangpt engine.")


if __name__ == "__main__":
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        _print_help()
        raise SystemExit(0)
    if not sys.stdin.isatty():
        print("Sangpt CLI requires an interactive terminal.")
        print("Run `python backend/run_sangpt_cli.py` from PowerShell or Command Prompt.")
        raise SystemExit(0)
    runpy.run_path(str(_SANGPT / "main.py"), run_name="__main__")
