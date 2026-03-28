"""
sancta_launcher.py — Sancta Control Center v3.0

Manages:
  - Ollama (connect to existing or start if needed)
  - SIEM dashboard server (backend.siem_server via uvicorn)
  - Sancta main agent loop (sancta.py)
  - Curiosity run (on-demand)
  - Phenomenology battery (on-demand)

New in v3.0+:
  - Per-process START / STOP for Ollama, SIEM, Sancta; STOP ALL stops Ollama (:11434 listener)
  - Sancta PID written to .agent.pid (same as SIEM Control tab)
  - Multi-source log panel: tails security.jsonl, red_team.jsonl,
    behavioral.jsonl, agent_activity.log in real-time
  - Filter dropdown: ALL | SANCTA | SIEM | OLLAMA | CURIOSITY |
    BEHAVIORAL | SECURITY | REDTEAM | ACTIVITY

Build to exe:
  pip install pyinstaller
  pyinstaller sancta_launcher.spec

Go port (GUI + embedded CLI tab): see tools/sancta-launcher/ — requires Go + CGO for Fyne.
"""

import tkinter as tk
from tkinter import font as tkfont
import shutil
import subprocess
import threading
import time
import sys
import os
import signal
import json
import webbrowser
import requests
import queue
import re
from datetime import datetime
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────────

OLLAMA_URL   = "http://127.0.0.1:11434"
SIEM_URL     = "http://127.0.0.1:8787"
OLLAMA_MODEL = "llama3.2"

def _get_paths() -> tuple[Path, Path]:
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        if (exe_dir.parent / "siem_server.py").exists():
            backend = exe_dir.parent
        elif (exe_dir.parent / "backend" / "siem_server.py").exists():
            backend = exe_dir.parent / "backend"
        else:
            backend = exe_dir.parent
        root = backend.parent if backend.name == "backend" else exe_dir.parent
        return root, backend
    _backend = Path(__file__).resolve().parent
    return _backend.parent, _backend

ROOT, BACKEND_DIR = _get_paths()
SANCTA_SCRIPT      = BACKEND_DIR / "sancta.py"
SANGPT_CLI_SCRIPT  = BACKEND_DIR / "run_sangpt_cli.py"
SANGPT_TRAIN_SCRIPT = BACKEND_DIR / "run_sancta_gpt_training.py"
CURIOSITY_FLAG     = "--curiosity-run"
PHENOMENOLOGY_FLAG = "--phenomenology-battery"
# Same file SIEM uses so Control tab + launcher stay in sync for agent state
AGENT_PID_PATH = ROOT / ".agent.pid"

def _python_exe() -> str:
    if getattr(sys, "frozen", False):
        for cand in ["python", "python3", "py"]:
            exe = shutil.which(cand)
            if exe:
                return exe
        return "python"
    return sys.executable

SIEM_CMD = [_python_exe(), "-m", "uvicorn", "backend.siem_server:app",
            "--host", "127.0.0.1", "--port", "8787"]

def _find_ollama_exe() -> str:
    """
    Resolve Ollama executable: OLLAMA_EXE env, then PATH, then common Windows install dirs.
    No machine-specific fallbacks — set OLLAMA_EXE if installs are non-standard.
    """
    cand = (os.environ.get("OLLAMA_EXE") or "").strip()
    if cand and Path(cand).exists():
        return cand
    which = shutil.which("ollama")
    if which:
        return which
    if os.name == "nt":
        for p in (
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Ollama" / "ollama.exe",
        ):
            if p.exists():
                return str(p)
    # Last resort: subprocess may resolve via shell PATH
    return "ollama"

OLLAMA_EXE = _find_ollama_exe()


def _write_agent_pid(pid: int) -> None:
    try:
        AGENT_PID_PATH.write_text(str(int(pid)), encoding="utf-8")
    except Exception:
        pass


def _clear_agent_pid_if_matches(pid: int | None) -> None:
    """Remove .agent.pid when it points at this PID (or always clear if pid is None)."""
    try:
        if not AGENT_PID_PATH.exists():
            return
        cur = AGENT_PID_PATH.read_text(encoding="utf-8").strip()
        if pid is None or cur == str(pid):
            AGENT_PID_PATH.unlink(missing_ok=True)
    except Exception:
        pass


def _kill_pid_hard(pid: int) -> bool:
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/F"],
                capture_output=True,
                timeout=8,
            )
        else:
            os.kill(pid, signal.SIGTERM)
        return True
    except Exception:
        try:
            if os.name != "nt":
                os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
        return False


def _pid_listening_on_port(port: int) -> int | None:
    """Best-effort PID listening on localhost:port (Windows netstat; Unix lsof)."""
    if os.name == "nt":
        try:
            r = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                text=True,
                timeout=20,
                encoding="utf-8",
                errors="ignore",
            )
            needle = f":{port}"
            for line in (r.stdout or "").splitlines():
                lu = line.upper()
                if "LISTENING" not in lu or needle not in line:
                    continue
                parts = line.split()
                if not parts:
                    continue
                try:
                    return int(parts[-1])
                except ValueError:
                    continue
        except Exception:
            pass
        return None
    try:
        r = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if r.returncode == 0 and (r.stdout or "").strip():
            return int((r.stdout or "").strip().split()[0])
    except Exception:
        pass
    return None


SANCTA_RESTART_DELAY = 10
MAX_LOG_LINES        = 400
POLL_INTERVAL        = 500   # ms
NET_CHECK_INTERVAL   = 3     # seconds
LOG_TAIL_INTERVAL    = 2     # seconds — log file poll cadence
REQUEST_TIMEOUT      = 1
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _sanitize_log_text(text: str) -> str:
    text = _ANSI_RE.sub("", text or "")
    cleaned = []
    for ch in text:
        if ch in ("\n", "\t", "\r") or ord(ch) >= 32:
            cleaned.append(ch)
    out = "".join(cleaned).replace("\r", " ").replace("\n", " ").strip()
    return " ".join(out.split())

# ─── Colors ──────────────────────────────────────────────────────────────────

C = {
    "bg":        "#030712",
    "bg2":       "#0a0f1e",
    "bg3":       "#111827",
    "border":    "#1e293b",
    "purple":    "#818cf8",
    "purple_dk": "#4338ca",
    "green":     "#10b981",
    "red":       "#ef4444",
    "amber":     "#f59e0b",
    "teal":      "#14b8a6",
    "magenta":   "#ec4899",
    "cyan":      "#22d3ee",
    "text":      "#e5e7eb",
    "muted":     "#6b7280",
    "dim":       "#374151",
}

# ─── State ───────────────────────────────────────────────────────────────────

processes      = {}
log_queue      = queue.Queue()
restart_counts = {}
curiosity_running = False

_net_status = {"ollama": False, "siem": False, "ollama_model": None}
_net_status_lock = threading.Lock()

# Log file tailing state — initialised after ROOT is known
LOG_FILES: dict[str, Path] = {}
_log_cursors: dict[str, int] = {}

def _init_log_files():
    global LOG_FILES, _log_cursors
    LOG_FILES = {
        "security":  ROOT / "logs" / "security.jsonl",
        "redteam":   ROOT / "logs" / "red_team.jsonl",
        "behavioral": ROOT / "logs" / "behavioral.jsonl",
        "activity":  ROOT / "logs" / "agent_activity.log",
    }
    _log_cursors = {}
    # Seek to end — don't replay old data on launch
    for src, path in LOG_FILES.items():
        try:
            _log_cursors[src] = path.stat().st_size if path.exists() else 0
        except Exception:
            _log_cursors[src] = 0

_init_log_files()

# ─── Background threads ───────────────────────────────────────────────────────

def _background_net_checker():
    while True:
        try:
            ok_ollama = is_ollama_running()
            ok_siem   = is_siem_running()
            model     = get_ollama_model() if ok_ollama else None
            with _net_status_lock:
                _net_status["ollama"]       = ok_ollama
                _net_status["siem"]         = ok_siem
                _net_status["ollama_model"] = model
        except Exception:
            pass
        time.sleep(NET_CHECK_INTERVAL)


def _tail_log_files():
    """
    Background daemon: every LOG_TAIL_INTERVAL seconds, check each log file
    for new bytes, parse and push entries to log_queue.
    """
    while True:
        for src, path in LOG_FILES.items():
            try:
                if not path.exists():
                    continue
                size = path.stat().st_size
                if size <= _log_cursors[src]:
                    continue
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(_log_cursors[src])
                    chunk = f.read(65536)  # max 64KB per cycle
                    _log_cursors[src] = f.tell()
                for raw in chunk.splitlines():
                    raw = raw.strip()
                    if not raw:
                        continue
                    if src == "activity":
                        clean = _sanitize_log_text(raw)
                        lvl = ("ERROR" if any(k in clean for k in ("ERROR", "Traceback"))
                               else "WARN" if any(k in clean for k in ("WARN", "WARNING"))
                               else "INFO")
                        log_queue.put((src, lvl, clean[:200]))
                    else:
                        try:
                            obj = json.loads(raw)
                            raw_lvl = str(obj.get("level", "")).upper()
                            lvl = ("ERROR" if raw_lvl in ("ERROR", "CRITICAL")
                                   else "WARN"  if raw_lvl in ("WARN", "WARNING")
                                   else "OK"    if (obj.get("blocked") or
                                                    obj.get("blocked_injection") or
                                                    obj.get("defended"))
                                   else "INFO")
                            msg = (obj.get("message") or obj.get("msg") or
                                   obj.get("event")   or obj.get("content") or
                                   str(obj)[:180])
                            log_queue.put((src, lvl, _sanitize_log_text(str(msg))[:200]))
                        except (json.JSONDecodeError, Exception):
                            log_queue.put((src, "INFO", _sanitize_log_text(raw)[:200]))
            except Exception:
                pass
        time.sleep(LOG_TAIL_INTERVAL)

# ─── Process management ───────────────────────────────────────────────────────

def is_ollama_running() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/version", timeout=REQUEST_TIMEOUT)
        return r.status_code == 200
    except Exception:
        return False

def is_siem_running() -> bool:
    try:
        r = requests.get(SIEM_URL, timeout=REQUEST_TIMEOUT)
        return r.status_code < 500
    except Exception:
        return False

def get_ollama_model() -> str | None:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            for m in r.json().get("models", []):
                if OLLAMA_MODEL in m["name"]:
                    return m["name"]
    except Exception:
        pass
    return None

def start_ollama() -> bool:
    if is_ollama_running():
        log_queue.put(("ollama", "INFO", "Already running — connecting"))
        return True
    log_queue.put(("ollama", "INFO", "Starting Ollama..."))
    try:
        proc = subprocess.Popen(
            [OLLAMA_EXE, "serve"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        processes["ollama"] = proc
        for _ in range(15):
            time.sleep(1)
            if is_ollama_running():
                log_queue.put(("ollama", "OK", "Ollama ready on :11434"))
                return True
        log_queue.put(("ollama", "ERROR", "Timeout waiting for Ollama"))
        return False
    except FileNotFoundError:
        log_queue.put(("ollama", "ERROR", f"Not found: {OLLAMA_EXE}"))
        return False


def stop_ollama(*, force_port: bool = True) -> None:
    """
    Terminate launcher-spawned ``ollama serve``; if still listening on 11434, kill that PID
    (covers tray-started Ollama so STOP ALL actually clears the port).
    """
    stop_process("ollama")
    time.sleep(0.4)
    if not force_port:
        return
    if not is_ollama_running():
        log_queue.put(("ollama", "INFO", "Ollama stopped"))
        return
    opid = _pid_listening_on_port(11434)
    if opid:
        log_queue.put(("ollama", "WARN", f"Stopping process on :11434 (PID {opid})"))
        _kill_pid_hard(opid)
        time.sleep(0.4)
    if is_ollama_running():
        log_queue.put(("ollama", "WARN", "Ollama API still up — close Ollama from the system tray if needed"))
    else:
        log_queue.put(("ollama", "INFO", "Ollama stopped"))


def start_process(name: str, script: Path, extra_args: list = None,
                  env_extra: dict = None, restart: bool = False,
                  args_override: list = None, cwd_override: Path = None) -> bool:
    py   = _python_exe()
    args = args_override or ([py, str(script)] + (extra_args or []))
    cwd  = str(cwd_override or BACKEND_DIR)
    env  = os.environ.copy()
    env["OLLAMA_CONTEXT_LENGTH"] = "8192"
    env["PYTHONUNBUFFERED"]      = "1"
    if env_extra:
        env.update(env_extra)
    MAX_RESTARTS = 5
    try:
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=cwd, env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        processes[name] = proc
        log_queue.put((name, "OK", f"Started (PID {proc.pid})"))
        if name == "sancta":
            _write_agent_pid(proc.pid)

        def _stream():
            for line in proc.stdout:
                line = _sanitize_log_text(line.rstrip())
                if not line:
                    continue
                lvl = ("ERROR" if any(k in line for k in ("ERROR", "Traceback"))
                       else "WARN" if any(k in line for k in ("WARNING", "WARN"))
                       else "INFO")
                log_queue.put((name, lvl, line[:200]))
            code = proc.wait()
            if name == "sancta":
                _clear_agent_pid_if_matches(proc.pid)
            log_queue.put((name, "WARN" if code != 0 else "INFO",
                           f"Exited (code {code})"))
            if restart and name in processes and not curiosity_running:
                restart_counts[name] = restart_counts.get(name, 0) + 1
                if restart_counts[name] > MAX_RESTARTS:
                    log_queue.put((name, "ERROR", f"Max restart attempts ({MAX_RESTARTS}) reached. Not restarting."))
                    return
                log_queue.put((name, "WARN",
                               f"Auto-restarting in {SANCTA_RESTART_DELAY}s... (attempt {restart_counts[name]}/{MAX_RESTARTS})"))
                time.sleep(SANCTA_RESTART_DELAY)
                if name in processes:
                    start_process(name, script, extra_args, env_extra, restart)

        threading.Thread(target=_stream, daemon=True).start()
        return True
    except Exception as e:
        log_queue.put((name, "ERROR", str(e)))
        return False

def start_interactive_console_process(name: str, script: Path, extra_args: list = None,
                                      env_extra: dict = None, cwd_override: Path = None) -> bool:
    """
    Launch an interactive process in its own console window.

    On Windows, explicitly launch a new PowerShell window for Sangpt CLI to ensure interactive input works.
    """
    py   = _python_exe()
    args = [py, str(script)] + (extra_args or [])
    cwd  = str(cwd_override or ROOT)
    env  = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if env_extra:
        env.update(env_extra)
    try:
        if sys.platform == "win32":
            # Use 'start' to launch a new console window reliably
            cmd_args = [py, str(script)] + (extra_args or [])
            cmd_str = " ".join(f'"{arg}"' for arg in cmd_args)
            powershell_cmd = [
                "cmd", "/c", "start", "powershell", "-NoExit", "-Command",
                f'cd "{cwd}"; & {cmd_str}'
            ]
            proc = subprocess.Popen(
                powershell_cmd,
                cwd=cwd,
                env=env,
                creationflags=0,
            )
        else:
            # Fallback: use CREATE_NEW_CONSOLE or default
            creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
            proc = subprocess.Popen(
                args,
                cwd=cwd,
                env=env,
                creationflags=creationflags,
            )
        processes[name] = proc
        log_queue.put((name, "OK", f"Opened interactive console (PID {proc.pid})"))

        def _watch():
            code = proc.wait()
            processes.pop(name, None)
            log_queue.put((name, "INFO", f"Interactive console exited (code {code})"))

        threading.Thread(target=_watch, daemon=True).start()
        return True
    except Exception as e:
        log_queue.put((name, "ERROR", str(e)))
        return False

def stop_process(name: str):
    proc = processes.pop(name, None)
    if not proc:
        return
    pid = proc.pid
    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        log_queue.put((name, "INFO", "Stopped"))
    if name == "sancta":
        _clear_agent_pid_if_matches(pid)

def stop_all():
    # Deterministic order: auxiliary jobs → agent → SIEM → Ollama (frees :11434)
    for name in ("phenomenology", "curiosity", "sancta", "siem"):
        stop_process(name)
    stop_ollama(force_port=True)

# ─── UI ──────────────────────────────────────────────────────────────────────

_FILTER_OPTS = [
    "ALL",
    "─ SUBPROCESS ─",
    "SANCTA", "SIEM", "OLLAMA", "CURIOSITY", "PHENOMENOLOGY",
    "─ LOG FILES ─",
    "SECURITY", "REDTEAM", "BEHAVIORAL", "ACTIVITY",
]
# Separator items (not selectable sources)
_FILTER_SEPS = {"─ SUBPROCESS ─", "─ LOG FILES ─"}

# Valid src tags for the text widget
_VALID_SRC_TAGS = {
    "src_sancta", "src_sangpt", "src_sangpt_train", "src_siem", "src_ollama",
    "src_curiosity", "src_phenomenology", "src_launcher",
    "src_security", "src_redteam", "src_behavioral", "src_activity",
}


class SanctaLauncher(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Sancta Control Center")
        self.configure(bg=C["bg"])
        self.geometry("1100x740")
        self.minsize(900, 600)
        self.resizable(True, True)

        # Fonts
        try:
            self.fn_head  = tkfont.Font(family="Consolas", size=11, weight="bold")
            self.fn_mono  = tkfont.Font(family="Consolas", size=9)
            self.fn_label = tkfont.Font(family="Consolas", size=9)
            self.fn_title = tkfont.Font(family="Consolas", size=13, weight="bold")
            self.fn_btn   = tkfont.Font(family="Consolas", size=9, weight="bold")
            self.fn_micro = tkfont.Font(family="Consolas", size=8)
        except Exception:
            self.fn_head  = tkfont.Font(size=11, weight="bold")
            self.fn_mono  = tkfont.Font(size=9)
            self.fn_label = tkfont.Font(size=9)
            self.fn_title = tkfont.Font(size=13, weight="bold")
            self.fn_btn   = tkfont.Font(size=9, weight="bold")
            self.fn_micro = tkfont.Font(size=8)

        # Status dot StringVars
        self.status_vars = {
            "ollama":        tk.StringVar(value="●"),
            "siem":          tk.StringVar(value="●"),
            "sancta":        tk.StringVar(value="●"),
            "sangpt":        tk.StringVar(value="●"),
            "sangpt_train":  tk.StringVar(value="●"),
            "curiosity":     tk.StringVar(value="●"),
            "phenomenology": tk.StringVar(value="●"),
        }
        self.status_labels = {}

        # Per-service control button refs
        self.svc_btns: dict[str, dict[str, tk.Button]] = {}

        self._start_time = time.time()
        self._build_ui()
        self._bind_close()

        # Start UI polling
        self.after(500, self._poll_status)
        self.after(100, self._drain_logs)

    # ─── UI Build ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=C["bg2"], height=52)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="⬡ SANCTA", font=self.fn_title,
                 fg=C["purple"], bg=C["bg2"]).pack(side="left", padx=20, pady=14)
        tk.Label(hdr, text="CONTROL CENTER", font=self.fn_label,
                 fg=C["muted"], bg=C["bg2"]).pack(side="left", pady=14)
        tk.Label(hdr, text="v3.0", font=self.fn_label,
                 fg=C["purple_dk"], bg=C["bg2"], padx=8, pady=2).pack(
            side="right", padx=20)
        tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

        # Main area
        main = tk.Frame(self, bg=C["bg"])
        main.pack(fill="both", expand=True)

        # ── Left panel ─────────────────────────────────────────────────────
        left = tk.Frame(main, bg=C["bg2"], width=320)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        tk.Frame(main, bg=C["border"], width=1).pack(side="left", fill="y")

        # SERVICES section — one row per process with per-service controls
        self._section(left, "SERVICES")

        self._service_row(left, "ollama",  "Ollama",          "LLM · :11434",
                          start_cmd=self._start_ollama_service,
                          stop_cmd=self._stop_ollama_service)

        self._service_row(left, "siem",    "SIEM Dashboard",   "FastAPI · :8787",
                          start_cmd=self._start_siem_service,
                          stop_cmd=self._stop_siem_service)

        self._service_row(left, "sancta",  "Sancta Agent",     "Main loop · auto-restart",
                          start_cmd=self._start_sancta_service,
                          stop_cmd=self._stop_sancta_service)

        self._service_row(left, "sangpt",  "Sangpt CLI",       "Interactive local GPT terminal",
                          start_cmd=self._start_sangpt_service,
                          stop_cmd=self._stop_sangpt_service)

        self._service_row(left, "sangpt_train",  "Sangpt Train", "Checkpointed GPT training run",
                          start_cmd=self._start_sangpt_train_service,
                          stop_cmd=self._stop_sangpt_train_service)

        self._toggle_service_row(left, "curiosity",     "Curiosity Run",
                                 "sancta.py --curiosity-run",
                                 run_text="◈  Run",
                                 run_color=C["amber"])

        self._toggle_service_row(left, "phenomenology", "Phenomenology",
                                 "11-vector attack battery",
                                 run_text="◇  Run",
                                 run_color=C["teal"])

        tk.Frame(left, bg=C["border"], height=1).pack(fill="x", padx=12, pady=8)

        # ACTIONS — global controls
        self._section(left, "ACTIONS")
        self.btn_start_all = self._btn(left, "▶  START ALL", C["green"],  self._start_all)
        self.btn_stop_all  = self._btn(left, "■  STOP ALL",  C["red"],    self._stop_all)
        self._btn(left, "⊞  OPEN DASHBOARD", C["purple"],
                  lambda: webbrowser.open(SIEM_URL))

        tk.Frame(left, bg=C["border"], height=1).pack(fill="x", padx=12, pady=8)

        # SESSION stats
        self._section(left, "SESSION")
        self.stat_vars = {
            "uptime":   tk.StringVar(value="00:00:00"),
            "restarts": tk.StringVar(value="0"),
            "model":    tk.StringVar(value="—"),
        }
        for label, var in [("Uptime",   self.stat_vars["uptime"]),
                            ("Restarts", self.stat_vars["restarts"]),
                            ("Model",    self.stat_vars["model"])]:
            row = tk.Frame(left, bg=C["bg2"])
            row.pack(fill="x", padx=16, pady=1)
            tk.Label(row, text=f"{label}:", font=self.fn_label,
                     fg=C["muted"], bg=C["bg2"], width=10, anchor="w").pack(side="left")
            tk.Label(row, textvariable=var, font=self.fn_label,
                     fg=C["text"], bg=C["bg2"]).pack(side="left")

        # ── Right: log panel ─────────────────────────────────────────────
        right = tk.Frame(main, bg=C["bg"])
        right.pack(side="left", fill="both", expand=True)

        # Log header
        log_hdr = tk.Frame(right, bg=C["bg3"], height=36)
        log_hdr.pack(fill="x")
        log_hdr.pack_propagate(False)

        tk.Label(log_hdr, text="LIVE LOG", font=self.fn_label,
                 fg=C["muted"], bg=C["bg3"]).pack(side="left", padx=12, pady=8)

        # Source filter dropdown
        self.log_filter = tk.StringVar(value="ALL")
        filter_menu = tk.OptionMenu(log_hdr, self.log_filter, *_FILTER_OPTS,
                                    command=self._on_filter_change)
        filter_menu.configure(font=self.fn_micro, bg=C["bg3"], fg=C["muted"],
                              activebackground=C["border"], activeforeground=C["text"],
                              relief="flat", bd=0, highlightthickness=0,
                              indicatoron=False)
        filter_menu["menu"].configure(font=self.fn_micro, bg=C["bg3"], fg=C["text"],
                                       activebackground=C["purple_dk"],
                                       tearoff=False)
        filter_menu.pack(side="left", padx=4)

        # Disable separator items in the menu
        menu = filter_menu["menu"]
        for i in range(menu.index("end") + 1):
            try:
                label = menu.entrycget(i, "label")
                if label in _FILTER_SEPS:
                    menu.entryconfig(i, state="disabled", foreground=C["dim"])
            except Exception:
                pass

        # CLR button
        tk.Button(log_hdr, text="CLR", font=self.fn_micro,
                  fg=C["muted"], bg=C["bg3"], relief="flat", bd=0,
                  padx=8, cursor="hand2", command=self._clear_log).pack(
            side="right", padx=10)

        # Scroll-to-bottom toggle (shows AUTO / MANUAL state)
        self._autoscroll = True
        self.btn_autoscroll = tk.Button(log_hdr, text="↓ AUTO", font=self.fn_micro,
                  fg=C["green"], bg=C["bg3"], relief="flat", bd=0,
                  padx=6, cursor="hand2",
                  command=self._toggle_autoscroll)
        self.btn_autoscroll.pack(side="right")

        # Log text widget
        log_frame = tk.Frame(right, bg=C["bg"])
        log_frame.pack(fill="both", expand=True, padx=1, pady=1)

        self.log_text = tk.Text(
            log_frame,
            bg=C["bg"], fg=C["text"],
            font=self.fn_mono,
            relief="flat", bd=0,
            state="disabled",
            wrap="none",
            insertbackground=C["purple"],
            selectbackground=C["purple_dk"],
        )
        sb = tk.Scrollbar(log_frame, orient="vertical",
                          command=self.log_text.yview,
                          bg=C["bg3"], troughcolor=C["bg"], relief="flat")
        sb.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=sb.set)
        self.log_text.pack(fill="both", expand=True)

        # Log color tags — subprocess sources
        self.log_text.tag_configure("ts",                   foreground=C["dim"])
        self.log_text.tag_configure("src_sancta",           foreground=C["purple"])
        self.log_text.tag_configure("src_sangpt",           foreground=C["cyan"])
        self.log_text.tag_configure("src_sangpt_train",     foreground=C["green"])
        self.log_text.tag_configure("src_siem",             foreground=C["teal"])
        self.log_text.tag_configure("src_ollama",           foreground=C["amber"])
        self.log_text.tag_configure("src_curiosity",        foreground=C["green"])
        self.log_text.tag_configure("src_phenomenology",    foreground=C["teal"])
        self.log_text.tag_configure("src_launcher",         foreground=C["muted"])
        # Log file sources
        self.log_text.tag_configure("src_security",         foreground=C["red"])
        self.log_text.tag_configure("src_redteam",          foreground=C["magenta"])
        self.log_text.tag_configure("src_behavioral",       foreground=C["purple"])
        self.log_text.tag_configure("src_activity",         foreground=C["cyan"])
        # Level tags
        self.log_text.tag_configure("lvl_ERROR",            foreground=C["red"])
        self.log_text.tag_configure("lvl_WARN",             foreground=C["amber"])
        self.log_text.tag_configure("lvl_OK",               foreground=C["green"])
        self.log_text.tag_configure("msg",                  foreground=C["text"])

        # Status bar
        self.statusbar = tk.Frame(self, bg=C["bg3"], height=24)
        self.statusbar.pack(fill="x", side="bottom")
        self.statusbar.pack_propagate(False)
        self.status_text = tk.StringVar(value="Ready")
        tk.Label(self.statusbar, textvariable=self.status_text,
                 font=self.fn_label, fg=C["muted"], bg=C["bg3"]).pack(
            side="left", padx=10)

        # Background workers
        threading.Thread(target=_background_net_checker,
                         daemon=True, name="net-checker").start()
        threading.Thread(target=_tail_log_files,
                         daemon=True, name="log-tailer").start()

    # ─── UI helpers ───────────────────────────────────────────────────────────

    def _section(self, parent, text: str):
        tk.Label(parent, text=text, font=self.fn_label,
                 fg=C["muted"], bg=C["bg2"]).pack(anchor="w", padx=16, pady=(12, 4))

    def _service_row(self, parent, key: str, name: str, sub: str,
                     start_cmd=None, stop_cmd=None):
        """Service row with status dot + name/sub + optional START and STOP buttons."""
        row = tk.Frame(parent, bg=C["bg2"])
        row.pack(fill="x", padx=10, pady=3)

        # Status dot
        lbl = tk.Label(row, textvariable=self.status_vars[key],
                       font=self.fn_head, fg=C["muted"], bg=C["bg2"], width=2)
        lbl.pack(side="left")
        self.status_labels[key] = lbl

        # Name + sub
        col = tk.Frame(row, bg=C["bg2"])
        col.pack(side="left", padx=4, fill="x", expand=True)
        tk.Label(col, text=name, font=self.fn_label,
                 fg=C["text"], bg=C["bg2"]).pack(anchor="w")
        tk.Label(col, text=sub, font=self.fn_micro,
                 fg=C["muted"], bg=C["bg2"]).pack(anchor="w")

        # Control buttons (right-aligned)
        self.svc_btns[key] = {}
        if start_cmd:
            b = tk.Button(row, text="▶", font=self.fn_micro, fg=C["green"],
                          bg=C["bg3"], relief="flat", bd=0, padx=7, pady=4,
                          cursor="hand2", command=start_cmd,
                          activebackground=C["border"], activeforeground=C["green"])
            b.pack(side="right", padx=1)
            self.svc_btns[key]["start"] = b
        if stop_cmd:
            b = tk.Button(row, text="■", font=self.fn_micro, fg=C["red"],
                          bg=C["bg3"], relief="flat", bd=0, padx=7, pady=4,
                          cursor="hand2", command=stop_cmd,
                          activebackground=C["border"], activeforeground=C["red"])
            b.pack(side="right", padx=1)
            self.svc_btns[key]["stop"] = b

    def _toggle_service_row(self, parent, key: str, name: str, sub: str,
                             run_text: str, run_color: str):
        """Service row for toggle-style processes (curiosity, phenomenology)."""
        row = tk.Frame(parent, bg=C["bg2"])
        row.pack(fill="x", padx=10, pady=3)

        lbl = tk.Label(row, textvariable=self.status_vars[key],
                       font=self.fn_head, fg=C["muted"], bg=C["bg2"], width=2)
        lbl.pack(side="left")
        self.status_labels[key] = lbl

        col = tk.Frame(row, bg=C["bg2"])
        col.pack(side="left", padx=4, fill="x", expand=True)
        tk.Label(col, text=name, font=self.fn_label,
                 fg=C["text"], bg=C["bg2"]).pack(anchor="w")
        tk.Label(col, text=sub, font=self.fn_micro,
                 fg=C["muted"], bg=C["bg2"]).pack(anchor="w")

        # Single toggle button
        toggle_cmd = (self._toggle_curiosity if key == "curiosity"
                      else self._toggle_phenomenology)
        btn = tk.Button(row, text=run_text, font=self.fn_micro, fg=run_color,
                        bg=C["bg3"], relief="flat", bd=0, padx=8, pady=4,
                        cursor="hand2", command=toggle_cmd,
                        activebackground=C["border"])
        btn.pack(side="right", padx=1)
        # Store so _poll_status can update it
        if key == "curiosity":
            self.btn_curiosity = btn
        else:
            self.btn_phenomenology = btn

    def _btn(self, parent, text: str, color: str, command) -> tk.Button:
        btn = tk.Button(
            parent, text=text, font=self.fn_btn, fg=color, bg=C["bg3"],
            relief="flat", bd=0, padx=14, pady=8, anchor="w",
            cursor="hand2", activebackground=C["border"], activeforeground=color,
            command=command,
        )
        btn.pack(fill="x", padx=16, pady=2)
        return btn

    def _bind_close(self):
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ─── Log management ───────────────────────────────────────────────────────

    def _drain_logs(self):
        """
        Batch drain: accumulate up to 50 messages → one widget update.
        Prevents per-line redraws which stutter Tkinter.
        """
        batch = []
        count = 0
        while not log_queue.empty() and count < 50:
            try:
                batch.append(log_queue.get_nowait())
                count += 1
            except queue.Empty:
                break

        if batch:
            filt = self.log_filter.get()
            # Separator items should never be selected, but guard anyway
            if filt in _FILTER_SEPS:
                filt = "ALL"

            self.log_text.configure(state="normal")

            # Trim excess lines (one operation, not per-message)
            lines = int(self.log_text.index("end-1c").split(".")[0])
            excess = lines - MAX_LOG_LINES + len(batch)
            if excess > 0:
                self.log_text.delete("1.0", f"{excess + 1}.0")

            for src, level, msg in batch:
                if filt != "ALL" and src.upper() != filt:
                    continue
                ts      = datetime.now().strftime("%H:%M:%S")
                src_tag = f"src_{src.lower()}"
                if src_tag not in _VALID_SRC_TAGS:
                    src_tag = "src_launcher"
                lvl_tag = f"lvl_{level}" if level in ("ERROR", "WARN", "OK") else "msg"

                self.log_text.insert("end", f"{ts} ", "ts")
                self.log_text.insert("end", f"[{src.upper():<11}] ", src_tag)
                self.log_text.insert("end", f"{msg}\n", lvl_tag)

            if self._autoscroll:
                self.log_text.see("end")
            self.log_text.configure(state="disabled")

        self.after(100, self._drain_logs)

    def _clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _on_filter_change(self, value):
        # Reject separator selections
        if value in _FILTER_SEPS:
            self.log_filter.set("ALL")

    def _toggle_autoscroll(self):
        self._autoscroll = not self._autoscroll
        if self._autoscroll:
            self.btn_autoscroll.configure(text="↓ AUTO", fg=C["green"])
        else:
            self.btn_autoscroll.configure(text="↓ MANUAL", fg=C["muted"])

    # ─── Status polling ───────────────────────────────────────────────────────

    def _poll_status(self):
        with _net_status_lock:
            ollama_ok = _net_status["ollama"]
            siem_ok   = _net_status["siem"]
            model     = _net_status["ollama_model"]

        # Ollama
        self._set_dot("ollama", "green" if ollama_ok else "red")
        self.stat_vars["model"].set(model or ("running, no model" if ollama_ok else "—"))
        # Start disabled when API up; Stop enabled when Ollama responds (matches SIEM Control tab)
        if "ollama" in self.svc_btns and "start" in self.svc_btns["ollama"]:
            self.svc_btns["ollama"]["start"].configure(
                state="disabled" if ollama_ok else "normal")
        if "ollama" in self.svc_btns and "stop" in self.svc_btns["ollama"]:
            self.svc_btns["ollama"]["stop"].configure(
                state="normal" if ollama_ok else "disabled")

        # SIEM
        siem_proc = processes.get("siem")
        siem_alive = siem_proc and siem_proc.poll() is None
        self._set_dot("siem", "green" if siem_ok else ("amber" if siem_alive else "red"))
        if "siem" in self.svc_btns:
            if "start" in self.svc_btns["siem"]:
                self.svc_btns["siem"]["start"].configure(
                    state="disabled" if siem_alive else "normal")
            if "stop" in self.svc_btns["siem"]:
                self.svc_btns["siem"]["stop"].configure(
                    state="normal" if siem_alive else "disabled")

        # Sancta
        sancta_proc = processes.get("sancta")
        sancta_alive = sancta_proc and sancta_proc.poll() is None
        self._set_dot("sancta",
                      "green" if sancta_alive else
                      ("amber" if "sancta" in processes else "red"))
        if "sancta" in self.svc_btns:
            if "start" in self.svc_btns["sancta"]:
                self.svc_btns["sancta"]["start"].configure(
                    state="disabled" if sancta_alive else "normal")
            if "stop" in self.svc_btns["sancta"]:
                self.svc_btns["sancta"]["stop"].configure(
                    state="normal" if sancta_alive else "disabled")

        # Sangpt CLI
        sangpt_proc = processes.get("sangpt")
        sangpt_alive = sangpt_proc and sangpt_proc.poll() is None
        self._set_dot("sangpt", "green" if sangpt_alive else "red")
        if "sangpt" in self.svc_btns:
            if "start" in self.svc_btns["sangpt"]:
                self.svc_btns["sangpt"]["start"].configure(
                    state="disabled" if sangpt_alive else "normal")
            if "stop" in self.svc_btns["sangpt"]:
                self.svc_btns["sangpt"]["stop"].configure(
                    state="normal" if sangpt_alive else "disabled")

        # Sangpt training
        sangpt_train_proc = processes.get("sangpt_train")
        sangpt_train_alive = sangpt_train_proc and sangpt_train_proc.poll() is None
        self._set_dot("sangpt_train", "green" if sangpt_train_alive else "red")
        if "sangpt_train" in self.svc_btns:
            if "start" in self.svc_btns["sangpt_train"]:
                self.svc_btns["sangpt_train"]["start"].configure(
                    state="disabled" if sangpt_train_alive else "normal")
            if "stop" in self.svc_btns["sangpt_train"]:
                self.svc_btns["sangpt_train"]["stop"].configure(
                    state="normal" if sangpt_train_alive else "disabled")

        # Curiosity
        cur_proc = processes.get("curiosity")
        cur_alive = cur_proc and cur_proc.poll() is None
        if cur_alive:
            self._set_dot("curiosity", "green")
            self.btn_curiosity.configure(text="■  Stop", fg=C["red"])
        else:
            if "curiosity" in processes:
                processes.pop("curiosity", None)
            self._set_dot("curiosity", "muted")
            self.btn_curiosity.configure(text="◈  Run", fg=C["amber"])

        # Phenomenology
        phen_proc = processes.get("phenomenology")
        phen_alive = phen_proc and phen_proc.poll() is None
        if phen_alive:
            self._set_dot("phenomenology", "green")
            self.btn_phenomenology.configure(text="■  Stop", fg=C["red"])
        else:
            if "phenomenology" in processes:
                processes.pop("phenomenology", None)
            self._set_dot("phenomenology", "muted")
            self.btn_phenomenology.configure(text="◇  Run", fg=C["teal"])

        # Session stats
        elapsed = int(time.time() - self._start_time)
        h, r = divmod(elapsed, 3600)
        m, s = divmod(r, 60)
        self.stat_vars["uptime"].set(f"{h:02d}:{m:02d}:{s:02d}")
        self.stat_vars["restarts"].set(str(sum(restart_counts.values())))

        self.after(POLL_INTERVAL, self._poll_status)

    def _set_dot(self, key: str, color: str):
        colors = {"green": C["green"], "red": C["red"],
                  "amber": C["amber"], "muted": C["muted"]}
        c = colors.get(color, C["muted"])
        self.status_vars[key].set("●")
        if key in self.status_labels:
            self.status_labels[key].configure(fg=c)

    # ─── Per-service actions ──────────────────────────────────────────────────

    def _start_ollama_service(self):
        threading.Thread(target=start_ollama, daemon=True).start()

    def _stop_ollama_service(self):
        log_queue.put(("launcher", "INFO", "Stopping Ollama..."))
        threading.Thread(target=lambda: stop_ollama(force_port=True), daemon=True).start()

    def _start_siem_service(self):
        if is_siem_running():
            log_queue.put(("siem", "INFO", "Already running on :8787"))
            return
        log_queue.put(("launcher", "INFO", "Starting SIEM server..."))
        threading.Thread(
            target=start_process,
            kwargs=dict(
                name="siem",
                script=BACKEND_DIR / "siem_server.py",
                restart=False,
                args_override=SIEM_CMD,
                cwd_override=ROOT,
                env_extra={
                    "SIEM_METRICS_SAFE_MODE": "false",
                    "SIEM_WS_SAFE_MODE": "false",
                },
            ),
            daemon=True,
        ).start()

    def _stop_siem_service(self):
        log_queue.put(("launcher", "INFO", "Stopping SIEM server..."))
        threading.Thread(target=stop_process, args=("siem",), daemon=True).start()

    def _start_sancta_service(self):
        proc = processes.get("sancta")
        if proc and proc.poll() is None:
            log_queue.put(("sancta", "INFO", "Already running"))
            return
        log_queue.put(("launcher", "INFO", "Starting Sancta agent..."))
        threading.Thread(
            target=start_process,
            kwargs=dict(name="sancta", script=SANCTA_SCRIPT, restart=True),
            daemon=True,
        ).start()

    def _stop_sancta_service(self):
        log_queue.put(("launcher", "INFO", "Stopping Sancta agent..."))
        threading.Thread(target=stop_process, args=("sancta",), daemon=True).start()

    def _start_sangpt_service(self):
        proc = processes.get("sangpt")
        if proc and proc.poll() is None:
            log_queue.put(("sangpt", "INFO", "Already running"))
            return
        log_queue.put(("launcher", "INFO", "Opening Sangpt CLI in a separate console..."))
        threading.Thread(
            target=start_interactive_console_process,
            kwargs=dict(name="sangpt", script=SANGPT_CLI_SCRIPT, cwd_override=ROOT),
            daemon=True,
        ).start()

    def _stop_sangpt_service(self):
        log_queue.put(("launcher", "INFO", "Stopping Sangpt CLI..."))
        threading.Thread(target=stop_process, args=("sangpt",), daemon=True).start()

    def _start_sangpt_train_service(self):
        proc = processes.get("sangpt_train")
        if proc and proc.poll() is None:
            log_queue.put(("sangpt_train", "INFO", "Training already running"))
            return
        log_queue.put(("launcher", "INFO", "Starting Sangpt training run..."))
        threading.Thread(
            target=start_process,
            kwargs=dict(name="sangpt_train", script=SANGPT_TRAIN_SCRIPT, extra_args=["25"], restart=False, cwd_override=ROOT),
            daemon=True,
        ).start()

    def _stop_sangpt_train_service(self):
        log_queue.put(("launcher", "INFO", "Stopping Sangpt training run..."))
        threading.Thread(target=stop_process, args=("sangpt_train",), daemon=True).start()

    # ─── Global actions ───────────────────────────────────────────────────────

    def _start_all(self):
        self.btn_start_all.configure(state="disabled")
        self.status_text.set("Starting services...")

        def _run():
            log_queue.put(("launcher", "INFO", "─── Starting Sancta stack ───"))

            if not start_ollama():
                log_queue.put(("launcher", "ERROR", "Ollama failed — check installation"))
                self.after(0, lambda: self.btn_start_all.configure(state="normal"))
                return

            time.sleep(1)

            if not is_siem_running():
                log_queue.put(("launcher", "INFO", "Starting SIEM server..."))
                start_process(
                    "siem", BACKEND_DIR / "siem_server.py",
                    restart=False, args_override=SIEM_CMD, cwd_override=ROOT,
                    env_extra={"SIEM_METRICS_SAFE_MODE": "false",
                               "SIEM_WS_SAFE_MODE": "false"},
                )
                for _ in range(10):
                    time.sleep(1)
                    if is_siem_running():
                        log_queue.put(("siem", "OK", "Dashboard ready on :8787"))
                        break
                else:
                    log_queue.put(("siem", "WARN", "SIEM slow to start — continuing"))
            else:
                log_queue.put(("siem", "INFO", "Already running on :8787"))

            log_queue.put(("launcher", "INFO", "Starting Sancta agent..."))
            start_process("sancta", SANCTA_SCRIPT, restart=True)
            time.sleep(2)

            log_queue.put(("launcher", "INFO", "Opening dashboard..."))
            time.sleep(1)
            webbrowser.open(SIEM_URL)

            log_queue.put(("launcher", "OK", "All services started ✓"))
            self.after(0, lambda: self.status_text.set("Running"))
            self.after(0, lambda: self.btn_start_all.configure(state="normal"))

        threading.Thread(target=_run, daemon=True).start()

    def _stop_all(self):
        self.status_text.set("Stopping...")
        log_queue.put(("launcher", "INFO", "─── Stopping all services ───"))

        def _run():
            stop_all()
            log_queue.put(("launcher", "OK", "All tracked services stopped (including Ollama if on :11434)"))
            self.after(0, lambda: self.status_text.set("Stopped"))

        threading.Thread(target=_run, daemon=True).start()

    def _toggle_curiosity(self):
        global curiosity_running
        proc = processes.get("curiosity")
        if proc and proc.poll() is None:
            log_queue.put(("curiosity", "INFO", "Stopping curiosity run..."))
            stop_process("curiosity")
            curiosity_running = False
        else:
            if not is_ollama_running():
                log_queue.put(("curiosity", "ERROR", "Ollama must be running first"))
                return
            log_queue.put(("curiosity", "INFO", "─── Starting curiosity run ───"))
            curiosity_running = True
            start_process("curiosity", SANCTA_SCRIPT,
                          extra_args=[CURIOSITY_FLAG], restart=False)

    def _toggle_phenomenology(self):
        proc = processes.get("phenomenology")
        if proc and proc.poll() is None:
            log_queue.put(("phenomenology", "INFO", "Stopping phenomenology battery..."))
            stop_process("phenomenology")
        else:
            log_queue.put(("phenomenology", "INFO",
                           "─── Running phenomenology attack battery (11 vectors) ───"))
            start_process("phenomenology", SANCTA_SCRIPT,
                          extra_args=[PHENOMENOLOGY_FLAG], restart=False)

    def _on_close(self):
        log_queue.put(("launcher", "INFO", "Shutting down..."))
        stop_all()
        self.after(800, self.destroy)


# ─── ANSI colors for CLI ─────────────────────────────────────────────────────

def _enable_win_ansi():
    """Enable ANSI escape codes + UTF-8 output on Windows console."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        # Enable virtual terminal processing on stdout
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class _A:
    """ANSI escape sequences."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    GRAY    = "\033[90m"

_SRC_COLORS = {
    "launcher":      _A.GRAY,
    "ollama":        _A.YELLOW,
    "siem":          _A.CYAN,
    "sancta":        _A.MAGENTA,
    "curiosity":     _A.GREEN,
    "phenomenology": _A.BLUE,
    "security":      _A.RED,
    "redteam":       _A.RED,
    "behavioral":    _A.MAGENTA,
    "activity":      _A.CYAN,
}

_LVL_COLORS = {
    "ERROR": _A.RED,
    "WARN":  _A.YELLOW,
    "OK":    _A.GREEN,
    "INFO":  _A.WHITE,
}


def _cli_print_log(src: str, level: str, msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    sc = _SRC_COLORS.get(src.lower(), _A.WHITE)
    lc = _LVL_COLORS.get(level, _A.WHITE)
    print(f"{_A.DIM}{ts}{_A.RESET} {sc}[{src.upper():<11}]{_A.RESET} {lc}{msg}{_A.RESET}",
          flush=True)


def _cli_drain_logs():
    """Drain log_queue to terminal — called from CLI log thread."""
    while True:
        try:
            src, level, msg = log_queue.get(timeout=0.3)
            _cli_print_log(src, level, msg)
        except queue.Empty:
            pass
        except Exception:
            break


def _cli_status_line() -> str:
    with _net_status_lock:
        ollama_ok = _net_status["ollama"]
        siem_ok   = _net_status["siem"]
        model     = _net_status["ollama_model"]

    sancta_proc = processes.get("sancta")
    sancta_ok   = sancta_proc and sancta_proc.poll() is None
    sangpt_proc = processes.get("sangpt")
    sangpt_ok   = sangpt_proc and sangpt_proc.poll() is None
    sangpt_train_proc = processes.get("sangpt_train")
    sangpt_train_ok   = sangpt_train_proc and sangpt_train_proc.poll() is None
    cur_proc    = processes.get("curiosity")
    cur_ok      = cur_proc and cur_proc.poll() is None
    phen_proc   = processes.get("phenomenology")
    phen_ok     = phen_proc and phen_proc.poll() is None

    def dot(ok):
        return f"{_A.GREEN}●{_A.RESET}" if ok else f"{_A.RED}●{_A.RESET}"

    parts = [
        f"  Ollama {dot(ollama_ok)} {model or '—':20s}",
        f"  SIEM   {dot(siem_ok)}  :8787",
        f"  Sancta {dot(sancta_ok)}  agent loop",
    ]
    sangpt_proc = processes.get("sangpt")
    sangpt_ok = sangpt_proc and sangpt_proc.poll() is None
    if sangpt_ok:
        parts.append(f"  Sangpt {dot(sangpt_ok)}")
    sangpt_train_proc = processes.get("sangpt_train")
    sangpt_train_ok = sangpt_train_proc and sangpt_train_proc.poll() is None
    if sangpt_train_ok:
        parts.append(f"  SangptTrain {dot(sangpt_train_ok)}")
    if cur_ok:
        parts.append(f"  Curiosity {dot(cur_ok)}")
    if phen_ok:
        parts.append(f"  Phenomenology {dot(phen_ok)}")
    return "  ".join(parts)


def _cli_start_all(open_browser: bool = True):
    """Start the full Sancta stack sequentially (blocking)."""
    log_queue.put(("launcher", "INFO", "─── Starting Sancta stack ───"))

    if not start_ollama():
        log_queue.put(("launcher", "ERROR", "Ollama failed — check installation"))
        return False

    time.sleep(1)

    if not is_siem_running():
        log_queue.put(("launcher", "INFO", "Starting SIEM server..."))
        start_process(
            "siem", BACKEND_DIR / "siem_server.py",
            restart=False, args_override=SIEM_CMD, cwd_override=ROOT,
            env_extra={"SIEM_METRICS_SAFE_MODE": "false",
                       "SIEM_WS_SAFE_MODE": "false"},
        )
        for _ in range(10):
            time.sleep(1)
            if is_siem_running():
                log_queue.put(("siem", "OK", "Dashboard ready on :8787"))
                break
        else:
            log_queue.put(("siem", "WARN", "SIEM slow to start — continuing"))
    else:
        log_queue.put(("siem", "INFO", "Already running on :8787"))

    log_queue.put(("launcher", "INFO", "Starting Sancta agent..."))
    start_process("sancta", SANCTA_SCRIPT, restart=True)
    time.sleep(2)

    if open_browser:
        log_queue.put(("launcher", "INFO", "Opening dashboard..."))
        time.sleep(1)
        webbrowser.open(SIEM_URL)

    log_queue.put(("launcher", "OK", "All services started ✓"))
    return True


def _cli_start_one(name: str) -> bool:
    """Start a single named service."""
    name = name.lower().strip()
    if name == "ollama":
        return start_ollama()
    elif name == "siem":
        if is_siem_running():
            log_queue.put(("siem", "INFO", "Already running on :8787"))
            return True
        log_queue.put(("launcher", "INFO", "Starting SIEM server..."))
        start_process(
            "siem", BACKEND_DIR / "siem_server.py",
            restart=False, args_override=SIEM_CMD, cwd_override=ROOT,
            env_extra={"SIEM_METRICS_SAFE_MODE": "false",
                       "SIEM_WS_SAFE_MODE": "false"},
        )
        return True
    elif name == "sancta":
        proc = processes.get("sancta")
        if proc and proc.poll() is None:
            log_queue.put(("sancta", "INFO", "Already running"))
            return True
        log_queue.put(("launcher", "INFO", "Starting Sancta agent..."))
        start_process("sancta", SANCTA_SCRIPT, restart=True)
        return True
    elif name == "sangpt":
        proc = processes.get("sangpt")
        if proc and proc.poll() is None:
            log_queue.put(("sangpt", "INFO", "Already running"))
            return True
        log_queue.put(("launcher", "INFO", "Opening Sangpt CLI in a separate console..."))
        start_interactive_console_process("sangpt", SANGPT_CLI_SCRIPT, cwd_override=ROOT)
        return True
    elif name in ("sangpt-train", "sangpt_train"):
        proc = processes.get("sangpt_train")
        if proc and proc.poll() is None:
            log_queue.put(("sangpt_train", "INFO", "Training already running"))
            return True
        log_queue.put(("launcher", "INFO", "Starting Sangpt training run..."))
        start_process("sangpt_train", SANGPT_TRAIN_SCRIPT,
                      extra_args=["25"], restart=False, cwd_override=ROOT)
        return True
    elif name == "curiosity":
        if not is_ollama_running():
            log_queue.put(("curiosity", "ERROR", "Ollama must be running first"))
            return False
        log_queue.put(("curiosity", "INFO", "─── Starting curiosity run ───"))
        start_process("curiosity", SANCTA_SCRIPT,
                      extra_args=[CURIOSITY_FLAG], restart=False)
        return True
    elif name == "phenomenology":
        log_queue.put(("phenomenology", "INFO",
                       "─── Running phenomenology attack battery (11 vectors) ───"))
        start_process("phenomenology", SANCTA_SCRIPT,
                      extra_args=[PHENOMENOLOGY_FLAG], restart=False)
        return True
    else:
        log_queue.put(("launcher", "ERROR", f"Unknown service: {name}"))
        return False


def _cli_stop_one(name: str):
    name = name.lower().strip()
    if name == "ollama":
        stop_ollama(force_port=True)
        return
    if name in processes:
        stop_process(name)
    else:
        log_queue.put(("launcher", "WARN", f"'{name}' is not tracked by launcher (already stopped?)"))


def _cli_print_status():
    with _net_status_lock:
        ollama_ok = _net_status["ollama"]
        siem_ok   = _net_status["siem"]
        model     = _net_status["ollama_model"]

    sancta_proc = processes.get("sancta")
    sancta_ok   = sancta_proc and sancta_proc.poll() is None
    sangpt_proc = processes.get("sangpt")
    sangpt_ok   = sangpt_proc and sangpt_proc.poll() is None
    sangpt_train_proc = processes.get("sangpt_train")
    sangpt_train_ok   = sangpt_train_proc and sangpt_train_proc.poll() is None
    cur_proc    = processes.get("curiosity")
    cur_ok      = cur_proc and cur_proc.poll() is None
    phen_proc   = processes.get("phenomenology")
    phen_ok     = phen_proc and phen_proc.poll() is None

    def row(name, ok, detail=""):
        icon = f"{_A.GREEN}● ONLINE{_A.RESET}" if ok else f"{_A.RED}● OFFLINE{_A.RESET}"
        det  = f"  {_A.DIM}{detail}{_A.RESET}" if detail else ""
        print(f"  {name:<16} {icon}{det}")

    print(f"\n{_A.BOLD}  SERVICE STATUS{_A.RESET}")
    print(f"  {'─' * 44}")
    row("Ollama",        ollama_ok,  model or "")
    row("SIEM Dashboard", siem_ok,   ":8787")
    row("Sancta Agent",  sancta_ok,  f"PID {sancta_proc.pid}" if sancta_ok else "")
    row("Sangpt CLI",    sangpt_ok,  f"PID {sangpt_proc.pid}" if sangpt_ok else "")
    row("Sangpt Train",  sangpt_train_ok,  f"PID {sangpt_train_proc.pid}" if sangpt_train_ok else "")
    row("Curiosity",     cur_ok)
    row("Phenomenology", phen_ok)
    print(f"  {'─' * 44}")
    print(f"  Restarts: {sum(restart_counts.values())}  |  "
          f"Processes tracked: {len(processes)}\n")


_CLI_HELP = f"""{_A.BOLD}
  Sancta CLI — Interactive Commands{_A.RESET}
  {'─' * 40}
  {_A.GREEN}start{_A.RESET}                  Start all services (Ollama → SIEM → Sancta)
  {_A.GREEN}start{_A.RESET} <service>        Start a single service
  {_A.RED}stop{_A.RESET}                   Stop all services
  {_A.RED}stop{_A.RESET} <service>         Stop a single service
  {_A.CYAN}status{_A.RESET}                 Show service status table
  {_A.YELLOW}curiosity{_A.RESET}              Run curiosity knowledge scan
  {_A.BLUE}phenomenology{_A.RESET}          Run 11-vector attack battery
  {_A.MAGENTA}dashboard{_A.RESET}              Open SIEM dashboard in browser
  {_A.WHITE}clear{_A.RESET}                  Clear terminal
  {_A.WHITE}help{_A.RESET}                   Show this help
  {_A.WHITE}exit{_A.RESET} / {_A.WHITE}quit{_A.RESET}           Stop all and exit

  Services: ollama, siem, sancta, sangpt, sangpt-train, curiosity, phenomenology
"""


def _cli_interactive():
    """Interactive CLI REPL — replaces Tkinter GUI."""
    _enable_win_ansi()

    print(f"""
{_A.BOLD}{_A.MAGENTA}  ⬡ SANCTA CONTROL CENTER{_A.RESET}  {_A.DIM}v3.0 — CLI Mode{_A.RESET}
  {'─' * 44}
  Type {_A.GREEN}help{_A.RESET} for commands, {_A.GREEN}start{_A.RESET} to launch all services
  Logs stream in real-time. Press {_A.YELLOW}Ctrl+C{_A.RESET} to stop.
""")

    # Start background threads
    threading.Thread(target=_background_net_checker, daemon=True, name="net-checker").start()
    threading.Thread(target=_tail_log_files, daemon=True, name="log-tailer").start()
    threading.Thread(target=_cli_drain_logs, daemon=True, name="cli-log-drain").start()

    # Give the net checker a moment to populate initial status
    time.sleep(1.5)

    try:
        while True:
            try:
                raw = input(f"{_A.DIM}sancta>{_A.RESET} ").strip()
            except EOFError:
                break
            if not raw:
                continue

            parts = raw.split(None, 1)
            cmd   = parts[0].lower()
            arg   = parts[1].strip() if len(parts) > 1 else ""

            if cmd in ("exit", "quit", "q"):
                log_queue.put(("launcher", "INFO", "Shutting down..."))
                stop_all()
                time.sleep(1)
                break
            elif cmd == "help" or cmd == "?":
                print(_CLI_HELP)
            elif cmd == "start":
                if arg:
                    threading.Thread(target=_cli_start_one, args=(arg,), daemon=True).start()
                else:
                    threading.Thread(target=_cli_start_all, daemon=True).start()
            elif cmd == "stop":
                if arg:
                    _cli_stop_one(arg)
                else:
                    log_queue.put(("launcher", "INFO", "─── Stopping all services ───"))
                    stop_all()
                    log_queue.put(("launcher", "OK", "All services stopped"))
            elif cmd == "status" or cmd == "st":
                _cli_print_status()
            elif cmd == "curiosity":
                threading.Thread(target=_cli_start_one, args=("curiosity",), daemon=True).start()
            elif cmd == "phenomenology" or cmd == "phenom":
                threading.Thread(target=_cli_start_one, args=("phenomenology",), daemon=True).start()
            elif cmd == "dashboard" or cmd == "dash":
                webbrowser.open(SIEM_URL)
                log_queue.put(("launcher", "INFO", "Opened dashboard in browser"))
            elif cmd == "clear" or cmd == "cls":
                os.system("cls" if sys.platform == "win32" else "clear")
            elif cmd == "restart":
                svc = arg or "sancta"
                _cli_stop_one(svc)
                time.sleep(2)
                threading.Thread(target=_cli_start_one, args=(svc,), daemon=True).start()
            else:
                print(f"  {_A.RED}Unknown command:{_A.RESET} {cmd}  — type {_A.GREEN}help{_A.RESET} for commands")

    except KeyboardInterrupt:
        print(f"\n{_A.YELLOW}  Interrupted — stopping services...{_A.RESET}")
        stop_all()
        time.sleep(1)


def _cli_run_and_attach():
    """Non-interactive: start all, stream logs until Ctrl+C."""
    _enable_win_ansi()

    print(f"""
{_A.BOLD}{_A.MAGENTA}  ⬡ SANCTA CONTROL CENTER{_A.RESET}  {_A.DIM}v3.0{_A.RESET}
  Starting all services... Press {_A.YELLOW}Ctrl+C{_A.RESET} to stop.
""")

    threading.Thread(target=_background_net_checker, daemon=True, name="net-checker").start()
    threading.Thread(target=_tail_log_files, daemon=True, name="log-tailer").start()
    threading.Thread(target=_cli_drain_logs, daemon=True, name="cli-log-drain").start()

    time.sleep(1)
    _cli_start_all(open_browser=True)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n{_A.YELLOW}  Interrupted — stopping services...{_A.RESET}")
        stop_all()
        time.sleep(1)


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        prog="sancta_launcher",
        description="Sancta Control Center — manage all Sancta services",
    )
    sub = parser.add_subparsers(dest="mode")

    # Default (no subcommand) → GUI
    sub.add_parser("gui",   help="Launch the Tkinter GUI (default)")
    sub.add_parser("cli",   help="Interactive CLI with live log streaming")
    sub.add_parser("start", help="Start all services and stream logs (non-interactive)")

    sp_run = sub.add_parser("run", help="Start a single service and stream logs")
    sp_run.add_argument("service", choices=["ollama", "siem", "sancta", "sangpt", "sangpt-train", "curiosity", "phenomenology"])

    sub.add_parser("status", help="Print service status and exit")

    args = parser.parse_args()

    if args.mode == "cli":
        _cli_interactive()
    elif args.mode == "start":
        _cli_run_and_attach()
    elif args.mode == "run":
        _enable_win_ansi()
        threading.Thread(target=_background_net_checker, daemon=True).start()
        threading.Thread(target=_tail_log_files, daemon=True).start()
        threading.Thread(target=_cli_drain_logs, daemon=True).start()
        time.sleep(1)
        _cli_start_one(args.service)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"\n{_A.YELLOW}  Stopping...{_A.RESET}")
            stop_all()
            time.sleep(1)
    elif args.mode == "status":
        _enable_win_ansi()
        threading.Thread(target=_background_net_checker, daemon=True).start()
        time.sleep(2)
        _cli_print_status()
    else:
        # Default: GUI
        app = SanctaLauncher()
        log_queue.put(("launcher", "OK",
                       "Sancta Control Center v3.0 ready — use START ALL or per-service buttons"))
        log_queue.put(("launcher", "INFO",
                       f"Tailing: {', '.join(LOG_FILES.keys())}"))
        app.mainloop()


if __name__ == "__main__":
    main()
