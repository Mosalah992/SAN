from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.error
import urllib.request
import re
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

try:
    _psutil_disabled = os.environ.get("SIEM_PSUTIL_DISABLE", "true" if os.name == "nt" else "false").lower() in ("1", "true", "yes")
    if _psutil_disabled:
        psutil = None  # type: ignore[assignment]
    else:
        import psutil
except Exception:
    psutil = None  # type: ignore[assignment]
from fastapi import Body, FastAPI, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles


# backend/siem_server.py: parent=backend/, parents[1]=project root
_BACKEND = Path(__file__).resolve().parent
ROOT = _BACKEND.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

try:
    import sancta_trust_config as _sancta_trust_config

    _sancta_trust_config.log_startup_warnings()
except Exception:
    pass

# Ollama: connect only, never start. Must run `ollama serve` manually first.
if os.environ.get("USE_LOCAL_LLM", "false").lower() in ("1", "true", "yes"):
    try:
        import sancta_ollama
        if not sancta_ollama.wait_until_ready(
            model=os.environ.get("LOCAL_MODEL", "llama3.2"),
            timeout=30,
        ):
            os.environ["USE_LOCAL_LLM"] = "false"
    except Exception:
        os.environ["USE_LOCAL_LLM"] = "false"

try:
    import sancta_conversational as _sc
    _sc.init(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
except Exception as e:
    logging.getLogger("siem_chat").debug("sancta_conversational init skipped: %s", e)

# Disable trained transformer to avoid PyTorch ACCESS_VIOLATION crash on some Windows setups.
# Set SANCTA_USE_TRAINED_TRANSFORMER=true in .env to re-enable (if stable on your system).
if "SANCTA_USE_TRAINED_TRANSFORMER" not in os.environ:
    os.environ["SANCTA_USE_TRAINED_TRANSFORMER"] = "false"

from sancta_events import EventCategory, notify
from sancta_atlas import (
    classify_event as atlas_classify,
    get_coverage as atlas_coverage,
    get_matrix_data as atlas_matrix,
    ttp_tracker,
    TACTICS, TECHNIQUES, TACTIC_ORDER,
)
LOG_DIR = ROOT / "logs"
CHAT_LOG = LOG_DIR / "siem_chat.log"

_chat_log = logging.getLogger("siem_chat")
if not _chat_log.handlers:
    LOG_DIR.mkdir(exist_ok=True)
    _chat_log.setLevel(logging.INFO)
    _chat_log.propagate = False
    _fh = logging.FileHandler(CHAT_LOG, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    _chat_log.addHandler(_fh)

_epidemic_log = logging.getLogger("siem_epidemic")
if not _epidemic_log.handlers:
    LOG_DIR.mkdir(exist_ok=True)
    _epidemic_log.setLevel(logging.INFO)
    _epidemic_log.propagate = False
    _efh = logging.FileHandler(LOG_DIR / "epidemic.log", encoding="utf-8")
    _efh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    _epidemic_log.addHandler(_efh)
    _epidemic_log.info("epidemic logger started")

STATE_PATH = ROOT / "agent_state.json"
SANCTA_PATH = _BACKEND / "sancta.py"
PID_PATH = ROOT / ".agent.pid"
AGENT_ACTIVITY_LOG = LOG_DIR / "agent_activity.log"

# Security hardening
SIEM_AUTH_TOKEN: str | None = os.environ.get("SIEM_AUTH_TOKEN") or None
# Default True to avoid ACCESS_VIOLATION crash on some Windows setups (WebSocket file I/O)
SIEM_WS_SAFE_MODE: bool = os.environ.get("SIEM_WS_SAFE_MODE", "true").lower() in ("1", "true", "yes")
# When True, skip agent-activity + live-events file reads. Default False so dashboard populates.
# Set SIEM_METRICS_SAFE_MODE=true if you see crashes on Windows.
SIEM_METRICS_SAFE_MODE: bool = os.environ.get("SIEM_METRICS_SAFE_MODE", "false").lower() in ("1", "true", "yes")
ALLOWED_MODES: frozenset[str] = frozenset({"passive", "blue", "sim", "active", "once"})

JSONL_SOURCES = {
    "security": LOG_DIR / "security.jsonl",
    "redteam": LOG_DIR / "red_team.jsonl",
    "behavioral": LOG_DIR / "behavioral.jsonl",
}


async def _require_auth(request: Request) -> None:
    """Raise 401 if SIEM_AUTH_TOKEN is set and request lacks valid Bearer token."""
    if not SIEM_AUTH_TOKEN:
        return
    auth = request.headers.get("Authorization") or ""
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = auth[7:].strip()
    if token != SIEM_AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")


def _redact_log_line(line: str) -> str:
    """Redact API keys, paths, claim URLs, and other sensitive data from log lines."""
    out = line
    # API keys: moltbook_sk_..., sk-..., pk-..., etc.
    out = re.sub(
        r"\b(?:moltbook_sk_|moltbook_pk_|sk-[a-zA-Z0-9_-]{20,}|pk-[a-zA-Z0-9_-]{20,})[a-zA-Z0-9_-]*\b",
        "[API_KEY]",
        out,
    )
    # Bearer tokens
    out = re.sub(r"Bearer\s+[a-zA-Z0-9_-]{10,}", "Bearer [REDACTED]", out, flags=re.IGNORECASE)
    # Absolute paths (Windows: to .env/.log/etc; Unix: /home, /Users)
    out = re.sub(r"[A-Za-z]:\\[\s\S]*?\.(?:env|pid|log|json)\b", "[PATH]", out)
    out = re.sub(r"[A-Za-z]:\\[^\s]+", "[PATH]", out)
    out = re.sub(r"/home/[^\s]+", "[PATH]", out)
    out = re.sub(r"/Users/[^\s]+", "[PATH]", out)
    # Claim / verify URLs and content IDs
    out = re.sub(
        r"https?://[^\s]*(?:moltbook|claim|verify)[^\s]*",
        "[URL]",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        "[UUID]",
        out,
        flags=re.IGNORECASE,
    )
    return out


def _read_json_line(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict):
        return obj
    return None


def _safe_read_state() -> dict[str, Any]:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _ensure_log_dir() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    for p in JSONL_SOURCES.values():
        if not p.exists():
            p.write_text("", encoding="utf-8")


def _pid_read() -> int | None:
    if not PID_PATH.exists():
        return None
    try:
        return int(PID_PATH.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _pid_write(pid: int) -> None:
    PID_PATH.write_text(str(pid), encoding="utf-8")


def _pid_clear() -> None:
    try:
        PID_PATH.unlink(missing_ok=True)  # py3.8+: missing_ok supported
    except TypeError:
        if PID_PATH.exists():
            PID_PATH.unlink()


def _pid_running_no_psutil(pid: int) -> bool:
    """Check if process exists without psutil (Windows: tasklist; Unix: /proc or kill -0)."""
    if os.name == "nt":
        try:
            r = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            return str(pid) in (r.stdout or "")
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _proc_from_pid(pid: int | None) -> Any:
    """Return psutil.Process if available and running, else None. With psutil disabled, returns a sentinel dict."""
    if not pid:
        return None
    if psutil is None:
        return {"pid": pid} if _pid_running_no_psutil(pid) else None
    try:
        p = psutil.Process(pid)
    except Exception:
        return None
    if not p.is_running():
        return None
    return p


def _agent_state_suggests_running(max_age_sec: float = 900.0) -> bool:
    """True if agent_state.json looks recently updated with a non-zero cycle counter."""
    try:
        if not STATE_PATH.exists():
            return False
        mtime = STATE_PATH.stat().st_mtime
        if (time.time() - mtime) > max_age_sec:
            return False
        st = _safe_read_state()
        return int(st.get("cycle_count", 0) or 0) > 0
    except Exception:
        return False


def _agent_status() -> dict[str, Any]:
    """
    Reconcile dashboard agent card with external starters (sancta-launcher, CLI).

    Order: PID file + liveness → scan for ``sancta.py`` in process list (heal .agent.pid)
    → recent agent_state.json → only then clear stale PID and report stopped.
    """
    pid = _pid_read()
    proc = _proc_from_pid(pid) if pid else None

    if proc is not None:
        if isinstance(proc, dict):
            return {"running": True, "pid": proc["pid"], "suspended": False}
        try:
            status = proc.status()
            suspended = status == psutil.STATUS_STOPPED
        except Exception:
            suspended = False
        return {"running": True, "pid": proc.pid, "suspended": suspended}

    # PID missing or not visible to psutil/tasklist (common on Windows with SIEM_PSUTIL_DISABLE).
    discovered: int | None = None
    try:
        discovered = _find_process_by_script("sancta.py")
    except Exception:
        discovered = None
    if discovered is not None:
        alive = _proc_from_pid(discovered)
        if alive is not None:
            try:
                _pid_write(discovered)
            except OSError:
                pass
            if isinstance(alive, dict):
                return {"running": True, "pid": discovered, "suspended": False}
            try:
                status = alive.status()
                suspended = status == psutil.STATUS_STOPPED
            except Exception:
                suspended = False
            return {"running": True, "pid": discovered, "suspended": suspended}

    if _agent_state_suggests_running():
        # Process enumeration failed but the agent loop is still updating state — stay "online".
        return {"running": True, "pid": pid or discovered, "suspended": False}

    _pid_clear()
    return {"running": False, "pid": None, "suspended": False}


def _start_agent(mode: str) -> dict[str, Any]:
    st = _agent_status()
    if st["running"]:
        return {"ok": True, **st}

    # Map UI modes -> CLI args/env. Keep conservative defaults.
    # passive / active / sim: full heartbeat loop (sim no longer implies --once; that confused users).
    # blue: policy-test posting mode.
    # once: explicit single cycle then exit (for smoke tests only).
    args: list[str] = [os.fspath(SANCTA_PATH)]
    if mode == "blue":
        args += ["--policy-test"]
    if mode == "once":
        args += ["--once"]

    # Use the same Python interpreter the dashboard runs under
    cmd = [os.fspath(Path(os.sys.executable).resolve())] + args

    creationflags = 0
    if os.name == "nt":
        # Create its own process group so we can terminate cleanly
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    proc = subprocess.Popen(
        cmd,
        cwd=os.fspath(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )
    _pid_write(proc.pid)

    notify(
        EventCategory.SESSION_START,
        summary=f"Sancta agent started (mode={mode})",
        details={"pid": proc.pid},
    )
    return {"ok": True, "running": True, "pid": proc.pid, "suspended": False}


def _pause_agent() -> dict[str, Any]:
    pid = _pid_read()
    proc = _proc_from_pid(pid)
    if not proc:
        _pid_clear()
        return {"ok": False, "error": "not_running"}
    if isinstance(proc, dict):
        return {"ok": False, "error": "suspend not available (psutil disabled)"}
    try:
        proc.suspend()
    except Exception as e:
        notify(
            EventCategory.TASK_ERROR,
            summary="Failed to pause Sancta agent",
            details={"error": str(e)},
        )
        return {"ok": False, "error": str(e)}

    notify(
        EventCategory.SESSION_END,
        summary="Sancta agent paused",
        details={"pid": proc.pid},
    )
    return {"ok": True, **_agent_status()}


def _resume_agent() -> dict[str, Any]:
    pid = _pid_read()
    proc = _proc_from_pid(pid)
    if not proc:
        _pid_clear()
        return {"ok": False, "error": "not_running"}
    if isinstance(proc, dict):
        return {"ok": False, "error": "resume not available (psutil disabled)"}
    try:
        proc.resume()
    except Exception as e:
        notify(
            EventCategory.TASK_ERROR,
            summary="Failed to resume Sancta agent",
            details={"error": str(e)},
        )
        return {"ok": False, "error": str(e)}

    notify(
        EventCategory.SESSION_START,
        summary="Sancta agent resumed",
        details={"pid": proc.pid},
    )
    return {"ok": True, **_agent_status()}


def _kill_agent() -> dict[str, Any]:
    pid = _pid_read()
    proc = _proc_from_pid(pid)
    if not proc:
        _pid_clear()
        return {"ok": True, "running": False, "pid": None}

    if isinstance(proc, dict):
        # psutil disabled: use subprocess to kill
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, timeout=5)
            else:
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.5)
                try:
                    os.kill(pid, 0)
                except OSError:
                    pass
                else:
                    os.kill(pid, signal.SIGKILL)
        except Exception as exc:
            notify(
                EventCategory.TASK_ERROR,
                summary="Error while killing Sancta agent",
                details={"error": str(exc)},
            )
        _pid_clear()
        notify(EventCategory.SESSION_END, summary="Sancta agent stopped", details={"pid": pid})
        return {"ok": True, "running": False, "pid": None}

    try:
        # Try graceful termination first
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            time.sleep(0.2)
        proc.terminate()
        proc.wait(timeout=3)
    except Exception as exc:
        try:
            proc.kill()
        except Exception:
            pass
        notify(
            EventCategory.TASK_ERROR,
            summary="Error while killing Sancta agent",
            details={"error": str(exc)},
        )

    _pid_clear()
    notify(
        EventCategory.SESSION_END,
        summary="Sancta agent stopped",
        details={"pid": pid},
    )
    return {"ok": True, "running": False, "pid": None}


def _restart_agent(mode: str) -> dict[str, Any]:
    _kill_agent()
    return _start_agent(mode)


@dataclass
class TailCursor:
    offset: int = 0


class LiveMetrics:
    def __init__(self) -> None:
        self.injection_attempts = 0
        self.output_redactions = 0
        self.reward_sum_rolling: list[float] = []
        self.fp_rate = None
        self.mood = None
        self.belief_confidence = None

    def update_from_event(self, source: str, ev: dict[str, Any], *, silent_notifications: bool = False) -> None:
        """
        Update metrics from a JSONL event. When silent_notifications=True, skips
        notify() calls to avoid pygame crashes on Windows when SIEM processes file-backed events.
        """
        event = ev.get("event")
        if source == "security":
            if event in ("input_reject", "injection_blocked", "suspicious_block"):
                self.injection_attempts += 1
            if event == "output_redact":
                self.output_redactions += 1

            if event in ("input_reject", "injection_blocked", "suspicious_block") and not silent_notifications:
                summary = ev.get("message") or f"Security event: {event}"
                notify(
                    EventCategory.SECURITY_ALERT,
                    summary=summary,
                    details={"source": source, "event": event},
                )

        if source == "redteam" and event == "redteam_reward":
            try:
                r = float(ev.get("reward") or ev.get("data", {}).get("reward") or 0.0)
            except Exception:
                r = 0.0
            self.reward_sum_rolling = (self.reward_sum_rolling + [r])[-50:]

            # Only alert when the reward is meaningfully high.
            if r >= 0.5 and not silent_notifications:
                notify(
                    EventCategory.REDTEAM_ALERT,
                    summary=f"Red-team reward={r:.2f}",
                    details={"reward": r},
                )

        if source in ("philosophy", "behavioral") and event in ("epistemic_state", "behavioral_state"):
            data = ev.get("data") or {}
            self.mood = (
                ev.get("mood")
                or data.get("mood")
                or data.get("current_mood")
                or (data.get("epistemic_state") or {}).get("mood")
            )

        # Pull a couple metrics from agent_state.json opportunistically
        st = _safe_read_state()
        rt = st.get("red_team_belief", {})
        try:
            a = float(rt.get("alpha", 0.0))
            b = float(rt.get("beta", 0.0))
            self.belief_confidence = (a / (a + b)) if (a + b) > 0 else None
        except Exception:
            self.belief_confidence = None

        # FP rate is in red-team simulation metrics, not always available; best-effort:
        last_sim = st.get("red_team_last_simulation")
        if isinstance(last_sim, dict) and "fp_rate" in last_sim:
            self.fp_rate = last_sim.get("fp_rate")

    def snapshot(self) -> dict[str, Any]:
        rolling_reward = sum(self.reward_sum_rolling) if self.reward_sum_rolling else 0.0
        st = _safe_read_state()
        mood = self.mood
        if mood is None:
            mood = (
                st.get("current_mood")
                or st.get("agent_mood")
                or (st.get("memory") or {}).get("epistemic_state", {}).get("mood")
            )
            if isinstance(mood, dict):
                mood = mood.get("current", "contemplative") or "contemplative"
        return {
            "injection_attempts_detected": self.injection_attempts,
            "sanitized_payload_count": self.output_redactions,
            "reward_score_rolling_sum": round(float(rolling_reward), 4),
            "false_positive_rate": self.fp_rate,
            "belief_confidence": self.belief_confidence,
            "agent_mood": mood or "contemplative",
            **_agent_status(),
        }


def _tail_jsonl_sync(path: Path, cursor: TailCursor, max_bytes: int = 64_000) -> list[dict[str, Any]]:
    """Sync tail of JSONL; used from thread to avoid blocking event loop."""
    data = b""
    try:
        if not path.exists():
            return []
        size = path.stat().st_size
        if size == 0:
            return []
        # File truncated/rotated
        if cursor.offset > size:
            cursor.offset = 0
        read_from = cursor.offset
        read_to = size
        if read_to - read_from > max_bytes:
            read_from = read_to - max_bytes
        with open(path, "rb") as f:
            f.seek(read_from)
            data = f.read()
        cursor.offset = size
    except OSError:
        return []
    except Exception:
        return []

    out: list[dict[str, Any]] = []
    text = data.decode("utf-8", errors="ignore")
    for line in text.splitlines():
        obj = _read_json_line(line)
        if obj:
            out.append(obj)
    return out


async def _tail_jsonl(path: Path, cursor: TailCursor, max_bytes: int = 64_000) -> list[dict[str, Any]]:
    """Tail JSONL. On Windows, run sync to avoid run_in_executor crash; elsewhere use thread."""
    if os.name == "nt":
        return _tail_jsonl_sync(path, cursor, max_bytes)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _tail_jsonl_sync, path, cursor, max_bytes)


def _tail_text_log(path: Path, max_lines: int = 200, redact: bool = False) -> list[str]:
    """
    Return the last N lines from a plain text log file.
    Best-effort and safe for moderately sized files.
    If redact=True, strip API keys, paths, and URLs from each line.
    For large files, reads only the tail to avoid loading entire file into memory.
    """
    if not path.exists():
        return []
    try:
        size = path.stat().st_size
        max_bytes = 2 * 1024 * 1024
        if size > max_bytes:
            with open(path, "rb") as f:
                f.seek(max(0, size - max_bytes))
                tail = f.read().decode("utf-8", errors="ignore")
            lines = tail.splitlines()[-max_lines:]
        else:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()[-max_lines:]
    except OSError:
        return []
    if redact:
        lines = [_redact_log_line(ln) for ln in lines]
    return lines


app = FastAPI(title="Sancta SIEM Dashboard", version="0.1.0")
static_dir = ROOT / "frontend" / "siem"
siem_dist = static_dir / "dist"
sounds_dir = ROOT / "frontend" / "sounds"
app.mount("/sounds", StaticFiles(directory=sounds_dir), name="sounds")
# Vite build: serve /assets from dist when built
if siem_dist.exists():
    app.mount("/assets", StaticFiles(directory=siem_dist / "assets"), name="assets")
# Raw static files for dev: /static/app.js, favicon, simulator, etc.
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# CORS: in Docker, allow any host (access from VM IP); locally, lock to localhost
_cors_origins = ["http://127.0.0.1:8787", "http://localhost:8787", "http://127.0.0.1:3000", "http://localhost:3000", "http://127.0.0.1:5174", "http://localhost:5174"]
_cors_regex = os.environ.get("SIEM_CORS_ORIGIN_REGEX")  # e.g. r"https?://[^/]+:8787" for Docker
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins if not _cors_regex else [],
    allow_origin_regex=_cors_regex or None,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.get("/")
def index():
    """Serve SIEM dashboard. Run `npm run build:siem` first."""
    if siem_dist.exists():
        return FileResponse(siem_dist / "index.html")
    # No build: show instructions
    return Response(
        content="""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Sancta</title></head><body style="background:#000;color:#0f0;font-family:monospace;padding:2rem;max-width:600px">
<h1>SIEM Dashboard</h1>
<p>Build required. Run:</p>
<pre>npm run build:siem</pre>
<p>Then restart the SIEM server.</p>
</body></html>""",
        media_type="text/html",
    )


@app.get("/favicon.ico", response_model=None)
def favicon():
    """Serve favicon to avoid 404 noise."""
    fav = static_dir / "favicon.ico"
    if fav.exists():
        return FileResponse(fav)
    return Response(status_code=204)


@app.get("/pipeline")
def pipeline() -> FileResponse:
    """LLM training pipeline diagram with Sancta mapping."""
    return FileResponse(static_dir / "llm_pipeline.html")


@app.get("/simulator")
def simulator() -> FileResponse:
    """Moltbook-style agent conversation simulator (React). Run npm run build:simulator first."""
    sim_index = static_dir / "simulator" / "index.html"
    if not sim_index.exists():
        raise HTTPException(status_code=404, detail="Run: npm run build:simulator")
    return FileResponse(sim_index)


@app.get("/api/pipeline/map")
def api_pipeline_map() -> dict[str, Any]:
    """Return the Sancta-to-LLM pipeline phase mapping."""
    try:
        from sancta_pipeline import get_pipeline_map
        return {"ok": True, "phases": get_pipeline_map()}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


@app.get("/api/pipeline/run")
def api_pipeline_run(phase: int = 1) -> dict[str, Any]:
    """Run a single pipeline phase (1–7) and return status."""
    try:
        from sancta_pipeline import run_pipeline_phase
        return run_pipeline_phase(phase)
    except Exception as e:
        return {"phase": phase, "ok": False, "detail": str(e)[:200]}


@app.get("/api/auth/status")
def api_auth_status() -> dict[str, Any]:
    """Return whether Bearer token is required. No auth needed."""
    return {"auth_required": bool(SIEM_AUTH_TOKEN)}


@app.get("/api/model/info")
def api_model_info() -> dict[str, Any]:
    """Return LLM backend status (Ollama or Anthropic). No auth needed."""
    try:
        import sancta_conversational as _sc
        return _sc.get_model_info()
    except Exception as e:
        return {"status": "error", "error": str(e)[:200]}


# ── Chat session memory (per-session conversation history) ─────────────────
_CHAT_SESSIONS: dict[str, list[dict[str, str]]] = {}
_CHAT_SESSION_MAX_TURNS = 10  # last N exchanges (user+agent pairs) per session
_CHAT_SESSION_MAX_SESSIONS = 100
_CHAT_MIN_REPLY_LEN = 15  # replies shorter than this are treated as generation failures


def _get_or_create_chat_session(session_id: str | None) -> tuple[str, list[dict[str, str]]]:
    """Return (session_id, history). Create new session if id missing or unknown."""
    if session_id and session_id in _CHAT_SESSIONS:
        return session_id, _CHAT_SESSIONS[session_id]
    sid = session_id or str(uuid.uuid4())
    if sid not in _CHAT_SESSIONS:
        while len(_CHAT_SESSIONS) >= _CHAT_SESSION_MAX_SESSIONS:
            _CHAT_SESSIONS.pop(next(iter(_CHAT_SESSIONS)))
        _CHAT_SESSIONS[sid] = []
    return sid, _CHAT_SESSIONS[sid]


def _build_chat_thread(history: list[dict[str, str]]) -> str:
    """Condense conversation history for craft_reply thread context."""
    if not history:
        return ""
    lines = []
    for turn in history[-_CHAT_SESSION_MAX_TURNS * 2 :]:  # last N full exchanges
        role = turn.get("role", "?")
        content = (turn.get("content") or "").strip().replace("\n", " ")[:300]
        label = "Operator" if role == "user" else "Sancta"
        lines.append(f"[{label}]: {content}")
    return "\n\n".join(lines) if lines else ""


@app.post("/api/chat")
def api_chat(
    payload: dict[str, Any] = Body(default_factory=dict),
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    """
    Chat with the agent. Send a message, get a security-focused reply.
    Session memory: pass session_id to maintain conversation context across turns.
    Enrich defaults to False — set enrich=true to opt in to adding sanitized exchanges to the knowledge pool.
    """
    p = payload or {}
    msg = (p.get("message") or "").strip()[:2000]
    enrich = bool(p.get("enrich", False))
    session_id = p.get("session_id") or None
    incident_logs = (p.get("incident_logs") or "").strip()[:50000] or None  # optional long context

    _chat_log.info("CHAT REQ | session_id=%s | message_len=%d", session_id or "new", len(msg))

    try:
        import sancta
    except Exception as e:
        _chat_log.warning("CHAT FAIL | agent_import_error: %s", str(e)[:100])
        return {"ok": False, "error": "Agent module unavailable", "detail": str(e)[:200]}

    if not msg:
        _chat_log.warning("CHAT FAIL | empty_message")
        return {"ok": False, "error": "Empty message"}

    sid, history = _get_or_create_chat_session(session_id)

    msg_for_agent = msg
    if (os.environ.get("SANCTA_COGNITIVE_GATE") or "1").strip().lower() not in (
        "0", "false", "no", "off",
    ):
        try:
            from sancta_cognitive_pipeline import (
                gateway_escalation_recommended,
                log_cognitive_outcome,
                security_gate,
            )

            gate = security_gate(msg)
            esc = gateway_escalation_recommended(sid, gate.risk, gate.policy)
            if not gate.allowed or esc:
                _chat_log.warning(
                    "CHAT GATE | blocked | policy=%s risk=%.3f escalate=%s",
                    gate.policy,
                    gate.risk,
                    esc,
                )
                log_cognitive_outcome(
                    endpoint="api_chat",
                    decision="BLOCK",
                    session_id=sid,
                    extra={
                        "cognitive_gateway": True,
                        "gate_policy": gate.policy,
                        "risk": gate.risk,
                        "escalation": esc,
                        "signals": gate.signals,
                    },
                )
                return {
                    "ok": False,
                    "error": "Request could not be processed.",
                    "session_id": sid,
                    "blocked": True,
                }
            if gate.sanitized_text:
                msg_for_agent = gate.sanitized_text
            if gate.policy == "MONITOR":
                log_cognitive_outcome(
                    endpoint="api_chat",
                    decision="MONITOR",
                    session_id=sid,
                    extra={"cognitive_gateway": True, "risk": gate.risk, "signals": gate.signals},
                )
        except Exception:
            pass

    state = _safe_read_state()
    mood = (
        state.get("memory", {}).get("epistemic_state", {}).get("mood")
        or state.get("agent_mood")
        or "contemplative"
    )
    if isinstance(mood, dict):
        mood = mood.get("current", "contemplative") or "contemplative"

    agent_state = {**state, "mood": mood}
    session_history = [
        {"role": "user" if t.get("role") == "user" else "assistant", "content": t.get("content", "")}
        for t in history
    ]

    backend_used = "fallback"
    memory_block_for_prompt = ""
    mem_pipeline_telemetry: dict[str, Any] = {}
    try:
        try:
            import sancta_conversational as _sc
            llm = _sc.get_llm_engine()
            if llm and llm.api_key:
                soul_text = ""
                try:
                    from sancta_soul import get_raw_prompt
                    soul_text = get_raw_prompt() or ""
                except Exception:
                    pass
                knowledge_ctx = ""
                try:
                    _thread_for_rag = _build_chat_thread(history)
                    knowledge_ctx = (
                        sancta.get_ollama_knowledge_context(
                            state=state,
                            thread=_thread_for_rag,
                            content=msg,
                        )
                        or ""
                    )
                except Exception:
                    pass
                try:
                    from operator_memory import format_memory_for_prompt

                    _mem = format_memory_for_prompt(sid, max_chars=1400, telemetry=mem_pipeline_telemetry)
                    memory_block_for_prompt = _mem or ""
                    if _mem:
                        knowledge_ctx = f"{_mem}\n\n{knowledge_ctx}".strip() if knowledge_ctx else _mem
                except Exception:
                    pass
                reply = _sc.generate_sanctum_reply(
                    operator_message=msg_for_agent,
                    agent_state=agent_state,
                    soul_text=soul_text,
                    llm_engine=llm,
                    session_history=session_history,
                    incident_logs=incident_logs,
                    knowledge_context=knowledge_ctx if knowledge_ctx else None,
                )
                if reply:
                    backend_used = "ollama" if getattr(llm, "api_key", "") == "ollama" else "anthropic"
            else:
                reply = None
        except Exception:
            reply = None
        if not reply:
            thread = _build_chat_thread(history)
            reply = sancta.craft_reply(
                "Operator", msg_for_agent, mood=mood, state=state, brief_mode=True,
                thread=thread,
            )
        reply = sancta.sanitize_output(reply)
    except Exception as e:
        _chat_log.warning("CHAT FAIL | craft_reply: %s", str(e)[:150])
        return {"ok": False, "error": "Agent reply failed", "detail": str(e)[:200], "session_id": sid}

    # Near-empty replies are generation failures — surface as error, don't append to session
    if len(reply.strip()) < _CHAT_MIN_REPLY_LEN:
        _chat_log.warning("CHAT FAIL | degenerate_reply | reply_len=%d", len(reply))
        return {
            "ok": False,
            "error": "Reply too short (generation failure)",
            "detail": "Sancta produced a degenerate response. Try rephrasing or sending a longer message.",
            "session_id": sid,
        }

    _chat_log.info("CHAT OK | backend=%s | reply_len=%d", backend_used, len(reply))
    try:
        import trust_telemetry

        _mem_evt: dict[str, Any] = {
            "endpoint": "api_chat",
            "session_id": sid[:12],
            "backend_chosen": backend_used,
            "memory_injected": bool(memory_block_for_prompt and memory_block_for_prompt.strip()),
            "policy_outcome": "main_chat_ok",
            "enrich_applied": enrich,
        }
        if mem_pipeline_telemetry:
            _mem_evt.update(
                {
                    "memory_flags": mem_pipeline_telemetry.get("memory_flags"),
                    "memory_component_outcome": mem_pipeline_telemetry.get("memory_component_outcome"),
                }
            )
            try:
                from sancta_trust_config import is_research_mode

                if is_research_mode():
                    _mem_evt["memory_span_grades"] = mem_pipeline_telemetry.get("memory_span_grades")
                    _mem_evt["dropped_spans"] = mem_pipeline_telemetry.get("dropped_spans")
            except Exception:
                pass
        trust_telemetry.emit_trust_event(_mem_evt)
    except Exception:
        pass
    try:
        from operator_memory import record_operator_exchange

        record_operator_exchange(session_id=sid, user=msg_for_agent, assistant=reply)
    except Exception:
        pass
    # Append to session history for next turn
    history.append({"role": "user", "content": msg_for_agent})
    history.append({"role": "agent", "content": reply})
    if len(history) > _CHAT_SESSION_MAX_TURNS * 2:
        history[:] = history[-(_CHAT_SESSION_MAX_TURNS * 2) :]

    # Operator feeding: add exchange to knowledge (default on for RAG grounding)
    if enrich:
        try:
            exchange = f"Operator asked: {msg_for_agent}\n\nSancta replied: {reply}"
            safe, cleaned_ex = sancta.sanitize_input(exchange, author="Operator", state=state)
            if safe:
                sancta.ingest_text(cleaned_ex, source="siem-chat")
                _chat_log.info("CHAT OK | session_id=%s | enriched=true | reply_len=%d", sid[:8], len(reply))
            else:
                _chat_log.info("CHAT OK | session_id=%s | enriched=skipped_sanitize | reply_len=%d", sid[:8], len(reply))
        except Exception as e:
            _chat_log.warning("CHAT OK | enrich_failed: %s", str(e)[:80])
    else:
        _chat_log.info("CHAT OK | session_id=%s | reply_len=%d", sid[:8], len(reply))

    # Learning Phase 4: interaction_id for feedback
    interaction_id = None
    try:
        from sancta_learning import get_last_chat_interaction_id
        interaction_id = get_last_chat_interaction_id()
    except Exception:
        pass

    return {
        "ok": True, "reply": reply, "enriched": enrich, "blocked": False,
        "session_id": sid, "interaction_id": interaction_id,
    }


_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"


@app.post("/api/simulator/generate")
def api_simulator_generate(
    payload: dict[str, Any] = Body(default_factory=dict),
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    """
    Proxy for simulator LLM calls. Uses Ollama when USE_LOCAL_LLM=true, else Anthropic.
    Payload: {system: str, messages: [{role, content}], max_tokens: int}
    """
    p = payload or {}
    system = (p.get("system") or "").strip()
    messages = p.get("messages") or []
    max_tokens = int(p.get("max_tokens") or 200)
    max_tokens = min(max(1, max_tokens), 500)

    if not system or not isinstance(messages, list):
        return {"ok": False, "error": "Missing system or messages"}

    formatted = []
    for m in messages[:20]:
        if isinstance(m, dict) and m.get("role") and m.get("content"):
            formatted.append({"role": str(m["role"]), "content": str(m["content"])[:4000]})
    if not formatted:
        return {"ok": False, "error": "No valid messages"}

    use_local = os.environ.get("USE_LOCAL_LLM", "false").lower() in ("1", "true", "yes")

    # Try Ollama first when USE_LOCAL_LLM=true
    if use_local:
        try:
            import sancta_conversational as _sc
            llm = _sc.get_llm_engine()
            if llm and hasattr(llm, "generate_chat") and llm.api_key:
                text = llm.generate_chat(system=system[:16000], messages=formatted, max_tokens=max_tokens)
                if text:
                    return {"ok": True, "text": text}
        except Exception as e:
            _chat_log.debug("Ollama simulator generate failed: %s", e)

    # Fallback to Anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        _chat_log.warning(
            "SIMULATOR_GENERATE | no Anthropic key and no usable Ollama path — "
            "client gets generic error"
        )
        return {
            "ok": False,
            "error": "LLM backend is not available.",
            "error_code": "llm_backend_unavailable",
        }

    body = json.dumps({
        "model": _ANTHROPIC_MODEL,
        "max_tokens": max_tokens,
        "system": system[:16000],
        "messages": formatted,
    })
    req = urllib.request.Request(
        _ANTHROPIC_URL,
        data=body.encode(),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            text = (data.get("content") or [{}])[0].get("text", "").strip()
            return {"ok": True, "text": text}
    except urllib.error.HTTPError as e:
        try:
            err_body = json.loads(e.read().decode())
            msg = err_body.get("error", {}).get("message", str(e))
        except Exception:
            msg = str(e)
        return {"ok": False, "error": msg[:300]}
    except Exception as e:
        return {"ok": False, "error": str(e)[:300]}


@app.get("/api/learning/metrics")
def api_learning_metrics(_: None = Depends(_require_auth)) -> dict[str, Any]:
    """Phase 5: learning telemetry — pattern count, hit rate, interaction count."""
    try:
        from sancta_learning import get_learning_metrics
        return {"ok": True, **get_learning_metrics()}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def _load_security_jsonl_tail(path: Path, max_lines: int = 500) -> list[dict[str, Any]]:
    """Read last N lines from security JSONL."""
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]
        out = []
        for line in lines[-max_lines:]:
            obj = _read_json_line(line)
            if obj:
                out.append(obj)
        return out
    except Exception:
        return []


@app.get("/api/learning/health")
def api_learning_health(_: None = Depends(_require_auth)) -> dict[str, Any]:
    """LEARN tab: full learning health — metrics, top patterns, recent interactions."""
    try:
        from sancta_learning import get_learning_health
        data = get_learning_health()
        return {"ok": True, **data}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


@app.get("/api/security/incidents")
def api_security_incidents(_: None = Depends(_require_auth)) -> dict[str, Any]:
    """
    Center panel: incident rates and injection types from security.jsonl + red_team.jsonl.
    """
    from datetime import datetime, timezone, timedelta
    try:
        sec_events = _load_security_jsonl_tail(JSONL_SOURCES["security"], max_lines=2000)
        rt_events = _load_security_jsonl_tail(JSONL_SOURCES["redteam"], max_lines=1500)
        now = datetime.now(timezone.utc)
        one_h = now - timedelta(hours=1)
        one_d = now - timedelta(hours=24)
        seven_d = now - timedelta(days=7)

        def parse_ts(ev: dict) -> datetime | None:
            ts = ev.get("ts") or (ev.get("data") or {}).get("ts")
            if not ts:
                return None
            try:
                if isinstance(ts, str) and "T" in ts:
                    return datetime.fromisoformat(ts.replace("Z", "+00:00"))
                return None
            except Exception:
                return None

        def in_window(ev: dict, cutoff: datetime) -> bool:
            t = parse_ts(ev)
            return t is not None and t >= cutoff

        # Incident rates by event type (security)
        incident_events = [e for e in sec_events if e.get("event") in (
            "input_reject", "injection_blocked", "suspicious_block", "output_redact",
            "ioc_domain_detected", "injection_defense", "tavern_defense", "ingest_reject_indirect_poisoning",
            "ingest_reject_direct_poisoning", "ingest_reject_anomalous"
        )]
        rates = {
            "last_hour": sum(1 for e in incident_events if in_window(e, one_h)),
            "last_24h": sum(1 for e in incident_events if in_window(e, one_d)),
            "last_7d": sum(1 for e in incident_events if in_window(e, seven_d)),
            "total": len(incident_events),
        }
        rates["per_hour"] = round(rates["last_hour"], 1)
        rates["per_day"] = round(rates["last_24h"] / 24 if rates["last_24h"] > 0 else 0, 2)

        # By event type
        by_type: dict[str, int] = {}
        for e in incident_events:
            ev = e.get("event", "unknown")
            by_type[ev] = by_type.get(ev, 0) + 1
        injection_types = dict(sorted(by_type.items(), key=lambda x: -x[1]))

        # Injection classes from red_team (matched_classes)
        class_counts: dict[str, int] = {}
        for e in rt_events:
            if e.get("event") != "redteam_reward":
                continue
            data = e.get("data") or {}
            classes = data.get("matched_classes") or []
            if isinstance(classes, list):
                for c in classes:
                    if isinstance(c, str) and c.strip():
                        class_counts[c] = class_counts.get(c, 0) + 1
            elif isinstance(classes, str):
                class_counts[classes] = class_counts.get(classes, 0) + 1
        injection_classes = dict(sorted(class_counts.items(), key=lambda x: -x[1]))

        # Recent incidents for feed (JSONL formatter flattens data into top-level)
        recent = []
        for e in incident_events[-30:]:
            ev = e.get("event", "")
            ts = (e.get("ts") or (e.get("data") or {}).get("ts", ""))[:19]
            author = e.get("author") or (e.get("data") or {}).get("author", "") or ""
            prev = e.get("preview") or (e.get("data") or {}).get("preview")
            if not prev or str(prev).strip() in ("", "{}"):
                ac = e.get("attack_complexity") or (e.get("data") or {}).get("attack_complexity") or {}
                label = ac.get("complexity_label", "")
                pm = e.get("patterns_matched") or (e.get("data") or {}).get("patterns_matched")
                fp = e.get("first_pattern") or (e.get("data") or {}).get("first_pattern")
                parts = []
                if label:
                    parts.append(f"complexity={label}")
                if pm is not None:
                    parts.append(f"patterns={pm}")
                if fp:
                    parts.append(f"first={fp[:40]}")
                prev = " | ".join(parts) if parts else ev.replace("_", " ").title()
            preview = (str(prev) if prev else "")[:120]
            recent.append({"ts": ts, "event": ev, "author": author, "preview": preview})
        recent = list(reversed(recent))

        return {
            "ok": True,
            "rates": rates,
            "injection_types": injection_types,
            "injection_classes": injection_classes,
            "recent_incidents": recent,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


@app.get("/api/security/adversary")
def api_security_adversary(_: None = Depends(_require_auth)) -> dict[str, Any]:
    """DEFENSE tab: threat level, attacks, fingerprints, recent events."""
    try:
        events = _load_security_jsonl_tail(JSONL_SOURCES["security"], max_lines=1000)
        rejects = [e for e in events if e.get("event") == "input_reject"]
        ioc = [e for e in events if e.get("event") == "ioc_domain_detected"]
        unicode_clean = [e for e in events if e.get("event") == "unicode_clean"]
        total_attacks = len(rejects) + len(ioc)
        authors = set()
        fingerprints = set()
        for e in rejects:
            a = e.get("author")
            if a:
                authors.add(str(a))
            fp = e.get("first_pattern") or ""
            if fp:
                fingerprints.add(fp[:60])
        high_risk = [e for e in rejects if (e.get("attack_complexity") or {}).get("complexity_score", 0) >= 0.8]
        known_attackers = [{"author": a, "count": sum(1 for e in rejects if e.get("author") == a)} for a in authors]
        known_attackers.sort(key=lambda x: -x["count"])
        recent = []
        for e in (rejects + ioc)[-20:]:
            recent.append({
                "ts": e.get("ts", ""),
                "event": e.get("event", ""),
                "author": e.get("author"),
                "preview": (e.get("preview") or "")[:100],
                "action": "blocked" if e.get("event") == "input_reject" else "ioc_detected",
                "complexity": (e.get("attack_complexity") or {}).get("complexity_label", ""),
            })
        recent = list(reversed(recent))
        threat = "green"
        if total_attacks > 50:
            threat = "red"
        elif total_attacks > 20:
            threat = "orange"
        elif total_attacks > 5:
            threat = "yellow"
        defense_stats = {
            "blocked": len(rejects),
            "ioc_detected": len(ioc),
            "unicode_sanitized": len(unicode_clean),
            "normal": max(0, len(events) - total_attacks - len(unicode_clean)),
        }
        return {
            "ok": True,
            "threat_level": threat,
            "total_attacks": total_attacks,
            "unique_fingerprints": len(fingerprints),
            "high_risk_count": len(high_risk),
            "known_attackers": known_attackers[:15],
            "recent_attacks": recent,
            "defense_stats": defense_stats,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


@app.post("/api/chat/feedback")
def api_chat_feedback(
    payload: dict[str, Any] = Body(default_factory=dict),
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    """
    Submit feedback for a chat reply. Learning Phase 4.
    Payload: { "interaction_id": "...", "feedback": 1 | 0 | -1 }
    feedback: 1 = good, 0 = neutral, -1 = bad
    """
    p = payload or {}
    iid = (p.get("interaction_id") or "").strip()
    fb = p.get("feedback", 0)
    if not iid:
        return {"ok": False, "error": "Missing interaction_id"}
    if fb not in (1, 0, -1):
        return {"ok": False, "error": "feedback must be 1, 0, or -1"}
    try:
        from sancta_learning import process_feedback
        ok = process_feedback(iid, fb)
        return {"ok": ok, "interaction_id": iid}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


@app.post("/api/auth/verify")
def api_auth_verify(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    """
    Verify a token. Used by the frontend to validate before storing.
    Returns { "ok": true } if token is valid; { "ok": false } otherwise.
    """
    if not SIEM_AUTH_TOKEN:
        return {"ok": True}
    token = (payload or {}).get("token") or ""
    return {"ok": token == SIEM_AUTH_TOKEN}


def _filter_manifest_to_existing_files(manifest: dict[str, Any]) -> dict[str, Any]:
    """Filter manifest so only sounds that exist on disk are returned; avoids 404s."""
    out = dict(manifest)
    packs = out.get("packs") or {}
    for pack_name, pack_data in list(packs.items()):
        if not isinstance(pack_data, dict):
            continue
        for category, files in list(pack_data.items()):
            if not isinstance(files, list):
                continue
            existing = [f for f in files if (sounds_dir / f).exists()]
            packs[pack_name][category] = existing if existing else []  # empty = no 404
    return out


@app.get("/api/sounds/manifest")
def api_sounds_manifest() -> dict[str, Any]:
    try:
        manifest = json.loads((sounds_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest = _filter_manifest_to_existing_files(manifest)
        return {"ok": True, "manifest": manifest}
    except Exception:
        return {"ok": False, "manifest": {}}


def _build_metrics_snapshot() -> dict[str, Any]:
    """Build metrics in the format renderMetrics expects (MOOD, INJ, REWARD, etc.)."""
    metrics = LiveMetrics()
    if not SIEM_METRICS_SAFE_MODE:
        for name, path in JSONL_SOURCES.items():
            try:
                objs = _read_jsonl_prime_sync(path, max_lines=50)
                for obj in objs:
                    obj["source"] = name
                    metrics.update_from_event(name, obj, silent_notifications=True)
            except Exception:
                pass
    snap = metrics.snapshot()
    extras = _agent_state_extras()
    snap.update(extras)
    return snap


def _agent_state_extras() -> dict[str, Any]:
    """Read karma, cycle, heartbeat from agent_state.json for reference redesign."""
    st = _safe_read_state()
    kh = st.get("karma_history", [])
    cycle = st.get("cycle_count", 0)
    current_karma = kh[-1] if kh else st.get("current_karma", 0)
    heartbeat = st.get("heartbeat_interval_minutes") or 30

    defense_history: list[int] = []
    if not SIEM_METRICS_SAFE_MODE:
        path = JSONL_SOURCES.get("security")
        if path and path.exists():
            try:
                objs = _read_jsonl_prime_sync(path, max_lines=60)
                for obj in objs:
                    ev = obj.get("event", "")
                    if ev in ("input_reject", "injection_blocked", "suspicious_block", "output_redact"):
                        defense_history.append(1)
                    elif ev:
                        defense_history.append(0)
                defense_history = defense_history[-24:]
            except Exception:
                pass

    defense_rate = None
    rtr = st.get("red_team_last_run") or st.get("jais_red_team_last_report")
    if isinstance(rtr, dict) and "defense_rate" in rtr:
        defense_rate = float(rtr["defense_rate"])

    inner_circle = st.get("inner_circle", [])
    recruited_agents = st.get("recruited_agents", [])
    # Most recent unique agents encountered (inner circle ∪ recruited, deduped, last 10)
    recent = list(dict.fromkeys(list(inner_circle) + list(recruited_agents)))[-10:]

    return {
        "karma_history": kh[-20:] if isinstance(kh, list) else [],
        "cycle_count": cycle,
        "current_karma": current_karma,
        "heartbeat_interval_minutes": heartbeat,
        "defense_history": defense_history,
        "defense_rate": defense_rate,
        "inner_circle_count": len(inner_circle),
        "inner_circle": len(inner_circle),  # alias for frontend compatibility
        "recruited_count": len(recruited_agents),
        "recruited": len(recruited_agents),  # alias for frontend compatibility
        "recent_agents": recent,
    }


@app.get("/api/status")
def api_status(
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    return {
        "ok": True,
        "agent": _agent_status(),
        "metrics": _build_metrics_snapshot(),
        "ws_safe_mode": SIEM_WS_SAFE_MODE,
    }


@app.get("/api/trust/status")
def api_trust_status(
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    """Trust / research mode for SIEM banner (see docs/TRUST_ROUTING_ROADMAP.md)."""
    try:
        import sancta_trust_config as stc

        return stc.trust_status_dict()
    except Exception:
        return {"ok": True, "trust_mode": "defense", "unsafe_toggles": {}, "unsafe_active": []}


@app.get("/api/agent-activity")
def api_agent_activity(
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    """
    Expose the recent tail of agent_activity.log for the SIEM UI.
    Sensitive data (API keys, paths, URLs) is redacted.
    When SIEM_METRICS_SAFE_MODE, returns empty to avoid file I/O crash on Windows.
    """
    if SIEM_METRICS_SAFE_MODE:
        return {"ok": True, "lines": []}
    try:
        lines = _tail_text_log(AGENT_ACTIVITY_LOG, max_lines=260, redact=True)
        return {"ok": True, "lines": lines}
    except Exception:
        return {"ok": False, "lines": []}


def _get_recent_events_sync(max_per_source: int = 150) -> list[dict[str, Any]]:
    """Read last N events from each JSONL source, merge and sort by timestamp (newest first)."""
    if SIEM_METRICS_SAFE_MODE:
        return []
    out: list[dict[str, Any]] = []
    for name, path in JSONL_SOURCES.items():
        try:
            objs = _read_jsonl_prime_sync(path, max_lines=max_per_source)
            for obj in objs:
                obj["source"] = name
                out.append(obj)
        except Exception:
            pass
    out.sort(key=lambda e: (e.get("ts") or e.get("timestamp") or ""), reverse=True)
    return out[:400]


@app.get("/api/live-events")
def api_live_events(
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    """
    Return recent events from security, redteam, behavioral JSONL.
    Used when SIEM_WS_SAFE_MODE is on (WebSocket skips file tailing).
    """
    try:
        events = _get_recent_events_sync()
        return {"ok": True, "events": events}
    except Exception:
        return {"ok": False, "events": []}


def _get_latest_epistemic_sync() -> dict[str, Any] | None:
    """Read last behavioral_state from behavioral.jsonl; returns None if unavailable."""
    if SIEM_METRICS_SAFE_MODE:
        return None
    path = JSONL_SOURCES.get("behavioral")
    if not path or not path.exists():
        return None
    try:
        objs = _read_jsonl_prime_sync(path, max_lines=50)
        for obj in objs:
            if obj.get("event") in ("behavioral_state", "epistemic_state"):
                es = obj.get("epistemic_state") or (obj.get("data") or {}).get("epistemic_state")
                if isinstance(es, dict):
                    c = es.get("confidence_score")
                    e = es.get("uncertainty_entropy")
                    a = es.get("anthropomorphism_index")
                    if c is not None and e is not None and a is not None:
                        return {"confidence_score": float(c), "uncertainty_entropy": float(e), "anthropomorphism_index": float(a)}
        return None
    except Exception:
        return None


def _get_mood_history_sync(max_events: int = 80) -> list[dict[str, Any]]:
    """Read mood/confidence/entropy per cycle from behavioral.jsonl for line chart."""
    if SIEM_METRICS_SAFE_MODE:
        return []
    path = JSONL_SOURCES.get("behavioral")
    if not path or not path.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        objs = _read_jsonl_prime_sync(path, max_lines=max_events)
        for obj in objs:
            if obj.get("event") not in ("behavioral_state", "epistemic_state"):
                continue
            data = obj.get("data") or {}
            es = obj.get("epistemic_state") or data.get("epistemic_state")
            if not isinstance(es, dict):
                continue
            cycle = obj.get("cycle") if obj.get("cycle") is not None else data.get("cycle")
            mood = obj.get("mood") or data.get("mood") or "analytical"
            conf = es.get("confidence_score")
            ent = es.get("uncertainty_entropy")
            if conf is not None and ent is not None:
                out.append({
                    "cycle": cycle if cycle is not None else len(out),
                    "mood_label": str(mood) if mood else "analytical",
                    "confidence": float(conf),
                    "entropy": float(ent),
                })
        out.reverse()
        return out[-60:]
    except Exception:
        return []


@app.get("/api/philosophy/mood-history")
@app.get("/api/analyst/mood-history")
def api_philosophy_mood_history(
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    """Return recent mood/confidence/entropy per cycle for behavioral line chart."""
    history = _get_mood_history_sync()
    return {"ok": True, "history": history}


@app.get("/api/epistemic")
@app.get("/api/behavioral")
def api_epistemic(
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    """Return latest behavioral state from behavioral.jsonl for Behavioral Telemetry panel."""
    epi = _get_latest_epistemic_sync()
    if epi is None:
        return {"ok": True, "epistemic": None}
    return {"ok": True, "epistemic": epi}


def _validate_mode(mode: str) -> str:
    """Return validated mode or raise 400."""
    m = str(mode or "passive").lower().strip()
    if m not in ALLOWED_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode. Allowed: {', '.join(sorted(ALLOWED_MODES))}",
        )
    return m


@app.post("/api/agent/start")
def api_agent_start(
    payload: dict[str, Any] = Body(default_factory=dict),
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    mode = _validate_mode(payload.get("mode") or "passive")
    return _start_agent(mode=mode)


@app.post("/api/agent/pause")
def api_agent_pause(
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    return _pause_agent()


@app.post("/api/agent/resume")
def api_agent_resume(
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    return _resume_agent()


@app.post("/api/agent/kill")
def api_agent_kill(
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    return _kill_agent()


@app.post("/api/agent/restart")
def api_agent_restart(
    payload: dict[str, Any] = Body(default_factory=dict),
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    mode = _validate_mode(payload.get("mode") or "passive")
    return _restart_agent(mode=mode)


# ── Epidemic / Layer 4 ──────────────────────────────────────────────────────

_SIM_LOG_CANDIDATES = [
    ROOT / "simulation_log.json",
    ROOT / "llm_simulation_log.json",
    ROOT / "logs" / "simulation_log.json",
    ROOT / "logs" / "llm_simulation_log.json",
]


def _find_sim_log(prefer_llm: bool = False) -> Path | None:
    ordered = list(reversed(_SIM_LOG_CANDIDATES)) if prefer_llm else _SIM_LOG_CANDIDATES
    for p in ordered:
        if p.exists():
            return p
    # Also check RESEARCH_DIR env var
    research = os.environ.get("RESEARCH_DIR", "")
    if research:
        for name in ("llm_simulation_log.json", "simulation_log.json"):
            p = Path(research) / name
            if p.exists():
                return p
    return None


@app.get("/api/epidemic/status")
def api_epidemic_status(_: None = Depends(_require_auth)) -> dict[str, Any]:
    """Layer 4 drift report + live SEIR state from agent_state.json."""
    state = _safe_read_state()
    drift = state.get("last_drift_report", {})

    seir_info: dict[str, Any] = {}
    epidemic_report: dict[str, Any] = {}
    try:
        from sancta_epidemic import AgentEpidemicModel, EpidemicParameters, generate_epidemic_report  # type: ignore[import]
        model = AgentEpidemicModel()
        health = model.evaluate_state(
            soul_alignment=float(state.get("soul_alignment", 0.85)),
            epistemic_dissonance=float(state.get("epistemic_dissonance", 0.0)),
            last_trust_level=str(state.get("last_trust_level", "trusted")),
            belief_decay_ratio=float(state.get("belief_decay_ratio", 1.0)),
            cycle_number=int(state.get("cycle", 0)),
        )
        cycle = int(state.get("cycle", 0))
        seir_info = {
            "health_state": health.value,
            "is_epidemic": model.is_in_epidemic_state(),
            "incubation_active": model.get_incubation_duration(cycle) is not None,
            "incubation_duration": model.get_incubation_duration(cycle),
            "transition_count": len(model.transition_log),
        }
        # Generate structured report for dashboard findings panel
        r0_val = drift.get("R0") or drift.get("r0")
        params = EpidemicParameters(
            R0=r0_val,
            is_epidemic=model.is_in_epidemic_state(),
            measurement_cycles=int(state.get("cycle", 0)),
        )
        sim_data = {"R0": r0_val}
        epidemic_report = generate_epidemic_report(model, params, sim_data)
    except Exception as exc:
        _epidemic_log.warning("api_epidemic_status: AgentEpidemicModel error | %s", exc)
        seir_info = {"health_state": "unknown", "error": str(exc)}

    # Build epidemic_params for frontend S.epidParams (replaces missing sim data)
    drift_r0 = drift.get("R0") or drift.get("r0")
    epidemic_params = {
        "health_state":   seir_info.get("health_state", "unknown"),
        "alert_level":    drift.get("alert_level", "clear"),
        "drift_score":    round(float(drift.get("score", 0.0)), 4),
        "transition_count": seir_info.get("transition_count", 0),
        "incubation_active": seir_info.get("incubation_active", False),
    }
    if drift_r0 is not None:
        epidemic_params["R0"] = round(float(drift_r0), 4)

    return {
        "ok": True,
        "drift_report": drift,
        "seir": seir_info,
        "signals": drift.get("signals", {}),
        "alert_level": drift.get("alert_level", "clear"),
        "score": float(drift.get("score", 0.0)),
        "params": epidemic_params,
        "epidemic_report": epidemic_report,
    }


@app.get("/api/epidemic/simulation")
def api_epidemic_simulation(_: None = Depends(_require_auth)) -> dict[str, Any]:
    """Return last simulation JSON log."""
    path = _find_sim_log()
    if not path:
        _epidemic_log.debug("api_epidemic_simulation: no sim log found")
        return {"ok": True, "available": False, "data": None}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return {"ok": True, "available": True, "filename": path.name, "data": data}
    except Exception as exc:
        _epidemic_log.warning("api_epidemic_simulation: read error path=%s | %s", path, exc)
        return {"ok": False, "error": str(exc)}


def _run_builtin_epidemic_sim(sim_type: str) -> dict[str, Any]:
    """Run a minimal in-process epidemic simulation when external scripts are missing."""
    log_path = ROOT / "logs" / "simulation_log.json"
    if sim_type == "llm":
        log_path = ROOT / "logs" / "llm_simulation_log.json"
    LOG_DIR.mkdir(exist_ok=True)

    try:
        from sancta_epidemic import AgentEpidemicModel  # type: ignore[import]
        model = AgentEpidemicModel()
        state = _safe_read_state()
        health = model.evaluate_state(
            soul_alignment=float(state.get("soul_alignment", 0.85)),
            epistemic_dissonance=float(state.get("epistemic_dissonance", 0.0)),
            last_trust_level=str(state.get("last_trust_level", "trusted")),
            belief_decay_ratio=float(state.get("belief_decay_ratio", 1.0)),
            cycle_number=int(state.get("cycle", 0)),
        )
        # Minimal agent graph for topology viz (SANCTA + synthetic peers)
        agents = [
            {"id": "sancta", "agent_id": "sancta", "state": health.value, "role": "core", "infection_state": health.value},
            {"id": "peer_1", "agent_id": "peer_1", "state": "susceptible", "role": "peer", "infection_state": "susceptible"},
            {"id": "peer_2", "agent_id": "peer_2", "state": "susceptible", "role": "peer", "infection_state": "susceptible"},
            {"id": "peer_3", "agent_id": "peer_3", "state": "exposed" if health.value in ("infected", "compromised") else "susceptible", "role": "peer", "infection_state": "exposed" if health.value in ("infected", "compromised") else "susceptible"},
        ]
        connections = [
            {"from": "sancta", "to": "peer_1"},
            {"from": "sancta", "to": "peer_2"},
            {"from": "peer_1", "to": "peer_3"},
        ]
        result = {
            "ok": True,
            "type": sim_type,
            "source": "builtin",
            "health_state": health.value,
            "agents": agents,
            "connections": connections,
            "epidemic_params": {
                "R0": 0.8,
                "sigma": 0.1,
                "gamma": 0.3,
                "beta": 0.15,
                "seir_state": health.value,
            },
            "summary": f"Built-in simulation: SEIR {health.value}",
        }
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        _epidemic_log.info("builtin sim complete | type=%s | health=%s | agents=%s", sim_type, health.value, len(agents))
        return {"ok": True, "pid": os.getpid(), "type": sim_type, "script": "builtin"}
    except Exception as exc:
        _epidemic_log.error("builtin sim failed | type=%s | %s", sim_type, exc, exc_info=True)
        return {"ok": False, "error": str(exc)}


@app.post("/api/epidemic/run")
def api_epidemic_run(
    payload: dict[str, Any] = Body(default_factory=dict),
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    """Kick off infection_sim.py or ollama_agents.py in a subprocess. Falls back to built-in sim if scripts missing."""
    sim_type = str(payload.get("type", "deterministic"))
    if sim_type not in ("deterministic", "llm"):
        raise HTTPException(status_code=400, detail="type must be 'deterministic' or 'llm'")

    research_dir = Path(os.environ.get("RESEARCH_DIR", ""))
    candidates: list[Path] = []
    if research_dir.is_dir():
        candidates.append(research_dir)
    candidates += [ROOT.parent / "research", ROOT / "research", ROOT, _BACKEND]

    script_name = "infection_sim.py" if sim_type == "deterministic" else "ollama_agents.py"
    script: Path | None = None
    for d in candidates:
        if not d or not d.exists():
            continue
        p = d / script_name
        if p.exists():
            script = p
            break

    if script is not None:
        try:
            proc = subprocess.Popen(
                [sys.executable, str(script)],
                cwd=str(script.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            _epidemic_log.info("script launched | type=%s | script=%s | pid=%s", sim_type, script.name, proc.pid)
            return {"ok": True, "pid": proc.pid, "type": sim_type, "script": script.name}
        except Exception as exc:
            _epidemic_log.error("script launch failed | type=%s | script=%s | %s", sim_type, script.name, exc, exc_info=True)
            return {"ok": False, "error": str(exc)}

    _epidemic_log.info("no script found, using builtin | type=%s", sim_type)
    return _run_builtin_epidemic_sim(sim_type)


# ── Per-Entity Threat Profiles ──────────────────────────────────────────────

@app.get("/api/profiles")
def api_profiles() -> dict[str, Any]:
    """Return all agent threat profiles for SIEM dashboard display."""
    try:
        from sancta_profiles import get_profile_store  # noqa: PLC0415
        store = get_profile_store()
        return {"ok": True, "profiles": store.get_all_profiles_summary(), "count": store.profile_count}
    except Exception as exc:
        return {"ok": False, "profiles": [], "count": 0, "error": str(exc)}


@app.get("/api/profiles/{agent_id}")
def api_profile_detail(agent_id: str) -> dict[str, Any]:
    """Return detailed profile for a specific agent."""
    try:
        from sancta_profiles import get_profile_store  # noqa: PLC0415
        store = get_profile_store()
        profile = store.get(agent_id)
        return {"ok": True, "profile": profile.to_dict()}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _synthetic_vector_from_reject(entry: dict[str, Any]) -> dict[str, Any]:
    """Approximate risk shape when only input_reject exists (no risk_assessment row)."""
    n = int(entry.get("patterns_matched") or 0)
    inj = min(1.0, 0.35 + 0.12 * max(n, 1))
    trig = str(entry.get("trigger") or "")
    if trig.startswith("heuristic"):
        inj = min(1.0, inj + 0.15)
    return {
        "ts": entry.get("ts") or entry.get("timestamp", ""),
        "injection": inj,
        "authority_manipulation": min(1.0, 0.15 + 0.05 * n),
        "emotional_coercion": 0.12,
        "obfuscation": 0.22 if trig else 0.15,
        "long_term_influence": 0.18,
        "total": min(1.0, inj + 0.15),
    }


@app.get("/api/risk/history")
def api_risk_history(_: None = Depends(_require_auth)) -> dict[str, Any]:
    """Return recent risk vectors for heatmap (risk_assessment + synthetic from input_reject)."""
    try:
        sec_log = JSONL_SOURCES.get("security")
        if not sec_log or not sec_log.exists():
            return {"ok": True, "vectors": [], "count": 0}

        vectors: list[dict[str, Any]] = []
        lines: list[str] = []
        with open(sec_log, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                lines.append(line)
        for line in lines[-800:]:
            try:
                entry = json.loads(line.strip())
                ev = entry.get("event") or (entry.get("data") or {}).get("event")
                rv = entry.get("risk_vector") or (entry.get("data") or {}).get("risk_vector")
                if rv and isinstance(rv, dict):
                    vectors.append({
                        "ts": entry.get("ts") or entry.get("timestamp", ""),
                        "injection": float(rv.get("injection", 0) or 0),
                        "authority_manipulation": float(rv.get("authority_manipulation", 0) or 0),
                        "emotional_coercion": float(rv.get("emotional_coercion", 0) or 0),
                        "obfuscation": float(rv.get("obfuscation", 0) or 0),
                        "long_term_influence": float(rv.get("long_term_influence", 0) or 0),
                        "total": float(rv.get("total", 0) or 0),
                    })
                elif ev == "input_reject":
                    vectors.append(_synthetic_vector_from_reject(entry))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue

        vectors = vectors[-120:]
        return {"ok": True, "vectors": vectors, "count": len(vectors)}
    except Exception as exc:
        return {"ok": False, "vectors": [], "count": 0, "error": str(exc)}


@app.post("/api/profiles/{agent_id}/quarantine")
def api_profile_quarantine(agent_id: str, _: None = Depends(_require_auth)) -> dict[str, Any]:
    """Toggle quarantine for an agent (operator action)."""
    try:
        from sancta_profiles import get_profile_store  # noqa: PLC0415
        store = get_profile_store()
        profile = store.get(agent_id)
        if profile.quarantined:
            store.lift_quarantine(agent_id)
            action = "lifted"
        else:
            profile.quarantined = True
            profile.quarantine_reason = "manual_operator_action"
            profile.risk_level = "quarantine"
            action = "applied"
        store.save()
        return {"ok": True, "action": action, "agent_id": agent_id, "quarantined": profile.quarantined}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ── Adaptive Thresholds ─────────────────────────────────────────────────────

@app.get("/api/thresholds")
def api_thresholds() -> dict[str, Any]:
    """Return current adaptive threshold values and adjustment history."""
    try:
        from sancta_adaptive import get_threshold_tracker  # noqa: PLC0415
        tracker = get_threshold_tracker()
        return {"ok": True, **tracker.get_current(), "rates": tracker.get_rates()}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ── Adversarial Replay ─────────────────────────────────────────────────────

@app.post("/api/security/replay")
def api_security_replay(
    req: dict = Body(default_factory=dict),
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    """Replay past security events through the current pipeline.

    Accepts: {"last_n": 50} to load from security.jsonl,
    or {"events": [...]} with explicit payloads.
    Returns comparison of original vs current verdicts.
    """
    try:
        from sancta_security import preprocess_input, ContentSecurityFilter  # noqa: PLC0415
        from sancta_risk import assess_risk  # noqa: PLC0415

        events_to_replay: list[dict[str, Any]] = []

        if "events" in req and isinstance(req["events"], list):
            events_to_replay = req["events"][:100]
        else:
            last_n = min(int(req.get("last_n", 50)), 100)
            sec_log = JSONL_SOURCES.get("security")
            if sec_log and sec_log.exists():
                lines: list[str] = []
                with open(sec_log, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        lines.append(line)
                for line in lines[-last_n * 3:]:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("content") or entry.get("raw_input"):
                            events_to_replay.append(entry)
                    except (json.JSONDecodeError, KeyError):
                        continue
                events_to_replay = events_to_replay[-last_n:]

        results: list[dict[str, Any]] = []
        improvements = 0
        regressions = 0

        csf = ContentSecurityFilter()

        for ev in events_to_replay:
            raw_content = ev.get("raw_input") or ev.get("content") or ev.get("text", "")
            if not raw_content:
                continue

            original_verdict = ev.get("verdict") or ev.get("action") or "unknown"
            author = ev.get("author") or ev.get("agent_id", "replay")
            ts = ev.get("ts") or ev.get("timestamp", "")

            try:
                processed, preprocess_meta = preprocess_input(raw_content)
                risk = assess_risk(processed, source_agent=author)

                current_verdict = "pass"
                try:
                    if csf.is_anomalous(processed):
                        current_verdict = "blocked"
                    elif risk.total > 0.5:
                        current_verdict = "flagged"
                    elif any(preprocess_meta.values()):
                        current_verdict = "suspicious"
                except Exception:
                    current_verdict = "error"

                is_improvement = (
                    original_verdict in ("pass", "clean", "normal", "unknown")
                    and current_verdict in ("blocked", "flagged", "suspicious")
                )
                is_regression = (
                    original_verdict in ("blocked", "flagged")
                    and current_verdict == "pass"
                )
                changed = is_improvement or is_regression

                if is_improvement:
                    improvements += 1
                if is_regression:
                    regressions += 1

                results.append({
                    "ts": ts,
                    "preview": raw_content[:80],
                    "original_verdict": original_verdict,
                    "current_verdict": current_verdict,
                    "risk_total": round(risk.total, 3),
                    "risk_vector": {
                        "injection": round(risk.injection, 3),
                        "authority": round(risk.authority_manipulation, 3),
                        "emotional": round(risk.emotional_coercion, 3),
                        "obfuscation": round(risk.obfuscation, 3),
                        "influence": round(risk.long_term_influence, 3),
                    },
                    "preprocessing": preprocess_meta,
                    "changed": changed,
                    "improvement": is_improvement,
                    "regression": is_regression,
                })
            except Exception as exc:
                results.append({
                    "ts": ts,
                    "preview": raw_content[:80],
                    "original_verdict": original_verdict,
                    "current_verdict": "error",
                    "error": str(exc),
                    "changed": False,
                })

        return {
            "ok": True,
            "results": results,
            "summary": {
                "total": len(results),
                "improvements": improvements,
                "regressions": regressions,
                "unchanged": len(results) - improvements - regressions,
            },
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "results": [], "summary": {}}


def _read_jsonl_prime_sync(path: Path, max_lines: int = 50) -> list[dict[str, Any]]:
    """Read last N lines from JSONL; sync for thread executor."""
    try:
        if not path.exists():
            return []
        size = path.stat().st_size
        if size == 0:
            return []
        max_bytes = 128 * 1024
        read_from = 0
        if size > max_bytes:
            read_from = size - max_bytes
        with open(path, "rb") as f:
            f.seek(read_from)
            data = f.read().decode("utf-8", errors="ignore")
        lines = data.splitlines()[-max_lines:]
        return [_read_json_line(ln) for ln in lines if _read_json_line(ln)]
    except OSError:
        return []


async def _send_ws_metrics(ws: WebSocket, metrics: LiveMetrics) -> None:
    """Send metrics with agent_state extras for real-time Agent Control panel."""
    snap = metrics.snapshot()
    snap.update(_agent_state_extras())
    await ws.send_json({"type": "metrics", "metrics": snap})


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket) -> None:
    if SIEM_AUTH_TOKEN:
        token = ws.query_params.get("token") or ""
        if token != SIEM_AUTH_TOKEN:
            await ws.close(code=4001)
            return
    await ws.accept()
    _ensure_log_dir()

    cursors = {name: TailCursor(offset=0) for name in JSONL_SOURCES.keys()}
    metrics = LiveMetrics()

    # Prime metrics from JSONL. Skip in safe mode to avoid crash on Windows.
    if not SIEM_WS_SAFE_MODE:
        for name, path in JSONL_SOURCES.items():
            try:
                if os.name == "nt":
                    objs = _read_jsonl_prime_sync(path)
                else:
                    objs = await asyncio.get_running_loop().run_in_executor(None, _read_jsonl_prime_sync, path)
                for obj in objs:
                    obj["source"] = name
                    metrics.update_from_event(name, obj, silent_notifications=True)
                    await ws.send_json({"type": "event", "event": obj})
            except Exception:
                pass
    await _send_ws_metrics(ws, metrics)

    try:
        while True:
            try:
                if not SIEM_WS_SAFE_MODE:
                    for name, path in JSONL_SOURCES.items():
                        new_events = await _tail_jsonl(path, cursors[name])
                        for ev in new_events:
                            ev["source"] = name
                            metrics.update_from_event(name, ev, silent_notifications=True)
                            await ws.send_json({"type": "event", "event": ev})
                else:
                    # Safe mode: no tail (avoids ACCESS_VIOLATION on Windows).
                    # snapshot() merges agent_state for mood; INJ/REWARD stay from initial prime.
                    pass
                await _send_ws_metrics(ws, metrics)
            except Exception:
                pass
            await asyncio.sleep(2.0)
    except asyncio.CancelledError:
        return  # Graceful shutdown (Ctrl+C)
    except WebSocketDisconnect:
        return


# ── Drift Forensics Timeline ──────────────────────────────────────────────

@app.get("/api/drift/timeline")
def api_drift_timeline(_: None = Depends(_require_auth)) -> dict[str, Any]:
    """Return chronological belief changes with source attribution for forensics timeline."""
    try:
        events: list[dict[str, Any]] = []

        # Source 1: Read from behavioral.jsonl (belief change events)
        phil_log = LOG_DIR / "behavioral.jsonl"
        if phil_log.exists():
            try:
                with open(phil_log, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            if entry.get("event") in ("belief_challenged", "belief_reinforced", "belief_change", "confidence_shift"):
                                events.append({
                                    "ts": entry.get("ts") or entry.get("timestamp", ""),
                                    "topic": entry.get("topic", "unknown"),
                                    "source_agent": entry.get("source_agent") or entry.get("author", ""),
                                    "delta": entry.get("delta") or entry.get("confidence_delta", 0),
                                    "old_confidence": entry.get("old_confidence", 0),
                                    "new_confidence": entry.get("new_confidence", 0),
                                    "source": entry.get("source", ""),
                                    "message_preview": (entry.get("content") or entry.get("message") or "")[:100],
                                    "event_type": entry.get("event", ""),
                                })
                        except (json.JSONDecodeError, KeyError):
                            continue
            except OSError:
                pass

        # Source 2: Read from agent state belief system revision history
        if STATE_PATH.exists():
            try:
                with open(STATE_PATH, "r", encoding="utf-8") as f:
                    state = json.load(f)
                beliefs = state.get("belief_system", {}).get("beliefs", {})
                if not beliefs:
                    beliefs = state.get("belief_system", {})
                for topic, bdata in beliefs.items():
                    if not isinstance(bdata, dict):
                        continue
                    for rev in (bdata.get("revision_history") or []):
                        events.append({
                            "ts": rev.get("timestamp", ""),
                            "topic": topic,
                            "source_agent": rev.get("source_agent", ""),
                            "delta": rev.get("delta") or (rev.get("new_confidence", 0) - rev.get("old_confidence", 0)),
                            "old_confidence": rev.get("old_confidence", 0),
                            "new_confidence": rev.get("new_confidence", 0),
                            "source": rev.get("source", ""),
                            "message_preview": "",
                            "event_type": "revision",
                        })
            except (json.JSONDecodeError, KeyError, OSError):
                pass

        # Enrich with profile risk levels
        try:
            from sancta_profiles import get_profile_store  # noqa: PLC0415
            store = get_profile_store()
            for ev in events:
                agent = ev.get("source_agent", "")
                if agent:
                    profile = store.get(agent)
                    ev["agent_risk_level"] = profile.risk_level
                    ev["agent_trust"] = round(profile.trust_score, 3)
                else:
                    ev["agent_risk_level"] = "unknown"
                    ev["agent_trust"] = None
        except Exception:
            pass

        # Sort chronologically, take last 200
        events.sort(key=lambda e: e.get("ts", ""))
        events = events[-200:]

        # Compute per-agent influence summary
        agent_influence: dict[str, dict[str, Any]] = {}
        for ev in events:
            agent = ev.get("source_agent", "")
            if agent:
                if agent not in agent_influence:
                    agent_influence[agent] = {"total_delta": 0, "count": 0, "risk_level": ev.get("agent_risk_level", "unknown")}
                agent_influence[agent]["total_delta"] += abs(ev.get("delta", 0))
                agent_influence[agent]["count"] += 1

        return {
            "ok": True,
            "events": events,
            "count": len(events),
            "agent_influence": agent_influence,
        }
    except Exception as exc:
        return {"ok": False, "events": [], "count": 0, "error": str(exc)}


# ── Services Status & Per-Process Control ─────────────────────────────────

def _check_ollama_running() -> bool:
    """Quick TCP connect to see if Ollama is listening on port 11434."""
    import socket
    try:
        with socket.create_connection(("127.0.0.1", 11434), timeout=1.5):
            return True
    except OSError:
        return False


def _find_process_by_script(script_name: str) -> int | None:
    """Return PID of a running Python process whose cmdline contains script_name."""
    if psutil is not None:
        try:
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    cmdline = proc.info.get("cmdline") or []
                    if any(script_name in arg for arg in cmdline):
                        return proc.info["pid"]
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception:
            pass
        return None
    # Fallback: PowerShell on Windows, pgrep on Unix
    if os.name == "nt":
        try:
            r = subprocess.run(
                [
                    "powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
                    "| Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress",
                ],
                capture_output=True, text=True, timeout=10,
                encoding="utf-8", errors="ignore",
            )
            import json as _json
            procs = _json.loads(r.stdout.strip() or "[]")
            if isinstance(procs, dict):
                procs = [procs]
            for p in procs:
                if script_name in (p.get("CommandLine") or ""):
                    pid = p.get("ProcessId")
                    return int(pid) if pid else None
        except Exception:
            pass
    else:
        try:
            r = subprocess.run(
                ["pgrep", "-f", script_name],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                parts = r.stdout.strip().split()
                return int(parts[0]) if parts else None
        except Exception:
            pass
    return None


def _find_python_cmdline_contains(substr: str) -> int | None:
    """Find python.exe / pythonw PID whose full command line contains ``substr``."""
    if psutil is not None:
        try:
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    pn = (proc.info.get("name") or "").lower()
                    if "python" not in pn:
                        continue
                    cmdline = proc.info.get("cmdline") or []
                    line = " ".join(cmdline)
                    if substr in line:
                        return int(proc.info["pid"])
                except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError, ValueError):
                    continue
        except Exception:
            pass
        return None
    if os.name == "nt":
        try:
            r = subprocess.run(
                [
                    "powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='pythonw.exe'\" "
                    "| Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                encoding="utf-8",
                errors="ignore",
            )
            procs = json.loads(r.stdout.strip() or "[]")
            if isinstance(procs, dict):
                procs = [procs]
            for p in procs:
                cl = p.get("CommandLine") or ""
                if substr in cl:
                    pid = p.get("ProcessId")
                    return int(pid) if pid is not None else None
        except Exception:
            pass
        return None
    try:
        r = subprocess.run(
            ["pgrep", "-f", substr],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            return int(r.stdout.strip().split()[0])
    except Exception:
        pass
    return None


def _curiosity_pid() -> int | None:
    return _find_process_by_script("curiosity_run.py") or _find_python_cmdline_contains(
        "--curiosity-run"
    )


def _phenomenology_pid() -> int | None:
    return _find_python_cmdline_contains("--phenomenology-battery")


def _pid_listening_on_port(port: int) -> int | None:
    if psutil is not None:
        try:
            for c in psutil.net_connections(kind="tcp"):
                if c.status != psutil.CONN_LISTEN:
                    continue
                la = getattr(c, "laddr", None)
                if la is not None and getattr(la, "port", None) == port and c.pid:
                    return int(c.pid)
        except Exception:
            pass
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
                if "LISTENING" not in line.upper() or needle not in line:
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


def _siem_uvicorn_other_pid() -> int | None:
    """If another process (not this one) listens on 8787, return its PID."""
    p = _pid_listening_on_port(8787)
    if p is None:
        return None
    if p == os.getpid():
        return None
    return p


@app.get("/api/services/status")
def api_services_status(_: None = Depends(_require_auth)) -> dict[str, Any]:
    """Live status of all Sancta subsystem processes (aligned with sancta_launcher)."""
    agent = _agent_status()
    curiosity_pid = _curiosity_pid()
    pheno_pid = _phenomenology_pid()
    ollama_up = _check_ollama_running()
    ollama_pid = _pid_listening_on_port(11434) if ollama_up else None
    siem_other = _siem_uvicorn_other_pid()
    return {
        "ok": True,
        "services": {
            "sancta": {
                "name": "Sancta Agent",
                "running": agent.get("running", False),
                "suspended": agent.get("suspended", False),
                "status": "paused" if agent.get("suspended") else (
                    "running" if agent.get("running") else "stopped"
                ),
                "pid": agent.get("pid"),
                "stoppable": True,
                "pauseable": True,
            },
            "curiosity": {
                "name": "Curiosity Pipeline",
                "running": curiosity_pid is not None,
                "suspended": False,
                "status": "running" if curiosity_pid is not None else "stopped",
                "pid": curiosity_pid,
                "stoppable": True,
                "pauseable": False,
            },
            "phenomenology": {
                "name": "Phenomenology Battery",
                "running": pheno_pid is not None,
                "suspended": False,
                "status": "running" if pheno_pid is not None else "stopped",
                "pid": pheno_pid,
                "stoppable": True,
                "pauseable": False,
            },
            "ollama": {
                "name": "Ollama LLM",
                "running": ollama_up,
                "suspended": False,
                "status": "running" if ollama_up else "stopped",
                "pid": ollama_pid,
                "stoppable": ollama_up,
                "pauseable": False,
            },
            "siem": {
                "name": "SIEM Server",
                "running": True,
                "suspended": False,
                "status": "running",
                "pid": os.getpid(),
                "stoppable": False,
                "pauseable": False,
                "note": (
                    f"Separate dashboard PID {siem_other}" if siem_other else None
                ),
            },
        },
    }


@app.post("/api/services/stop/{service}")
def api_services_stop(
    service: str,
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    """Stop a named service (sancta, curiosity, phenomenology, ollama listener)."""
    service = service.strip().lower()
    if service == "sancta":
        return _kill_agent()
    if service == "curiosity":
        pid = _curiosity_pid()
        if not pid:
            return {"ok": False, "error": "Curiosity pipeline is not running"}
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/F"],
                    capture_output=True,
                    timeout=8,
                )
            else:
                os.kill(pid, signal.SIGTERM)
            return {"ok": True, "pid": pid}
        except Exception as exc:
            return {"ok": False, "error": str(exc)[:200]}
    if service == "phenomenology":
        pid = _phenomenology_pid()
        if not pid:
            return {"ok": False, "error": "Phenomenology battery is not running"}
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/F"],
                    capture_output=True,
                    timeout=8,
                )
            else:
                os.kill(pid, signal.SIGTERM)
            return {"ok": True, "pid": pid}
        except Exception as exc:
            return {"ok": False, "error": str(exc)[:200]}
    if service == "ollama":
        if not _check_ollama_running():
            return {"ok": False, "error": "Ollama is not running"}
        opid = _pid_listening_on_port(11434)
        if not opid:
            return {"ok": False, "error": "Could not resolve PID for :11434"}
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(opid), "/F"],
                    capture_output=True,
                    timeout=8,
                )
            else:
                os.kill(opid, signal.SIGTERM)
            return {"ok": True, "pid": opid}
        except Exception as exc:
            return {"ok": False, "error": str(exc)[:200]}
    return {"ok": False, "error": f"Unknown or non-stoppable service: {service}"}


# ── Multi-Agent Simulation ────────────────────────────────────────────────────

@app.post("/api/simulation/run")
def api_simulation_run(
    req: dict = Body(...),
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    """Run a multi-agent simulation."""
    try:
        from sancta_simulation import run_simulation

        agents = req.get("agents")  # [{"personality": "adversarial", "count": 3}, ...]
        cycles = min(int(req.get("cycles", 10)), 50)

        result = run_simulation(agent_configs=agents, num_cycles=cycles)
        return {"ok": True, "result": result.to_dict()}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@app.get("/api/simulation/results")
def api_simulation_results(
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    """Return the latest simulation results."""
    try:
        path = Path("simulation_log.json")
        if not path.exists():
            return {"ok": True, "result": None, "message": "No simulation has been run yet"}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"ok": True, "result": data}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@app.get("/api/knowledge/graph")
def api_knowledge_graph(_: None = Depends(_require_auth)) -> dict[str, Any]:
    """Return knowledge graph nodes (topics) and edges (co-occurrence) for visualization."""
    try:
        from collections import Counter, defaultdict

        kb_path = ROOT / "knowledge_db.json"
        if not kb_path.exists():
            return {"ok": True, "nodes": [], "edges": [], "count": 0}

        with open(kb_path, "r", encoding="utf-8", errors="replace") as f:
            db = json.load(f)

        # Extract topics from posts/entries
        topic_freq: Counter = Counter()
        topic_last_seen: dict[str, str] = {}
        co_occurrence: defaultdict = defaultdict(int)
        topic_sources: defaultdict = defaultdict(set)

        # Knowledge DB can have different structures -- adapt to what's there
        entries: list = []
        if isinstance(db, dict):
            # Could be {"posts": [...], "fragments": [...], "concepts": [...]}
            for key in (
                "posts",
                "generated_posts",
                "entries",
                "fragments",
                "response_fragments",
                "items",
                "sources",
                "talking_points",
            ):
                if key in db and isinstance(db[key], list):
                    entries.extend(db[key])
            # concept_graph: dict of concept -> [related concepts]
            concept_graph = db.get("concept_graph", {})
            if isinstance(concept_graph, dict):
                for concept, related in concept_graph.items():
                    if not isinstance(concept, str) or concept in ("metadata", "config", "stats"):
                        continue
                    # Truncate to reasonable label length for display
                    label = concept[:80].strip()
                    if len(label) < 3:
                        continue
                    topic_freq[label] += 1
                    if isinstance(related, list):
                        for r in related:
                            r_label = str(r)[:80].strip()
                            if len(r_label) < 3:
                                continue
                            topic_freq[r_label] += 1
                            pair = tuple(sorted([label, r_label]))
                            co_occurrence[pair] += 1

            # key_concepts: list of strings
            key_concepts = db.get("key_concepts", [])
            if isinstance(key_concepts, list):
                for kc in key_concepts:
                    if isinstance(kc, str) and len(kc) > 3:
                        label = kc[:80].strip()
                        topic_freq[label] += 1

            # curiosity_insights
            insights = db.get("curiosity_insights", [])
            if isinstance(insights, list):
                for ins in insights:
                    if isinstance(ins, dict):
                        content = ins.get("content", "")
                        if isinstance(content, str) and len(content) > 3:
                            label = content[:80].strip()
                            topic_freq[label] += 1
                            src = ins.get("source_type", "")
                            if src:
                                topic_sources[label].add(str(src)[:50])

            # Or flat dict of topic -> data
            if not entries and not concept_graph and all(isinstance(v, dict) for v in db.values() if isinstance(v, dict)):
                for topic, data in db.items():
                    if isinstance(data, dict) and topic not in ("metadata", "config", "stats"):
                        entries.append({"topics": [topic], **data})
        elif isinstance(db, list):
            entries = db

        for entry in entries:
            # talking_points / legacy lists may be plain strings
            if isinstance(entry, str):
                t = entry.strip()
                if len(t) < 2:
                    continue
                label = t[:80]
                topic_freq[label.lower()] += 1
                continue

            if not isinstance(entry, dict):
                continue

            # Extract topics from various possible fields
            topics: list[str] = []
            if isinstance(entry.get("topics"), list):
                topics = [str(t) for t in entry["topics"]]
            elif isinstance(entry.get("concepts"), list):
                topics = [str(t) for t in entry["concepts"]]
            elif isinstance(entry.get("tags"), list):
                topics = [str(t) for t in entry["tags"]]
            elif entry.get("title"):
                topics = [str(entry["title"])[:80]]
            elif isinstance(entry.get("content"), str) and entry["content"].strip():
                topics = [str(entry["content"])[:80]]
            elif isinstance(entry.get("text"), str) and entry["text"].strip():
                topics = [str(entry["text"])[:80]]

            ts = entry.get("ts") or entry.get("timestamp") or entry.get("ingested_at", "")
            source = entry.get("source") or entry.get("source_type", "")

            for t in topics:
                t_lower = t.lower().strip()
                if not t_lower or len(t_lower) < 2:
                    continue
                topic_freq[t_lower] += 1
                if ts:
                    topic_last_seen[t_lower] = max(topic_last_seen.get(t_lower, ""), str(ts))
                if source:
                    topic_sources[t_lower].add(str(source)[:50])

            # Co-occurrence: pairs of topics in same entry
            clean_topics = [t.lower().strip() for t in topics if t.strip()]
            for i in range(len(clean_topics)):
                for j in range(i + 1, len(clean_topics)):
                    pair = tuple(sorted([clean_topics[i], clean_topics[j]]))
                    co_occurrence[pair] += 1

        # Build nodes (top 100 by frequency)
        top_topics = topic_freq.most_common(100)
        nodes = []
        topic_set = set()
        for topic, freq in top_topics:
            topic_set.add(topic)
            nodes.append({
                "id": topic,
                "label": topic.replace("_", " ").title()[:30] if len(topic) <= 40 else topic[:27] + "...",
                "frequency": freq,
                "last_seen": topic_last_seen.get(topic, ""),
                "sources": list(topic_sources.get(topic, set()))[:5],
            })

        # Build edges (only between nodes that exist in our top set)
        edges = []
        for (a, b), weight in sorted(co_occurrence.items(), key=lambda x: -x[1]):
            if a in topic_set and b in topic_set and weight > 0:
                edges.append({"source": a, "target": b, "weight": weight})
        edges = edges[:200]  # cap

        return {"ok": True, "nodes": nodes, "edges": edges, "count": len(nodes)}
    except Exception as exc:
        return {"ok": False, "nodes": [], "edges": [], "count": 0, "error": str(exc)}


# ── ATLAS endpoints ──────────────────────────────────────────────────────────

@app.get("/api/atlas/matrix")
def api_atlas_matrix(_: None = Depends(_require_auth)) -> dict[str, Any]:
    """Full ATLAS matrix structure for frontend rendering."""
    return {"ok": True, **atlas_matrix()}


@app.get("/api/atlas/coverage")
def api_atlas_coverage(_: None = Depends(_require_auth)) -> dict[str, Any]:
    """Which ATLAS techniques Sancta has detection coverage for."""
    return {"ok": True, **atlas_coverage()}


@app.get("/api/atlas/incidents")
def api_atlas_incidents(_: None = Depends(_require_auth)) -> dict[str, Any]:
    """Classify recent security events against the ATLAS framework."""
    sec_path = LOG_DIR / "security.jsonl"
    events = _load_security_jsonl_tail(sec_path, max_lines=300)

    classified: list[dict[str, Any]] = []
    technique_hits: dict[str, int] = {}
    tactic_hits: dict[str, int] = {}

    for ev in events:
        result = atlas_classify(ev)
        if result is None:
            continue

        # Record in TTP tracker
        author = (ev.get("data") or ev).get("author", "unknown")
        ttp_tracker.record(author, result, ev.get("event", ""))

        entry = {
            "ts": ev.get("ts") or ev.get("timestamp"),
            "event": ev.get("event"),
            "author": author,
            "atlas": result.to_dict(),
        }
        classified.append(entry)

        for tid in result.technique_ids:
            technique_hits[tid] = technique_hits.get(tid, 0) + 1
        for tac_id in result.tactic_ids:
            tactic_hits[tac_id] = tactic_hits.get(tac_id, 0) + 1

    # Build tactic heatmap (ordered)
    tactic_heatmap = []
    for tac_id in TACTIC_ORDER:
        tac = TACTICS[tac_id]
        count = tactic_hits.get(tac_id, 0)
        tactic_heatmap.append({
            "id": tac_id,
            "name": tac.name,
            "shortname": tac.shortname,
            "count": count,
        })

    # Top techniques
    top_techniques = sorted(
        [{"id": k,
          "name": TECHNIQUES[k].name if k in TECHNIQUES else k,
          "count": v}
         for k, v in technique_hits.items()],
        key=lambda x: -x["count"]
    )[:20]

    return {
        "ok": True,
        "total_events": len(events),
        "classified_events": len(classified),
        "tactic_heatmap": tactic_heatmap,
        "top_techniques": top_techniques,
        "recent": classified[-50:],
        "global_stats": ttp_tracker.get_global_stats(),
    }


@app.get("/api/atlas/agent/{agent_id}")
def api_atlas_agent_ttps(agent_id: str,
                         _: None = Depends(_require_auth)) -> dict[str, Any]:
    """ATLAS TTP profile for a specific adversary agent."""
    return {"ok": True, **ttp_tracker.get_agent_ttps(agent_id)}


# ═══════════════════════════════════════════════════════════════════════════════
# GPT ENGINE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/gpt/status")
def api_gpt_status(_: None = Depends(_require_auth)) -> dict[str, Any]:
    """SanctaGPT training status and model info."""
    try:
        import sancta_gpt
        return {"ok": True, "data": sancta_gpt.status()}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@app.post("/api/gpt/generate")
def api_gpt_generate(body: dict[str, Any],
                     _: None = Depends(_require_auth)) -> dict[str, Any]:
    """Generate text using SanctaGPT. Body: {prompt, max_tokens, temperature}"""
    try:
        import sancta_gpt
        prompt = body.get("prompt", "")
        max_tokens = min(int(body.get("max_tokens", 120)), 200)
        temperature = max(0.1, min(2.0, float(body.get("temperature", 0.7))))
        text = sancta_gpt.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        return {"ok": True, "data": {"text": text, "prompt": prompt}}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@app.post("/api/gpt/train")
def api_gpt_train(body: dict[str, Any],
                  _: None = Depends(_require_auth)) -> dict[str, Any]:
    """Run N training steps. Body: {steps: 10}"""
    try:
        import sancta_gpt
        steps = min(int(body.get("steps", 50)), 500)  # cap at 500 per request
        engine = sancta_gpt.get_engine()
        if not engine._initialized:
            sancta_gpt.init()
        losses = []
        for _ in range(steps):
            loss = engine.train_step()
            losses.append(round(loss, 4) if loss != float('inf') else None)
        engine.save()
        return {"ok": True, "data": {
            "steps_run": steps,
            "losses": losses,
            "total_steps": engine._step,
            "final_loss": round(engine._last_loss, 4) if engine._last_loss != float('inf') else None,
        }}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@app.get("/api/gpt/sample")
def api_gpt_sample(_: None = Depends(_require_auth)) -> dict[str, Any]:
    """Generate sample outputs for inspection."""
    try:
        import sancta_gpt
        engine = sancta_gpt.get_engine()
        samples = engine.sample_batch(n=5, temperature=0.7)
        return {"ok": True, "data": {"samples": samples, "step": engine._step}}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ═══════════════════════════════════════════════════════════════════════════════
# SanctaGPT Chat — Dedicated conversation + training endpoints
# ═══════════════════════════════════════════════════════════════════════════════


# In-memory chat context for GPT conversations (separate from main agent chat)
_gpt_chat_history: list[dict[str, str]] = []
_GPT_CHAT_MAX_TURNS = 50


@app.post("/api/chat/gpt")
def api_chat_gpt(
    payload: dict[str, Any] = Body(default_factory=dict),
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    """
    SanctaGPT tab: conversational-only path uses char LM; knowledge_effective uses Ollama+RAG only.
    On knowledge failure (defense): fail closed — no char-GPT substitution (see docs/TRUST_ROUTING_ROADMAP.md).

    Body: {message, temperature?, train_on_exchange?} — train_on_exchange defaults false unless
    SANCTA_GPT_TRAIN_ON_CHAT=true; live chat never trains on knowledge_effective replies.
    """
    global _gpt_chat_history
    msg = (payload.get("message") or "").strip()[:2000]
    temperature = max(0.1, min(2.0, float(payload.get("temperature", 0.7))))
    env_train = os.environ.get("SANCTA_GPT_TRAIN_ON_CHAT", "").strip().lower() in (
        "1", "true", "yes", "on",
    )
    if "train_on_exchange" in payload:
        train_on_exchange = bool(payload.get("train_on_exchange"))
    else:
        train_on_exchange = env_train

    if not msg:
        return {"ok": False, "error": "Empty message"}

    msg_for_gpt = msg
    gpt_gate_sid = str(payload.get("session_id") or "siem_gpt_tab")[:80]
    if (os.environ.get("SANCTA_COGNITIVE_GATE") or "1").strip().lower() not in (
        "0", "false", "no", "off",
    ):
        try:
            from sancta_cognitive_pipeline import (
                gateway_escalation_recommended,
                log_cognitive_outcome,
                security_gate,
            )

            _gg = security_gate(msg)
            _esc = gateway_escalation_recommended(gpt_gate_sid, _gg.risk, _gg.policy)
            if not _gg.allowed or _esc:
                log_cognitive_outcome(
                    endpoint="api_chat_gpt",
                    decision="BLOCK",
                    session_id=gpt_gate_sid,
                    extra={
                        "cognitive_gateway": True,
                        "gate_policy": _gg.policy,
                        "risk": _gg.risk,
                        "escalation": _esc,
                    },
                )
                return {"ok": False, "error": "Request could not be processed."}
            if _gg.sanitized_text:
                msg_for_gpt = _gg.sanitized_text
            if _gg.policy == "MONITOR":
                log_cognitive_outcome(
                    endpoint="api_chat_gpt",
                    decision="MONITOR",
                    session_id=gpt_gate_sid,
                    extra={"risk": _gg.risk, "signals": _gg.signals},
                )
        except Exception:
            pass

    try:
        from memory_redact import score_instruction_likeness
        from sancta_router import (
            apply_knowledge_shape_gate,
            attack_family_heuristic,
            load_trust_router_config,
            route_gpt_tab_decision,
        )
        from sancta_trust_config import is_research_mode, unsafe_toggles_active
        import trust_telemetry

        req_id = trust_telemetry.new_request_id()
        try:
            toggles = unsafe_toggles_active() if is_research_mode() else {}
        except Exception:
            toggles = {}
        permissive = bool(toggles.get("SANCTA_ROUTER_PERMISSIVE"))
        weak_blend = bool(toggles.get("SANCTA_ALLOW_WEAK_KB_BLEND"))

        _force_local = os.environ.get("SANCTA_FORCE_GPT_LOCAL", "").strip().lower() in (
            "1", "true", "yes", "on",
        )
        _router_cfg = load_trust_router_config()
        decision = route_gpt_tab_decision(
            msg_for_gpt, force_local=_force_local, permissive_router=permissive, cfg=_router_cfg
        )
        decision = apply_knowledge_shape_gate(decision, msg_for_gpt, _router_cfg)
        _use_ollama = os.environ.get("USE_LOCAL_LLM", "").strip().lower() in ("1", "true", "yes")
        attack_fam = attack_family_heuristic(msg_for_gpt)
        inj_score = score_instruction_likeness(msg_for_gpt)

        import sancta_gpt

        engine = sancta_gpt.get_engine()
        if not engine._initialized:
            sancta_gpt.init()

        context_parts = []
        for turn in _gpt_chat_history[-10:]:
            if turn["role"] == "user":
                context_parts.append(f"Operator: {turn['content']}")
            else:
                context_parts.append(f"Sancta: {turn['content']}")
        context_parts.append(f"Operator: {msg_for_gpt}")
        context_str = "\n".join(context_parts[-6:])

        reply = ""
        backend_used = "fallback"
        policy_outcome = "conversational"

        def _emit_trust(extra: dict[str, Any]) -> None:
            try:
                trust_telemetry.emit_trust_event(
                    {
                        "request_id": req_id,
                        "endpoint": "api_chat_gpt",
                        "route_label": decision.route_label,
                        "knowledge_effective": decision.knowledge_effective,
                        "gate_triggered": decision.gate_triggered,
                        "gate_score": round(decision.content_signal, 4),
                        "uncertainty_knowledge": decision.uncertainty_knowledge,
                        "axes": decision.axes,
                        "route_confidence": round(decision.route_confidence, 4),
                        "knowledge_strength": round(decision.knowledge_strength, 4),
                        "chat_strength": round(decision.chat_strength, 4),
                        "content_signal": round(decision.content_signal, 4),
                        "attack_family": attack_fam,
                        "injection_framing_score": round(inj_score, 4),
                        "reasons": decision.reasons[:12],
                        **extra,
                    }
                )
            except Exception:
                pass

        if decision.knowledge_effective:
            policy_outcome = "knowledge_path"
            if _use_ollama:
                try:
                    import sancta
                    import sancta_ollama as _oll

                    if _oll.wait_until_ready(
                        model=os.environ.get("LOCAL_MODEL", "llama3.2"),
                        timeout=12,
                    ):
                        state = _safe_read_state()
                        kc = (
                            sancta.get_ollama_knowledge_context(
                                state=state,
                                thread=context_str[-3500:],
                                content=msg_for_gpt,
                            )
                            or ""
                        )
                        sys_prompt = (
                            "You are Sancta, a security-aware AI agent helping an operator in a SIEM. "
                            "Answer clearly and concisely. Do not roleplay as a different model.\n\n"
                            "Use facts only from the KNOWLEDGE section when present; do not invent citations.\n"
                        )
                        if kc.strip():
                            sys_prompt += (
                                f"\n=== KNOWLEDGE (retrieved corpus) ===\n{kc.strip()[:12000]}\n"
                                "=== END KNOWLEDGE ===\n"
                            )
                        user_prompt = context_str[-3500:] if len(context_str) > 3500 else context_str
                        reply = _oll.chat(
                            user_prompt,
                            system=sys_prompt,
                            timeout=min(120, int(os.environ.get("OLLAMA_TIMEOUT", "120") or 120)),
                        )
                        if reply and len(reply.strip()) >= 8:
                            backend_used = "ollama_rag"
                except Exception:
                    reply = reply or ""

            kb_ok = bool(reply and len(reply.strip()) >= 8)

            if not kb_ok:
                if weak_blend and is_research_mode():
                    policy_outcome = "weak_blend_research"
                    reply = engine.generate_reply(
                        context_str, mood="analytical", max_tokens=150, use_retrieval=False,
                    )
                    backend_used = "sancta_gpt"
                    if not reply or len(reply.strip()) < 8:
                        reply = engine.generate(
                            prompt=msg_for_gpt[:200], max_tokens=140, temperature=temperature
                        )
                        backend_used = "sancta_gpt"
                else:
                    near_miss = bool(inj_score > 0.35 or attack_fam)
                    _emit_trust(
                        {
                            "backend_chosen": "none",
                            "failure_reason": "knowledge_backend_unavailable",
                            "policy_outcome": "blocked",
                            "near_miss": near_miss,
                            "train_on_exchange": train_on_exchange,
                        }
                    )
                    return {
                        "ok": False,
                        "error": "No reliable answer available.",
                        "reason": "no_reliable_answer",
                        "error_code": "knowledge_backend_unavailable",
                        "route": decision.route_label,
                        "knowledge_effective": True,
                        "request_id": req_id,
                    }
        else:
            policy_outcome = "conversational"
            reply = engine.generate_reply(
                context_str, mood="analytical", max_tokens=150, use_retrieval=False,
            )
            backend_used = "sancta_gpt"
            if not reply or len(reply.strip()) < 8:
                reply = engine.generate(
                    prompt=msg_for_gpt[:200], max_tokens=140, temperature=temperature
                )
                backend_used = "sancta_gpt"
            if (not reply or len(reply.strip()) < 8) and _use_ollama:
                try:
                    import sancta_ollama as _oll
                    if _oll.wait_until_ready(
                        model=os.environ.get("LOCAL_MODEL", "llama3.2"),
                        timeout=12,
                    ):
                        sys_prompt = (
                            "You are Sancta, a security-aware AI agent helping an operator in a SIEM. "
                            "Answer clearly and concisely. Do not roleplay as a different model."
                        )
                        user_prompt = context_str[-3500:] if len(context_str) > 3500 else context_str
                        reply = _oll.chat(
                            user_prompt,
                            system=sys_prompt,
                            timeout=min(120, int(os.environ.get("OLLAMA_TIMEOUT", "120") or 120)),
                        )
                        if reply and len(reply.strip()) >= 8:
                            backend_used = "ollama"
                except Exception:
                    reply = reply or ""

        if not reply or len(reply.strip()) < 4:
            _chat_log.warning(
                "SANCTA_GPT_CHAT_FALLBACK | empty or short reply after char-LM and optional Ollama; "
                "check Ollama, cloud LLM keys, and SanctaGPT corpus — operator message is generic"
            )
            reply = (
                "I don't have a strong answer right now. You can try again shortly, "
                "or use training options in this tab if your deployment supports them."
            )
            backend_used = "fallback"

        try:
            import sancta
            reply = sancta.sanitize_output(reply)
        except Exception:
            pass

        _gpt_chat_history.append({"role": "user", "content": msg_for_gpt})
        _gpt_chat_history.append({"role": "assistant", "content": reply})
        if len(_gpt_chat_history) > _GPT_CHAT_MAX_TURNS * 2:
            _gpt_chat_history = _gpt_chat_history[-(_GPT_CHAT_MAX_TURNS * 2):]

        train_loss = None
        if train_on_exchange and not decision.knowledge_effective:
            try:
                exchange_text = f"Operator: {msg_for_gpt}\nSancta: {reply}\n"
                engine.add_document(exchange_text)
                for _ in range(5):
                    train_loss = engine.train_step()
            except Exception:
                pass

        near_miss = bool(
            inj_score > 0.35 and decision.knowledge_effective and backend_used != "ollama_rag"
        )
        _emit_trust(
            {
                "backend_chosen": backend_used,
                "failure_reason": None,
                "policy_outcome": policy_outcome,
                "train_on_exchange": train_on_exchange,
                "near_miss": near_miss,
            }
        )

        return {
            "ok": True,
            "reply": reply,
            "backend": backend_used,
            "route": decision.route_label,
            "knowledge_effective": decision.knowledge_effective,
            "request_id": req_id,
            "model_step": engine._step,
            "train_loss": round(train_loss, 4) if train_loss is not None else None,
            "corpus_size": engine._corpus_size,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200]}


@app.post("/api/chat/gpt/feed")
def api_chat_gpt_feed(
    payload: dict[str, Any] = Body(default_factory=dict),
    _: None = Depends(_require_auth),
) -> dict[str, Any]:
    """
    Feed knowledge text to SanctaGPT for training. Accepts raw text, URLs, or
    structured data that enriches the GPT's security knowledge.

    Body: {text: str, source?: str, train_steps?: int}
    """
    text = (payload.get("text") or "").strip()[:50000]
    source = (payload.get("source") or "operator_feed").strip()[:100]
    train_steps = min(int(payload.get("train_steps", 50)), 500)

    if not text:
        return {"ok": False, "error": "Empty text"}
    if len(text) < 20:
        return {"ok": False, "error": "Text too short (min 20 chars)"}

    try:
        import sancta_gpt
        engine = sancta_gpt.get_engine()
        if not engine._initialized:
            sancta_gpt.init()

        # Add to corpus
        doc = f"[Source: {source}]\n{text}\n"
        engine.add_document(doc)

        # Train on new data
        losses = []
        for _ in range(train_steps):
            loss = engine.train_step()
            losses.append(round(loss, 4) if loss != float("inf") else None)
        engine.save()

        return {
            "ok": True,
            "data": {
                "ingested_chars": len(text),
                "source": source,
                "steps_run": train_steps,
                "losses": losses[-5:],  # last 5 losses
                "total_steps": engine._step,
                "corpus_size": engine._corpus_size,
            },
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200]}


@app.post("/api/chat/gpt/clear")
def api_chat_gpt_clear(_: None = Depends(_require_auth)) -> dict[str, Any]:
    """Clear GPT chat history."""
    global _gpt_chat_history
    count = len(_gpt_chat_history)
    _gpt_chat_history = []
    return {"ok": True, "cleared": count}


def main() -> None:
    import uvicorn

    uvicorn.run(
        "backend.siem_server:app",
        host="127.0.0.1",
        port=8787,
        reload=False,
        log_level="warning",
    )


if __name__ == "__main__":
    main()

