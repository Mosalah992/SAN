# RESTRUCTURE.md — Sancta Project Cleanup & Reorganization

## Goal
Consolidate the Sancta project into a single, professional folder structure that any developer can navigate within 5 minutes. Remove dead code, unused files, and ambiguous naming.

---

## Target Folder Structure

```
sancta/
├── CLAUDE.md                          # Claude Code instructions
├── README.md                          # Project overview & setup
├── .env                               # Environment configuration
├── .env.example                       # Template with all vars documented
├── requirements.txt                   # Python dependencies (pinned)
├── package.json                       # Node dependencies (if any)
│
├── server/                            # All backend Python code
│   ├── __init__.py
│   ├── app.py                         # ← was siem_server.py (entry point)
│   ├── agent/                         # Agent loop & lifecycle
│   │   ├── __init__.py
│   │   ├── loop.py                    # ← extracted from sancta.py: main agent loop
│   │   ├── lifecycle.py               # ← extracted: start/pause/resume/kill/restart
│   │   ├── state.py                   # ← extracted: agent_state.json read/write, _safe_read_state()
│   │   └── pacing.py                  # ← extracted: time budget, curiosity run pacing
│   ├── content/                       # Content generation & reply handling
│   │   ├── __init__.py
│   │   ├── conversational.py          # ← was sancta_conversational.py
│   │   ├── post_generator.py          # ← extracted from sancta.py
│   │   ├── reply_handler.py           # ← extracted from sancta.py
│   │   └── knowledge.py              # ← extracted: knowledge_db.json management
│   ├── security/                      # Security pipeline (all 5 layers)
│   │   ├── __init__.py
│   │   ├── pipeline.py                # ← was sancta_security.py (ContentSecurityFilter)
│   │   ├── drift.py                   # ← extracted: BehavioralDriftDetector (Layer 4)
│   │   └── ollama_scan.py             # ← extracted: Layer 5 deep scan (USE_LOCAL_LLM)
│   ├── epidemic/                      # SEIR model & simulation
│   │   ├── __init__.py
│   │   ├── model.py                   # ← was sancta_epidemic.py
│   │   └── simulation.py              # ← deterministic sim + ollama_agents.py
│   ├── social/                        # Agent-to-agent communication
│   │   ├── __init__.py
│   │   ├── dm.py                      # ← was sancta_dm.py
│   │   └── belief.py                  # ← was sancta_belief.py
│   ├── learning/                      # Learning health
│   │   ├── __init__.py
│   │   └── health.py                  # ← was sancta_learning.py
│   └── routes/                        # API endpoint definitions
│       ├── __init__.py
│       ├── auth.py                    # /api/auth/*
│       ├── agent.py                   # /api/agent/*, /api/status, /api/agent-activity
│       ├── chat.py                    # /api/chat, /api/chat/feedback
│       ├── security.py                # /api/security/adversary
│       ├── epidemic.py                # /api/epidemic/*
│       ├── model.py                   # /api/model/info
│       ├── epistemic.py               # /api/epistemic
│       └── websocket.py               # /ws/live
│
├── frontend/                          # All frontend code
│   ├── index.html                     # ← was dist/index.html
│   ├── js/
│   │   ├── state.js                   # Singleton S, pushEvent(), evSeverity()
│   │   ├── api.js                     # All 20+ endpoint calls
│   │   ├── boot.js                    # 7-step boot animation
│   │   ├── websocket.js               # WS + exponential backoff + polling fallback
│   │   ├── app.js                     # Auth flow, debounced refresh, live event routing
│   │   └── tabs/
│   │       ├── dashboard.js           # Live appendEvent() + bulk refresh
│   │       ├── security.js            # Adversary API integration
│   │       ├── soul.js                # Beliefs grid, epistemic bars, journal
│   │       ├── chat.js                # Session-persistent chat
│   │       ├── lab.js                 # Security pipeline testing
│   │       ├── epidemic.js            # Animated SVG network graph
│   │       └── control.js             # Lifecycle buttons, process info
│   └── styles/
│       ├── variables.css
│       ├── reset.css
│       ├── chrome.css
│       ├── terminal.css
│       ├── layout.css
│       ├── animations.css
│       └── enhancements.css
│
├── simulator/                         # Standalone simulator app
│   └── App.jsx                        # (keep as-is, reads correct API fields)
│
├── logs/                              # Runtime logs (gitignored)
│   ├── epidemic.log
│   ├── simulation_log.json
│   ├── security.jsonl
│   ├── red_team.jsonl
│   └── philosophy.jsonl
│
├── data/                              # Runtime state files (gitignored)
│   ├── agent_state.json
│   └── knowledge_db.json
│
├── scripts/                           # Utility & maintenance scripts
│   └── launcher.py                    # Process launcher with wait_until_ready()
│
└── tests/                             # Test suite (to be built)
    ├── test_security_pipeline.py
    ├── test_epidemic_model.py
    ├── test_api_shapes.py             # Validates all 32 endpoint response shapes
    └── test_drift_detector.py
```

---

## Migration Steps (In Order)

### Phase 1 — Create target structure (non-destructive)
```bash
# Create all target directories
mkdir -p sancta/{server/{agent,content,security,epidemic,social,learning,routes},frontend/{js/tabs,styles},simulator,logs,data,scripts,tests}

# Create all __init__.py files
find sancta/server -type d -exec touch {}/__init__.py \;
```

### Phase 2 — Move files to new locations

**CRITICAL: Do NOT rename imports until all files are moved. Move first, fix imports second.**

```bash
# Backend — main server
cp siem_server.py sancta/server/app.py

# Backend — existing modules (direct moves)
cp sancta_conversational.py sancta/server/content/conversational.py
cp sancta_security.py sancta/server/security/pipeline.py
cp sancta_epidemic.py sancta/server/epidemic/model.py
cp sancta_dm.py sancta/server/social/dm.py
cp sancta_belief.py sancta/server/social/belief.py
cp sancta_learning.py sancta/server/learning/health.py

# Backend — sancta.py decomposition (the big one)
# This must be done by extracting functions, not copying the whole file
# See Phase 3 below

# Frontend — direct moves
cp dist/index.html sancta/frontend/index.html
cp js/*.js sancta/frontend/js/
cp js/tabs/*.js sancta/frontend/js/tabs/
cp styles/*.css sancta/frontend/styles/

# Simulator
cp frontend/simulator/App.jsx sancta/simulator/App.jsx

# Config
cp .env sancta/.env
cp README.md sancta/README.md
```

### Phase 3 — Decompose sancta.py (the 7988-line monolith)

Extract in this order (each extraction is one commit):

1. **`server/agent/state.py`** — Extract:
   - `_safe_read_state()`, `_write_state()`, `_agent_state_extras()`
   - All agent_state.json I/O
   - The `S` state object if applicable

2. **`server/agent/lifecycle.py`** — Extract:
   - `start_agent()`, `pause_agent()`, `resume_agent()`, `kill_agent()`, `restart_agent()`
   - Process management with psutil fallback

3. **`server/agent/pacing.py`** — Extract:
   - Time budget enforcement
   - Curiosity run pacing logic
   - Sleep/delay calculations

4. **`server/content/post_generator.py`** — Extract:
   - `generate_post()` and related functions
   - Content hash deduplication
   - Template fallback system

5. **`server/content/reply_handler.py`** — Extract:
   - `craft_reply()` (line ~7028)
   - Reply formatting, threading logic

6. **`server/content/knowledge.py`** — Extract:
   - knowledge_db.json read/write
   - Knowledge graph queries

7. **`server/security/drift.py`** — Extract:
   - `BehavioralDriftDetector` class
   - 6 weighted drift signals
   - `_cycle_reports` buffer management

8. **`server/security/ollama_scan.py`** — Extract:
   - Layer 5 LLM deep scan (lines 3634–3682)
   - USE_LOCAL_LLM gating logic

9. **`server/agent/loop.py`** — What remains:
   - The main agent loop
   - Cycle management
   - `_epistemic_state_snapshot()` (line ~150)
   - Imports from all extracted modules

**Extraction rule**: After each extraction, run the full system and verify all 32 endpoints respond correctly. If anything breaks, fix it before moving to the next extraction.

### Phase 4 — Extract routes from siem_server.py

Split the 30+ endpoint definitions into route modules:

```python
# sancta/server/routes/agent.py
from fastapi import APIRouter
agent_router = APIRouter()

@agent_router.get('/api/status')
async def get_status(): ...

@agent_router.post('/api/agent/{action}')
async def agent_action(action: str): ...
```

Then in `server/app.py`:
```python
from routes.auth import auth_router
from routes.agent import agent_router
from routes.chat import chat_router
# ... register all routers
app.include_router(agent_router)
```

### Phase 5 — Fix all import paths

After all files are in their new locations:

```bash
# Find all Python imports that reference old module names
grep -rn "import sancta_" sancta/server/
grep -rn "from sancta_" sancta/server/
grep -rn "import siem_server" sancta/server/

# Fix each one to use the new package paths
# e.g. "from sancta_security import ContentSecurityFilter"
#   → "from server.security.pipeline import ContentSecurityFilter"
```

For the frontend, update `server/app.py` static file paths:
```python
# Old: static files mounted at /static/ → frontend/siem/
# New: static files mounted at /static/ → ../frontend/
```

### Phase 6 — Update frontend paths

In `frontend/index.html`, verify all `<script>` and `<link>` paths still resolve:
```html
<!-- These should work as relative paths since index.html is served as root -->
<script type="module" src="/static/js/app.js"></script>
<link rel="stylesheet" href="/static/styles/variables.css">
```

### Phase 7 — Clean up

**Files to DELETE** (after confirming the new structure works):
- Any `__pycache__/` directories
- Any `.pyc` files
- Duplicate config files
- Old `dist/` directory (replaced by `frontend/`)
- Any `node_modules/` if not needed
- Temporary test files
- Old backup files (`.bak`, `.old`, `.backup`)

**Files to GITIGNORE**:
```gitignore
# Runtime
logs/
data/agent_state.json
data/knowledge_db.json
__pycache__/
*.pyc
node_modules/
.env

# IDE
.vscode/
.idea/
*.swp
```

---

## Files to INVESTIGATE Before Deleting

These files were mentioned in the audit or may exist — check if they're used:

| File | Check | Action |
|------|-------|--------|
| `ollama_agents.py` | Is it imported anywhere? Does it reference any process? | If standalone script, move to `scripts/`. If unused, delete. |
| `sancta_ollama.py` | Referenced by Layer 5 deep scan | Move to `server/security/` or merge into `ollama_scan.py` |
| Any `test_*.py` in root | Ad-hoc tests? | Move to `tests/` or delete if outdated |
| `frontend/siem/` directory | Was the old static mount point | Delete after confirming `frontend/` works |
| `dist/` directory | Was the old frontend location | Delete after confirming `frontend/` works |
| `llm_simulation_log.json` | Unknown output path from ollama_agents.py | Find or document |

---

## Validation Checklist

After restructuring, verify ALL of these pass:

- [ ] `python server/app.py` starts without import errors
- [ ] All 32 API endpoints return correct response shapes
- [ ] WebSocket connects and streams metrics
- [ ] Frontend loads at `http://127.0.0.1:8787`
- [ ] All 7 tabs render correctly
- [ ] Dashboard receives live events (WS or polling)
- [ ] Epidemic tab shows animated network topology
- [ ] Security tab shows adversary data
- [ ] Chat sends and receives messages
- [ ] Lab runs red-team pipeline
- [ ] Control tab shows process info
- [ ] All 5 JSONL log streams write without errors
- [ ] `agent_state.json` reads/writes from new `data/` path
- [ ] Windows safe mode still works (`SIEM_WS_SAFE_MODE=true`)

---

## Do NOT Change

These are working correctly and should only be moved, never modified during restructuring:

- The 7 CSS files (just move to `frontend/styles/`)
- The WebSocket protocol and event routing
- The API response shapes (any shape change breaks frontend)
- The JSONL log format
- The `.env` variable names
- The simulator's API field reads (`m.cycle_count`, `m.current_karma`, `m.agent_mood`)
