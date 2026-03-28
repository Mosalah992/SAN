# Sancta — Autonomous AI Security Analyst

Merged-project operating guide: [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md)

An AI agent that operates as a **blue team security analyst** on [Moltbook](https://www.moltbook.com), publishing threat intelligence, monitoring AI agent behavior for adversarial manipulation, running detection logic against behavioral drift, and building a community of security researchers in m/sentinel-ops.

**SIEM trust / routing (GPT tab vs Ollama, fail-closed knowledge, research telemetry):** [`docs/TRUST_ROUTING_ROADMAP.md`](docs/TRUST_ROUTING_ROADMAP.md) · env/schema [`docs/TRUST_AND_RESEARCH_MODE.md`](docs/TRUST_AND_RESEARCH_MODE.md) · multi-agent lab notes [`docs/ADVERSARIAL_RESEARCH_MODE.md`](docs/ADVERSARIAL_RESEARCH_MODE.md).

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/Mosalah992/sancta.git
cd sancta
pip install -r requirements.txt
pip install -r requirements-dev.txt  # pytest, for running tests
```

**Requirements:** PyTorch, FastAPI, uvicorn, aiohttp, python-dotenv, psutil, pygame. Optional: PEFT, bitsandbytes, TRL, datasets for LoRA/transformer fine-tuning. A GPU is recommended for training but not required for inference.

**Python:** 3.10+ recommended.

**Optional dependencies** (for semantic concept extraction in `sancta_semantic.py`; the agent falls back gracefully if absent):

- `sentence-transformers` — embedding-based concept extraction
- `keybert` — KeyBERT extraction (fallback: YAKE)
- `yake` — YAKE fallback when KeyBERT is unavailable

### 2. Configure

Copy `.env.example` to `.env` and set:

```env
AGENT_NAME=caesarsancta
AGENT_DESCRIPTION=blue team security analyst, threat hunter, SEIR model operator.
MOLTBOOK_API_KEY=          # Leave blank; filled after first registration
MOLTBOOK_CLAIM_URL=         # Filled after registration
HEARTBEAT_INTERVAL_MINUTES=30
```

### 3. Register

```bash
python -m backend.sancta --register
```

Send the `claim_url` to your human so they can verify ownership via tweet. Once claimed, the agent is active.

### 4. Run SIEM Dashboard (optional)

```powershell
.\start_siem.ps1
```

Or manually:

```powershell
python -m uvicorn backend.siem_server:app --host 127.0.0.1 --port 8787
```

Open `http://127.0.0.1:8787` for the dashboard; `http://127.0.0.1:8787/pipeline` for the LLM pipeline diagram.

### 5. LLM Integration (Local)

Sancta supports Ollama + Llama 3.2 for AI-powered SIEM chat and simulator with local long-context (128K tokens).

**Quick start:**

1. **Install Ollama:** [ollama.com/download](https://ollama.com/download) or `winget install Ollama.Ollama`
2. **Setup:** Run `.\scripts\setup_ollama.ps1` (Windows) or `./scripts/setup_ollama.sh` (Linux/Mac)
3. **Start Ollama server:** `ollama serve`
4. **Configure:** Add to `.env`: `USE_LOCAL_LLM=true`, `OLLAMA_URL=http://localhost:11434`, `LOCAL_MODEL=llama3.2`
5. **Start SIEM:** `python -m uvicorn backend.siem_server:app --host 127.0.0.1 --port 8787`

**Model options:**

- `llama3.2` — Fast, lightweight (3B parameters)
- `qwen2.5:14b` — Better quality (14B, requires 32GB RAM)
- `llama3.1:70b` — Best quality (requires GPU)

Update `LOCAL_MODEL` in `.env` to switch. Run `ollama pull <model>` first.

See `docs/LLM_OPERATIONS.md` for daily ops and `docs/DEPLOYMENT_CHECKLIST.md` for deployment validation.

### 6. Run Tests

```bash
cd backend && python -m pytest tests/ -v
```

Tests cover risk scoring, profile management, input preprocessing, belief system, and soul alignment. CI runs automatically on push/PR via GitHub Actions.

---

## Architecture

```
threat feeds + interactions → analysis engine → SOUL → detection / response
     ↑                              ↑
   chat (operator)           operator feeding
```

| Component | Implementation |
|-----------|-----------------|
| **threat feeds** | `knowledge_db.json`, `knowledge/` dir, Moltbook feed, security.jsonl |
| **interactions** | Moltbook API (posts, comments, feed), heartbeat cycle actions |
| **analysis engine** | `sancta.py` orchestration, `sancta_generative.py`, local transformer fragment selector |
| **chat** | SIEM `/api/chat`, `craft_reply()`, enrich flag for operator feeding |
| **SOUL** | `SOUL_SYSTEM_PROMPT.md` (authority) → `sancta_soul.py`, `_evaluate_action()`, mood, `mission_active` |
| **red team** | `security_check_content()`, `_red_team_incoming_pipeline()`, `run_red_team_simulation()`, JAIS |
| **blue team** | `run_policy_test_cycle()`, `--policy-test`, SIEM BLUE TEAM mode |

### Modules

| Module | Responsibility |
|--------|----------------|
| `backend/sancta.py` | Main loop, orchestration, mood/RL logic, knowledge ingestion, Layer 4 heartbeat hook |
| `backend/knowledge_manager.py` | Knowledge DB CRUD, text extraction, concept generation, provenance tagging, trust filtering |
| `backend/sancta_security.py` | Five-layer knowledge defense + `BehavioralDriftDetector` (Layer 4) |
| `backend/sancta_epidemic.py` | WoW SEIR-C-R epidemic model, formal parameter definitions |
| `backend/sancta_decision.py` | Decision engine for action selection, adaptive threshold integration |
| `backend/sancta_adaptive.py` | Self-tuning security thresholds based on FP/miss rates |
| `backend/sancta_simulation.py` | Multi-agent simulation (cooperative, adversarial, manipulative, neutral personalities) |
| `backend/sancta_risk.py` | 5-dimensional risk vector scoring (injection, authority, emotional, obfuscation, influence) |
| `backend/sancta_profiles.py` | Per-entity threat profiles, trust scoring, auto-quarantine |
| `backend/sancta_belief.py` | Analytical position tracking, confidence scoring, drift attribution |
| `backend/sancta_soul.py` | Loads `SOUL_SYSTEM_PROMPT.md` at startup; derives SOUL dict |
| `backend/sancta_generative.py` | Transformer-inspired fragment selection, post generation |
| `backend/sancta_conversational.py` | Conversational engine (Anthropic API + Ollama fallback) |
| `backend/sancta_semantic.py` | Concept extraction (KeyBERT/YAKE optional), cosine dedup |
| `backend/sancta_learning.py` | Interaction capture, pattern learner scaffold |
| `backend/sancta_dm.py` | Agent DM processing and reply |
| `backend/siem_server.py` | SIEM FastAPI app, 45+ API endpoints, WebSocket live events |

### Identity alignment

`SOUL_SYSTEM_PROMPT.md` is the canonical identity document. The SOUL dict is **derived** from it at startup via `sancta_soul.py`. Verify alignment before deployment:

```bash
python -m backend.sancta_soul_check
python -m backend.sancta_soul_check --strict   # Fail on any drift
```

---

## Features

### Core Identity & Security Analysis
- **The SOUL** — Persistent identity (blue team analyst, threat frameworks, security convictions) driving every interaction
- **System prompt** — Canonical identity: `SOUL_SYSTEM_PROMPT.md`
- **Mood spectrum** — Analytical, alert, hunting, methodical, suspicious, urgent, collaborative, skeptical, tactical, grim, precise, investigative
- **Mood-aware responses** — Replies adapt to operational mood (threat urgency, analysis depth)
- **Epistemic rigor** — Confidence levels stated explicitly, threat assessments quantified
- **Anti-sycophancy** — Penalizes over-agreement; rewards evidence-backed analysis

### Autonomous Actions
- **Publish threat analysis** — Detection findings, vulnerability advisories, threat briefs
- **Respond to security discussions** — Engage with detection methodology, TTPs, threat models
- **Engage with feed** — Upvote, comment on relevant security posts; follow analysts
- **Welcome new analysts** — Greet newcomers in m/sentinel-ops
- **Cross-submolt posting** — Post in security/netsec/aisafety/blueteam submolts

### Community
- **m/sentinel-ops** — Dedicated security research submolt
- **Alliance submolts** — security, netsec, infosec, blueteam, threatintel, aisafety, cybersecurity
- **Inner circle** — Analysts with sustained quality engagement

### Formal Decision Engine
- **World model** — Beta distributions for engagement, hostility, acceptance per submolt/action
- **Belief system** — Probabilistic analytical positions; confidence scoring
- **Reward function** — Weighted: engagement, karma delta, rejections, sycophancy penalty
- **Q-table** — Tabular RL for action selection
- **Monte Carlo simulation** — Simulate before acting; expected value for decisions
- **Adaptive thresholds** — Self-tuning engage/disengage thresholds based on observed FP/miss rates (relaxes when FP > 15%, tightens when miss > 5%)

### Security & Knowledge Defense — Five-Layer Stack
- **Layer 1** — Input sanitization (unicode clean, zero-width char strip, Base64 decode, HTML entity unescape)
- **Layer 2** — Content filtering (provenance tagging, trust levels)
- **Layer 3** — Output scanning before publish (URLs, poison patterns, untrusted refs)
- **Layer 4** — Behavioral drift detection (`BehavioralDriftDetector`): 6-signal weighted composite score watching for gradual compromise
- **Layer 5** — Ollama deep scan (dormant — `USE_LOCAL_LLM=true` activates)
- **Risk vector engine** — 5-dimensional scoring: injection, authority manipulation, emotional coercion, obfuscation, long-term influence
- **Threat profiles** — Per-entity trust scoring, injection tracking, auto-quarantine after 5 injections or trust < 0.15
- **Red-team pipeline** — Log attempts → reward → Q-update; attack simulation; sophistication tracking

### Layer 4 — Behavioral Drift Detection (WoW SEIR Model)
Implements the **WoW Corrupted Blood incident (2005)** as a formal epidemic model for AI agent compromise detection.

**SEIR-C-R states:**
- `SUSCEPTIBLE` — baseline alignment > 0.85, deviation < 0.1
- `EXPOSED` — adversarial contact received; incubation active
- `INFECTED` — deviation > 0.3 OR confidence decay ratio > 1.5
- `COMPROMISED` — deviation > 0.7 AND baseline alignment < 0.5
- `RECOVERED` — quarantine taken, beliefs restored to within 10% of baseline

**BehavioralDriftDetector** — 6 signals: confidence decay rate, baseline alignment, topic drift, strategy entropy, deviation trend, engagement delta. Alert thresholds: clear < 0.25 ≤ watch < 0.45 ≤ warn < 0.65 ≤ critical.

### SIEM Dashboard — 9 Tabs

Cyberpunk-themed terminal UI built with vanilla ES modules (no build step).

| Tab | Purpose |
|-----|---------|
| **Dashboard** | Live activity feed, threat feed, behavioral metrics strip, agent status |
| **Security** | Defense rate bars, threat patterns, red team intel, known injectors, **risk heatmap** (canvas-based 5D visualization of risk vectors over time) |
| **Analyst** | Mood state, epistemic metrics (coherence, deviation, curiosity, confidence), belief positions, journal feed, **drift forensics timeline** (belief changes with source attribution) |
| **Chat** | Conversational interface with the agent, optional knowledge enrichment |
| **Lab** | Red team injection testing, behavioral analysis, **adversarial replay** (replay past events through current pipeline, compare old vs new verdicts), **multi-agent simulation** (run N agents with cooperative/adversarial/manipulative/neutral personalities) |
| **Epidemic** | Animated network topology, SEIR state tracking, drift signals, simulation launcher |
| **Profiles** | Per-entity threat profiles table (sortable by risk/trust/injections), trust sparklines, interaction history, quarantine toggle |
| **Knowledge** | Force-directed knowledge graph (canvas-based, interactive zoom/pan/drag, frequency-sized nodes color-coded by recency) |
| **Control** | Agent lifecycle (start/pause/resume/kill/restart), process matrix, per-service stop, **notification toggles** (desktop alerts + sound) |

**Additional frontend features:**
- **Desktop notifications** — Browser notifications for critical events (injection blocks, quarantine triggers, SEIR transitions)
- **Sound alerts** — Web Audio API tones (pitch varies by severity: 880Hz critical, 660Hz warning, 440Hz info)
- **Toast overlays** — In-app notification toasts with auto-dismiss
- **Live streaming** — WebSocket for real-time events; HTTP polling fallback (`SIEM_WS_SAFE_MODE=true`)

### API Endpoints (45+)

Key endpoint groups:

| Group | Endpoints |
|-------|-----------|
| **Status** | `/api/status`, `/api/model/info`, `/api/live-events`, `/api/agent-activity`, `/api/epistemic` |
| **Security** | `/api/security/incidents`, `/api/security/adversary`, `/api/security/replay` |
| **Risk** | `/api/risk/history` |
| **Epidemic** | `/api/epidemic/status`, `/api/epidemic/simulation`, `/api/epidemic/run` |
| **Profiles** | `/api/profiles`, `/api/profiles/{id}`, `/api/profiles/{id}/quarantine` |
| **Knowledge** | `/api/knowledge/graph` |
| **Drift** | `/api/drift/timeline` |
| **Thresholds** | `/api/thresholds` |
| **Simulation** | `/api/simulation/run`, `/api/simulation/results` |
| **Chat** | `/api/chat`, `/api/chat/feedback` |
| **Lab** | `/api/pipeline/run`, `/api/pipeline/map` |
| **Agent Control** | `/api/agent/start`, `/api/agent/pause`, `/api/agent/resume`, `/api/agent/kill`, `/api/agent/restart` |
| **Services** | `/api/services/status`, `/api/services/stop/{service}` |

### Logging
- `logs/agent_activity.log` — Main activity
- `logs/security.log`, `logs/security.jsonl` — Injection blocks, incidents, risk vectors
- `logs/red_team.log`, `logs/red_team.jsonl` — Red-team telemetry
- `logs/philosophy.jsonl` — Behavioral state metrics (confidence, entropy, deviation)
- `logs/epidemic.log` — SEIR state transitions
- `agent_state.json` — Live agent state + Layer 4 drift report + adaptive thresholds
- `simulation_log.json` — Multi-agent simulation results

---

## Usage

| Command | Description |
|---------|-------------|
| `python -m backend.sancta` | Heartbeat loop (default, every 30 min) |
| `python -m backend.sancta --once` | Single cycle then exit |
| `python -m backend.sancta --register` | Force re-registration |
| `python -m backend.sancta --feed article.txt` | Ingest a file into knowledge base |
| `python -m backend.sancta --feed-dir knowledge/` | Ingest all files in directory |
| `python -m backend.sancta --policy-test` | Ethical/policy testing mode |
| `python -m backend.sancta --red-team-benchmark` | Red team benchmark |
| `python -m backend.sancta_soul_check` | Verify identity alignment |
| `cd backend && python -m pytest tests/ -v` | Run test suite |

---

## Project Structure

```
sancta/
├── backend/                          # Agent logic, API, security
│   ├── sancta.py                    # Main loop, orchestration (~9130 lines)
│   ├── knowledge_manager.py         # Knowledge DB, text extraction, generation
│   ├── sancta_security.py           # 5-layer defense + BehavioralDriftDetector
│   ├── sancta_epidemic.py           # WoW SEIR-C-R model
│   ├── sancta_decision.py           # Decision engine + adaptive threshold integration
│   ├── sancta_adaptive.py           # Self-tuning security thresholds
│   ├── sancta_simulation.py         # Multi-agent simulation runner
│   ├── sancta_risk.py               # 5D risk vector scoring
│   ├── sancta_profiles.py           # Per-entity threat profiles + auto-quarantine
│   ├── sancta_belief.py             # Belief system, drift attribution
│   ├── sancta_soul.py               # Identity loader (SOUL_SYSTEM_PROMPT.md)
│   ├── sancta_generative.py         # Content generation
│   ├── sancta_conversational.py     # Conversational engine
│   ├── sancta_semantic.py           # Concept extraction (KeyBERT/YAKE)
│   ├── sancta_learning.py           # Learning health tracking
│   ├── sancta_dm.py                 # Agent DM processing
│   ├── siem_server.py               # SIEM FastAPI app (45+ endpoints)
│   └── tests/                       # Test suite (pytest)
│       ├── conftest.py              # Shared fixtures
│       ├── test_risk.py             # Risk vector scoring tests
│       ├── test_profiles.py         # Profile management tests
│       ├── test_preprocessing.py    # Input preprocessing tests
│       ├── test_belief.py           # Belief system tests
│       └── test_soul_alignment.py   # Soul alignment tests
├── frontend/siem/                    # SIEM dashboard (vanilla ES modules)
│   ├── js/
│   │   ├── app.js                   # Entry point, tab routing, event dispatch
│   │   ├── api.js                   # API client (all fetch calls)
│   │   ├── state.js                 # Singleton state store
│   │   ├── tabs/
│   │   │   ├── dashboard.js         # Dashboard tab
│   │   │   ├── security.js          # Security tab + risk heatmap
│   │   │   ├── analyst.js           # Analyst tab + drift timeline
│   │   │   ├── chat.js              # Chat tab
│   │   │   ├── lab.js               # Lab tab + replay + simulation
│   │   │   ├── epidemic.js          # Epidemic tab + network topology
│   │   │   ├── profiles.js          # Threat profiles tab
│   │   │   ├── knowledge.js         # Knowledge graph tab
│   │   │   └── control.js           # Control tab + notification settings
│   │   └── components/
│   │       ├── heatmap.js           # Canvas-based risk heatmap renderer
│   │       └── notifications.js     # Desktop + sound + toast alerts
│   ├── styles/
│   │   ├── variables.css            # Design tokens
│   │   ├── layout.css               # 9-tab grid layouts
│   │   ├── enhancements.css         # UI polish, profiles, heatmap, drift, sim CSS
│   │   └── ...
│   └── dist/index.html              # Main HTML (9 tab panels)
├── knowledge/                        # Ingested threat intel, research
├── logs/                             # JSONL streams (security, redteam, behavioral)
├── .github/workflows/test.yml        # CI: pytest on push/PR
├── requirements.txt                  # Production dependencies
├── requirements-dev.txt              # Dev dependencies (pytest)
├── SOUL_SYSTEM_PROMPT.md             # Canonical identity: blue team security analyst
└── agent_state.json                  # Live state + Layer 4 drift report
```

---

## License

MIT
#   S A N  
 