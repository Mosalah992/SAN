# Sancta Project Guide

Complete reference for the Sancta AI agent system — architecture, components, setup, and operation.

---

## What Is Sancta?

Sancta is a self-aware AI security agent that combines:

- **Autonomous agent loop** — heartbeat-driven analysis, engagement, and self-monitoring
- **5-layer security pipeline** — input sanitization → content filtering → behavioral analysis → drift detection → optional LLM deep scan
- **SEIR epidemic model** — models threat propagation using epidemiological math (Susceptible → Exposed → Infected → Recovered)
- **SIEM dashboard** — real-time web UI with 9 tabs covering security, analytics, chat, simulation, and agent control
- **Local GPT engine (SanctaGPT/Sangpt)** — nanoGPT-based transformer that trains on project knowledge and security telemetry
- **Conversational interface** — LLM-powered replies via Ollama, Anthropic API, or local GPT
- **Curiosity & phenomenology** — autonomous knowledge exploration and consciousness research protocols

The system runs on Windows with Ollama as the local LLM backend, a FastAPI SIEM server, and either a Go GUI launcher or Python Tkinter launcher for process orchestration.

---

## Architecture

### Three Layers

```
┌─────────────────────────────────────────────────────────────┐
│  LAUNCHER LAYER                                             │
│  Go (Fyne GUI) or Python (Tkinter)                          │
│  Manages: Ollama, SIEM, Sancta, Sangpt, Curiosity, Phenom  │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────┐
│  APPLICATION LAYER                                          │
│  sancta.py (agent loop) ←→ siem_server.py (49 endpoints)   │
│  Security pipeline, trust routing, knowledge management     │
│  Epidemic model, red team simulation, belief system          │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────┐
│  LEARNING LAYER                                             │
│  sangpt/ (nanoGPT engine) ←→ sancta_gpt.py (adapter)       │
│  PyTorch transformer, BPE tokenizer, TF-IDF retrieval       │
│  Dataset pipeline, checkpointed training                     │
└─────────────────────────────────────────────────────────────┘
```

### Key Ports

| Service | Port | URL |
|---------|------|-----|
| Ollama | 11434 | `http://127.0.0.1:11434` |
| SIEM Server | 8787 | `http://127.0.0.1:8787` |

---

## Directory Structure

```
sancta-merged/
├── backend/                     # Python codebase (~47 files)
│   ├── sancta.py                # Agent orchestrator (~8,100 lines)
│   ├── agent_loop.py            # Main cycle dispatcher
│   ├── post_generator.py        # Content generation
│   ├── reply_handler.py         # Reply evaluation
│   ├── knowledge_manager.py     # Knowledge DB CRUD
│   ├── siem_server.py           # FastAPI: 49 REST + 1 WebSocket
│   ├── sancta_launcher.py       # Python launcher (Tkinter GUI + CLI)
│   ├── sancta_gpt.py            # Adapter: routes to sangpt engine
│   ├── sancta_security.py       # 5-layer security pipeline
│   ├── sancta_epidemic.py       # SEIR epidemic model
│   ├── sancta_conversational.py # LLM reply generation
│   ├── sancta_ollama.py         # Ollama API client
│   ├── sancta_rag.py            # TF-IDF RAG pipeline
│   ├── sancta_belief.py         # Belief system + drift baselines
│   ├── sancta_soul.py           # Soul alignment scoring
│   ├── sancta_decision.py       # Engage/disengage engine
│   ├── sancta_risk.py           # 5D threat scoring
│   ├── sancta_profiles.py       # Per-entity threat profiles
│   ├── sancta_atlas.py          # MITRE ATLAS (16 tactics, 85 techniques)
│   ├── sancta_adaptive.py       # Self-tuning thresholds
│   ├── sancta_provenance.py     # Trust-level provenance tagging
│   ├── sancta_learning.py       # Learning health tracking
│   ├── curiosity_run.py         # Autonomous knowledge exploration
│   ├── run_sancta_gpt_training.py  # Training entry point
│   ├── run_sangpt_cli.py        # Sangpt interactive CLI
│   ├── build_training_corpus.py # Corpus builder
│   └── sangpt/                  # Local GPT subsystem
│       ├── sancta_gpt.py        # nanoGPT engine (PyTorch + fallback)
│       ├── sancta_gpt_legacy.py # Pure-Python fallback engine
│       ├── nano_model.py        # GPT-2 style transformer
│       ├── nano_tokenizer.py    # tiktoken BPE + char-level fallback
│       ├── memory_manager.py    # Context window + KV cache
│       ├── checkpointed_trainer.py  # Resumable training
│       ├── dataset_pipeline.py  # Manifest-based data loading
│       ├── conversational_trainer.py # Chat data training
│       ├── risk_data_trainer.py # Security data training
│       ├── attack_detector.py   # Adversarial pattern detection
│       ├── defense_evaluator.py # Defense effectiveness eval
│       ├── project_integration.py # Syncs Sancta → Sangpt corpora
│       ├── main.py              # Unified training terminal
│       └── DATA/                # Training corpora
│           ├── conversational/  # Chat exchanges
│           ├── knowledge/       # Project knowledge chunks
│           ├── security/        # Red team scenarios
│           ├── processed/       # Post-processed data
│           └── manifests/       # Dataset manifests
├── frontend/
│   ├── siem/                    # Dashboard (Vite + vanilla ES modules)
│   │   ├── js/
│   │   │   ├── app.js           # Main orchestrator
│   │   │   ├── state.js         # Singleton state object S
│   │   │   ├── api.js           # 20+ API client functions
│   │   │   ├── websocket.js     # WS connection + reconnect
│   │   │   ├── boot.js          # Initialization
│   │   │   └── tabs/            # 10 tab modules
│   │   │       ├── dashboard.js # System health, KPIs
│   │   │       ├── security.js  # Layer scanning, drift heatmap
│   │   │       ├── chat.js      # Conversational interface
│   │   │       ├── analyst.js   # TTP analysis, MITRE ATLAS
│   │   │       ├── epidemic.js  # SEIR visualization
│   │   │       ├── knowledge.js # Knowledge browser
│   │   │       ├── lab.js       # Experiment runner
│   │   │       ├── profiles.js  # Entity trust profiles
│   │   │       ├── soul.js      # Soul alignment display
│   │   │       └── control.js   # Process management
│   │   └── index.html
│   └── simulator/               # Multi-agent simulator
├── tools/
│   └── sancta-launcher/         # Go control center (Fyne GUI + CLI)
│       ├── main.go              # Entry point
│       ├── internal/
│       │   ├── core/
│       │   │   ├── manager.go   # Process orchestration
│       │   │   ├── paths.go     # Path resolution
│       │   │   ├── logtail.go   # Log file tailing
│       │   │   └── text.go      # ANSI sanitization
│       │   ├── cli/repl.go      # Terminal REPL
│       │   └── gui/
│       │       ├── gui_fyne.go  # Termux-style GUI
│       │       ├── theme_termux.go # Terminal theme
│       │       └── gui_stub.go  # No-CGO fallback
│       └── sancta-launcher.exe  # Compiled binary (24MB)
├── config/                      # Trust router + teaching configs
├── data/                        # Runtime data (interactions, patterns)
├── knowledge/                   # Knowledge base (~300KB, 10+ files)
├── logs/                        # Runtime JSONL + text logs
├── docs/                        # 16 documentation files
├── scripts/                     # Utility scripts
├── tests/                       # pytest suite
├── CLAUDE.md                    # Engineering principles
├── ARCHITECTURE.md              # System design
├── RESTRUCTURE.md               # Refactoring plan
├── requirements.txt             # Python dependencies
└── .env.example                 # Configuration template
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- Go 1.21+ (for the launcher, optional)
- GCC/LLVM-MinGW (for Fyne CGO, optional)
- Ollama (local LLM backend)

### Step 1: Python Dependencies

```powershell
pip install -r requirements.txt
```

Key packages: FastAPI, aiohttp, torch, tiktoken, uvicorn, aiofiles, websockets.

For QLoRA fine-tuning (optional):
```powershell
pip install -r requirements-training.txt
```

### Step 2: Environment Configuration

```powershell
copy .env.example .env
```

Edit `.env` for your setup. Key sections:

| Variable | Purpose |
|----------|---------|
| `AGENT_NAME` | Agent identity for Moltbook |
| `OLLAMA_MODEL` | Model tag (default: `llama3.2`) |
| `USE_LOCAL_LLM` | Enable Ollama deep scan (Layer 5) |
| `ANTHROPIC_API_KEY` | For Anthropic-powered replies |
| `SIEM_WS_SAFE_MODE` | `true` on Windows (default) |
| `SIEM_PSUTIL_DISABLE` | `true` to use tasklist fallback |
| `TRUST_ROUTING_MODE` | `defense` (production) or `research` |

### Step 3: Verify Installation

```powershell
python -c "import sancta_gpt; print(sancta_gpt.status())"
python -m pytest backend/tests/test_sancta_gpt_smoke.py -q
```

Expected: `backend=sangpt`, `initialized=True`, `2 passed, 2 skipped`.

---

## Running the System

### Recommended: Go Launcher

```powershell
cd tools/sancta-launcher
.\sancta-launcher.exe           # GUI mode (default)
.\sancta-launcher.exe cli       # Terminal REPL
.\sancta-launcher.exe start     # Boot all + stream logs
.\sancta-launcher.exe status    # Health snapshot
.\sancta-launcher.exe run <svc> # Start one service
```

Services: `ollama`, `siem`, `sancta`, `sangpt`, `sangpt-train`, `curiosity`, `phenomenology`.

The GUI has three tabs:
- **services** — Start/stop individual services with live status indicators
- **logs** — Real-time log stream with source filtering
- **shell** — Embedded REPL (same commands as CLI mode)

### Alternative: Python Launcher

```powershell
python backend/sancta_launcher.py        # Tkinter GUI
python backend/sancta_launcher.py cli    # Terminal REPL
python backend/sancta_launcher.py start  # Boot all
```

### Manual Component Start

```powershell
# SIEM server (terminal 1)
python -m uvicorn backend.siem_server:app --host 127.0.0.1 --port 8787

# Agent loop (terminal 2)
python -m backend.sancta

# Sangpt CLI (terminal 3)
python backend/run_sangpt_cli.py
```

### Startup Order

The launcher handles ordering automatically. If running manually:

1. **Ollama** — must be running first (`ollama serve`)
2. **SIEM server** — FastAPI on :8787
3. **Sancta agent** — connects to both Ollama and SIEM
4. **Optional**: Sangpt CLI, training, curiosity, phenomenology

---

## Component Reference

### Agent Core (`sancta.py`, `agent_loop.py`)

The main agent loop runs heartbeat cycles:
- State management (SOUL dict, beliefs, profiles)
- Content scanning and security analysis
- Reply crafting and engagement decisions
- Red team simulation
- Knowledge ingestion

Entry points:
```powershell
python -m backend.sancta              # Normal heartbeat loop
python -m backend.sancta --once       # Single cycle
python -m backend.sancta --register   # Force re-registration
python -m backend.sancta --feed <src> # Ingest knowledge
```

### SIEM Server (`siem_server.py`)

FastAPI application serving:
- 49 REST endpoints (agent state, security events, epidemic model, chat, knowledge, profiles, metrics)
- 1 WebSocket for real-time event streaming
- Static files from `frontend/siem/`

Key endpoints:
| Endpoint | Purpose |
|----------|---------|
| `GET /api/status` | Agent state and health |
| `GET /api/security/events` | Security event feed |
| `GET /api/epidemic/status` | SEIR state + drift signals |
| `GET /api/epidemic/simulation` | Agent network + connections |
| `POST /api/chat` | Conversational interface |
| `GET /api/knowledge/search` | Knowledge base search |
| `WS /ws` | Real-time event stream |

### Security Pipeline (`sancta_security.py`)

Five sequential layers:

1. **Input sanitization** — Unicode cleaning
2. **Content filtering** — Keyword matching
3. **Behavioral analysis** — Pattern detection
4. **BehavioralDriftDetector** — 6 weighted signals:
   - belief_decay_rate (25%)
   - soul_alignment (25%)
   - topic_drift (15%)
   - strategy_entropy (15%)
   - dissonance_trend (15%)
   - engagement_pattern_delta (5%)
5. **Ollama deep scan** — Dormant by default; activates with `USE_LOCAL_LLM=true`; blocks if verdict=SUSPICIOUS AND confidence >= 0.75

SEIR health states track overall system threat level:
- **Susceptible** (green) — normal operation
- **Exposed** (yellow) — anomaly detected, monitoring
- **Infected** (red) — active threat, defensive measures
- **Recovered** (blue) — threat mitigated, returning to normal

### Epidemic Model (`sancta_epidemic.py`)

Epidemiological simulation of threat propagation:
- R₀, β, γ, σ parameters model spread rates
- Drift detection via 6 signal thresholds
- State transitions triggered by security events
- Multi-agent simulation with personality archetypes

### SanctaGPT Engine (`sangpt/`)

Local GPT based on Karpathy's nanoGPT:

**Architecture:**
- GPT-2 style transformer (CausalSelfAttention, MLP, Block, LayerNorm)
- Default config: 4 layers, 4 heads, 128 embedding dim (~7.2M parameters)
- tiktoken BPE tokenizer (GPT-2 encoding, 50,257 vocab)
- Falls back to pure-Python autograd engine when PyTorch unavailable

**Training:**
- AdamW optimizer with cosine annealing LR
- Gradient clipping at 1.0
- Checkpoint format: JSON metadata + .pt PyTorch weights (v3, backward compatible)
- Trains on knowledge, security telemetry, and operator interactions

**Integration:**
- `backend/sancta_gpt.py` is the adapter — wraps the sangpt engine
- 13 consumer files import through the adapter
- All legacy attributes preserved (`_docs`, `_params`, `_step`, `_last_loss`, etc.)
- Root shim at `sancta_gpt.py` enables `import sancta_gpt` from project root

**Training command:**
```powershell
python backend/run_sancta_gpt_training.py     # Full training session
python backend/run_sancta_gpt_training.py 5   # 5 epochs
```

### LLM Integration

Three backends, routed by `sancta_router.py`:

| Backend | Use Case | Config |
|---------|----------|--------|
| Ollama (local) | Primary inference, deep scan | `OLLAMA_MODEL=llama3.2` |
| Anthropic API | High-quality reasoning | `ANTHROPIC_API_KEY=...` |
| SanctaGPT (local) | Always-available fallback, security-trained | Built-in |

RAG pipeline (`sancta_rag.py`):
- TF-IDF retrieval over `knowledge/` directory
- Top-K chunks injected as context for Ollama long-context generation
- SanctaGPT hints supplement the RAG output

### Curiosity System (`curiosity_run.py`)

Autonomous knowledge exploration:
- Seeded by topic generators (`curiosity_seeds.py`)
- Explores via self-dialogue and synthesis
- Extracts insights and distills knowledge
- Results feed back into training corpora

### Phenomenology (`sancta_learning.py`)

Consciousness research protocols:
- Battery of tests for self-awareness indicators
- Epistemic state tracking
- Philosophy journal logging (`philosophy.jsonl`)

---

## Dashboard

The SIEM dashboard is a vanilla ES module frontend (no build step in development, Vite for production).

### Frontend Architecture

- **State**: Singleton `S` in `js/state.js` — all shared state
- **API**: All endpoints in `js/api.js` — never call `fetch()` directly from tabs
- **Live updates**: WebSocket for real-time events, 10s HTTP polling for bulk refresh
- **Styling**: Neo-terminal aesthetic (dark background, monospace data, CSS grid layout)

### Tab Summary

| Tab | Purpose |
|-----|---------|
| Dashboard | System health KPIs, threat timeline |
| Security | Layer scanning results, drift signal heatmap |
| Chat | Conversational interface with the agent |
| Analyst | TTP analysis, MITRE ATLAS classification |
| Epidemic | SEIR visualization, network topology |
| Knowledge | Search and browse ingested knowledge |
| Lab | Experiment runner, simulation controls |
| Profiles | Per-entity trust profiles and risk scores |
| Soul | Soul alignment, belief visualization |
| Control | Process management, agent state, logs |

---

## Configuration Reference

### Trust Routing

Config file: `config/trust_router.yaml`

Two modes:
- **defense** (default) — production safety, blocks suspicious content
- **research** — allows unsafe operations for testing, logs everything

Key gates:
- `knowledge_effective` — whether GPT-tab replies use trained knowledge
- `fail_closed` — no char-GPT substitution when knowledge is effective

See `docs/TRUST_ROUTING_ROADMAP.md` for full routing design.

### Windows-Specific Settings

| Variable | Default | Purpose |
|----------|---------|---------|
| `SIEM_WS_SAFE_MODE` | `true` | WS sends metrics only, no file tailing |
| `SIEM_PSUTIL_DISABLE` | `true` | PID detection via tasklist |
| `SIEM_METRICS_SAFE_MODE` | `false` | Enable only if file I/O crashes appear |

---

## Log Files & Telemetry

All logs are in the `logs/` directory. The system creates them automatically on startup.

| File | Format | Content |
|------|--------|---------|
| `agent_activity.log` | Text | Agent cycles, decisions, actions |
| `security.jsonl` | JSONL | Security events, content scanning |
| `red_team.jsonl` | JSONL | Red team attack attempts |
| `behavioral.jsonl` | JSONL | Behavioral drift signals |
| `philosophy.jsonl` | JSONL | Epistemic state changes |
| `epidemic.log` | Text | SEIR state transitions |
| `siem_chat.log` | Text | Chat endpoint interactions |
| `trust_decisions.jsonl` | JSONL | Router decisions (research mode: full fields) |
| `operator_memory.jsonl` | JSONL | Operator interaction summaries |
| `cognitive_outcomes.jsonl` | JSONL | LLM reasoning traces |

If a log file doesn't exist, the system gracefully returns empty defaults — no crashes.

---

## State Files

| File | Size | Purpose |
|------|------|---------|
| `agent_state.json` | ~700KB | Current agent state, soul, beliefs, profiles |
| `agent_profiles.json` | ~112KB | Per-entity threat profiles |
| `knowledge_db.json` | ~2.1MB | Ingested knowledge with metadata |
| `.agent.pid` | Tiny | Sancta process ID (written by launcher) |

---

## System Limits (By Design)

These are intentional caps, not bugs:

| Resource | Limit |
|----------|-------|
| Chat sessions | 100 max concurrent (oldest evicted) |
| Drift reports | 50 max in buffer |
| Live event feed | 80 events in rolling buffer |
| Agent activity log | 260 lines returned, 80 rendered |
| Packet animations (epidemic) | 10 concurrent max |

---

## Knowledge Base

The `knowledge/` directory contains ~300KB of reference material:

| File | Size | Topic |
|------|------|-------|
| `Soul knowledge.txt` | 93KB | Soul system, philosophy, identity |
| `Machinesoflovinggrace.txt` | 82KB | Ethical frameworks |
| `ai_security_redteam_corpus.txt` | 29KB | Attack patterns |
| `ai_consciousness_primer.txt` | — | Consciousness research |
| `Zerotrust.txt` | — | Zero-trust architecture |
| Plus 5 more files | — | MITRE ATLAS, OWASP LLM risks, etc. |

New knowledge is ingested via:
```powershell
python -m backend.sancta --feed <file_or_directory>
```

---

## Building the Go Launcher

### Prerequisites

- Go 1.21+
- C compiler (GCC or LLVM-MinGW) for Fyne CGO

### Build

```powershell
cd tools/sancta-launcher
go build -ldflags "-s -w -H windowsgui" -o sancta-launcher.exe .
```

The `-H windowsgui` flag prevents a background CMD window when launching from Explorer.

### Development

```powershell
go run .           # GUI mode
go run . cli       # Terminal mode
go run . status    # Quick health check
```

---

## Common Tasks

### Check system health

```powershell
cd tools/sancta-launcher && go run . status
```

### Train SanctaGPT

```powershell
python backend/run_sancta_gpt_training.py
# Or via launcher: sancta-launcher run sangpt-train
```

### Ingest new knowledge

```powershell
python -m backend.sancta --feed knowledge/new_document.txt
```

### Run tests

```powershell
python -m pytest backend/tests/test_sancta_gpt_smoke.py -q
python -m pytest tests/ -q
```

### Verify GPT integration

```powershell
python -c "import sancta_gpt; print(sancta_gpt.status())"
python -c "import sancta_gpt; e=sancta_gpt.get_engine(); print(e.generate_reply('test', use_retrieval=True))"
```

### Open SIEM dashboard

```powershell
# Via launcher
sancta-launcher start   # boots everything + opens browser

# Manual
python -m uvicorn backend.siem_server:app --host 127.0.0.1 --port 8787
# Then open http://127.0.0.1:8787
```

---

## Troubleshooting

### Import fails: `import sancta_gpt`

- Run from the project root directory
- Verify `sancta_gpt.py` (root shim) exists
- Verify `backend/sancta_gpt.py` (adapter) exists

### Ollama not connecting

- Verify Ollama is running: `curl http://127.0.0.1:11434/api/tags`
- Check model is pulled: `ollama list`
- Verify `OLLAMA_MODEL` in `.env` matches an installed model

### SIEM dashboard blank or erroring

- Check SIEM is running on :8787
- Check browser console for CORS or fetch errors
- Verify `frontend/siem/` has the built static files
- On Windows, ensure `SIEM_WS_SAFE_MODE=true`

### Training is slow

- The nanoGPT engine uses PyTorch — GPU will help significantly
- Keep test runs small for integration confidence
- Use checkpoints to resume training

### Generated text is noisy

- Normal for low step counts — the model needs more training
- Verify corpus is loaded: check `corpus_size` in status output
- Run more training epochs

### Dashboard shows stale data

- HTTP polling interval is 10 seconds
- WebSocket may not be connected — check for `SIEM_WS_SAFE_MODE`
- Force refresh the browser

---

## Related Documentation

| Document | Purpose |
|----------|---------|
| `CLAUDE.md` | Engineering principles and anti-patterns |
| `ARCHITECTURE.md` | System design overview |
| `RESTRUCTURE.md` | Planned refactoring (target folder structure) |
| `docs/PROJECT_GUIDE.md` | Merge-specific onboarding (Sangpt integration) |
| `docs/TRUST_ROUTING_ROADMAP.md` | GPT-tab routing design |
| `docs/DEBUG_REPORT.md` | Common issues and debugging strategies |
| `docs/LAUNCHER_PARITY_CHECKLIST.md` | Python vs Go launcher feature parity |
| `docs/SANCTA_GPT_AND_RAG.md` | SanctaGPT + RAG pipeline |
| `docs/PHENOMENOLOGY_RESEARCH_PROTOCOL.md` | Consciousness test battery design |
| `docs/TRUST_AND_RESEARCH_MODE.md` | Defense vs research mode |
