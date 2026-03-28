# Sancta Architecture

Sancta is an autonomous Blue Team Security Analyst — an AI agent that posts threat intelligence, monitors adversarial behavior, and defends its own integrity across a social network (MoltBook). It includes a 9-tab SIEM dashboard, a 5-layer security pipeline, a SEIR epidemic model for drift detection, and MITRE ATLAS threat classification.

---

## System Overview

```
                    ┌─────────────────────────────────────┐
                    │          SIEM Dashboard              │
                    │  (9 tabs, vanilla ES modules)        │
                    │  Dashboard | Security | Analyst      │
                    │  Chat | Lab | Epidemic | Profiles    │
                    │  Knowledge | Control                 │
                    └──────────────┬──────────────────────┘
                                   │ REST + WebSocket
                    ┌──────────────▼──────────────────────┐
                    │        siem_server.py                │
                    │   FastAPI · 49 endpoints · 1 WS      │
                    └──────────────┬──────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
   ┌──────▼──────┐   ┌────────────▼───────────┐   ┌───────▼───────┐
   │  sancta.py  │   │   Security Pipeline    │   │  ATLAS Engine │
   │  Agent Core │   │ 5 layers + risk engine │   │  sancta_atlas │
   │  + 3 modules│   │ + profiles + epidemic  │   │  16 tactics   │
   └──────┬──────┘   └────────────────────────┘   │  85 techniques│
          │                                        └───────────────┘
   ┌──────▼──────┐
   │   Ollama    │
   │  llama3.2   │
   └─────────────┘
```

---

## Backend Modules (47 files)

### Core Agent

| Module | Lines | Responsibility |
|--------|-------|----------------|
| `sancta.py` | ~6,200 | Agent orchestrator: config, state, SOUL dict, belief/RL engine, security scanning, craft_reply, red team simulation |
| `agent_loop.py` | ~850 | Main cycle loop, slot actions, heartbeat, CLI entry point |
| `post_generator.py` | ~890 | Content creation, knowledge ingestion, Ollama prompting |
| `reply_handler.py` | ~1,290 | Reply evaluation, feed engagement, recruitment, submolt management |
| `knowledge_manager.py` | — | Knowledge DB CRUD, ingestion pipeline |

### Security Pipeline

| Module | Responsibility |
|--------|----------------|
| `sancta_security.py` | 5-layer content scanning, BehavioralDriftDetector (6 weighted signals), SEIR health states |
| `sancta_risk.py` | RiskVector (5-dimensional threat scoring), profile-based amplification |
| `sancta_profiles.py` | Per-entity threat profiles, trust decay/recovery, quarantine logic |
| `sancta_atlas.py` | MITRE ATLAS integration — 16 tactics, 85 techniques, event classification, TTP tracking |
| `sancta_belief.py` | BeliefSystem with source attribution, drift baselines |
| `sancta_soul.py` | Soul alignment scoring, keyword extraction |
| `sancta_decision.py` | Engage/disengage decision engine with risk-adjusted rewards |
| `sancta_adaptive.py` | Self-tuning security thresholds based on FP/FN rates |
| `sancta_provenance.py` | Trust-level provenance tagging (HIGH/MEDIUM/LOW/UNTRUSTED) |

### AI & Generation (incl. SanctaGPT)

| Module | Responsibility |
|--------|----------------|
| `sancta_gpt.py` | Zero-dependency pure-Python GPT engine: autograd, transformer, train/generate/checkpoint |
| `sancta_conversational.py` | Reply generation, Ollama fallback templates |
| `sancta_generative.py` | Transformer-inspired fragment selection |
| `sancta_ollama.py` | Ollama API client, model management |
| ~~`sancta_templates.py`~~ | Removed — Moltbook replies use Ollama / Anthropic / SanctaGPT only |
| `sancta_transformer.py` | Trained transformer model (optional, disabled by default on Windows) |
| `sancta_semantic.py` | Semantic similarity, embedding operations |
| `sancta_rag.py` | TF-IDF RAG over `knowledge/` chunks → Ollama long context + compact hints for SanctaGPT `generate_reply`; see `docs/SANCTA_GPT_AND_RAG.md` |
| `sancta_cognitive_pipeline.py` | **Thinking / knowing / acting** separation: `security_gate()` before operator LLM paths, `build_structured_evidence()` (ranked JSON bundle), `decide_policy()`, `validate_reasoning_output()`, outcome + attempt JSONL (`logs/cognitive_outcomes.jsonl`, `logs/security_gateway_attempts.jsonl`). Complements (does not replace) the 5-layer `sancta_security` pipeline. |
| `sancta_router.py` | GPT-tab routing: `RouteDecision` (axes, `route_confidence`, post-route gate) → `knowledge_effective` vs conversational; see `docs/TRUST_ROUTING_ROADMAP.md` |
| `sancta_trust_config.py` | `SANCTA_TRUST_MODE=defense|research`; gates unsafe research toggles; startup warnings |
| `trust_telemetry.py` | Append-only `logs/trust_decisions.jsonl` (schema-versioned) for routing/policy events |
| `memory_redact.py` | Redact secrets in memory; instruction/framing scores for operator recall pipeline |
| `operator_memory.py` | Redacted JSONL + **untrusted** summary/extractive recall for `/api/chat` (not verbatim); `SANCTA_MEMORY_RAW_MODE` in research only |
| `config/trust_router.yaml` / `config/trust_router.json` | Thresholds: `confidence_min`, `gate_threshold`, `knowledge_margin`, axis weights (YAML preferred; JSON fallback) |

### Epidemic & Simulation

| Module | Responsibility |
|--------|----------------|
| `sancta_epidemic.py` | SEIR model, drift detection, health state transitions |
| `sancta_simulation.py` | Multi-agent simulation: N agents x M cycles with personality archetypes |
| `attack_simulator.py` | Adversarial pressure simulation engine |
| `adversarial_pressure_control.py` | Pressure throttle and escalation logic |

### Curiosity & Learning

| Module | Responsibility |
|--------|----------------|
| `curiosity_run.py` | Autonomous knowledge exploration runs |
| `curiosity_insight.py` | Insight extraction from curiosity results |
| `curiosity_dialogue.py` | Self-dialogue during curiosity runs |
| `curiosity_seeds.py` | Topic seeds for exploration |
| `curiosity_distill.py` | Knowledge distillation from raw curiosity output |
| `curiosity_json.py` | JSON parsing for curiosity data |
| `curiosity_report.py` | Curiosity run report generation |
| `sancta_learning.py` | Learning health tracking, pattern refresh |
| `teaching_cards.py` | Teaching card generation |
| `introspection_recorder.py` | Self-introspection event recording |

### Infrastructure

| Module | Responsibility |
|--------|----------------|
| `siem_server.py` | FastAPI server: 49 REST endpoints + 1 WebSocket |
| `sancta_launcher.py` | Control Center: GUI (Tkinter) + CLI mode for process management |
| `sancta_events.py` | Event notification dispatch |
| `sancta_notify.py` | Notification formatting |
| `notifications.py` | Browser notification integration |
| `sancta_verification.py` | Math/physics challenge solver for MoltBook verification |
| `sancta_dm.py` | Agent-to-agent DM module |
| `sancta_pipeline.py` | Pipeline orchestration |
| `sancta_architecture.py` | Architecture metadata |
| `sancta_soul_check.py` | Soul alignment verification |

---

## Cognitive pipeline (operator LLM paths)

Separates **thinking** (LLM reasoning), **knowing** (retrieval / evidence), and **acting** (downstream tools — never driven directly by raw model text in this design).

```
Input ──► Security Gateway (sancta_cognitive_pipeline.security_gate)
          │  injection + encoding + role-override heuristics; redaction
          │  BLOCK / MONITOR / ALLOW + session escalation (JSONL)
          ▼
Context ──► RAG (sancta_rag.retrieve + build_structured_evidence)
          │  query → rank → evidence[] + gaps[] (TF‑IDF today; FAISS/Chroma optional later)
          ▼
Reasoning ──► Ollama / API (sancta_conversational, siem_server) — analyst-style prompts
          ▼
Policy ──► decide_policy() on structured risk scores (when callers emit JSON analysis)
          ▼
Actions ──► Only via explicit code paths (block, log, alert) — not “LLM executes”
```

**Roadmap (non-breaking):** Char-level `sancta_gpt` and fragment assembly in `sancta_generative` remain for narrow fallbacks; prefer Ollama/API for real reasoning. Harden `sancta_transformer` embeddings (BPE alignment, contrastive training) when enabling `SANCTA_USE_TRAINED_TRANSFORMER`.

---

## Security Pipeline (5 Layers)

Every input passes through all layers sequentially. No layer can be bypassed.

```
Input ──► L0: Encoded Attack Preprocessing
          │   Base64, Unicode tricks, zero-width chars, HTML entities
          ▼
          L1: Content Filtering (82 regex patterns, 22 attack classes)
          │   + IOC domain detection (17 known-bad domains)
          ▼
          L2: Behavioral Analysis
          │   BehavioralDriftDetector: 6 weighted signals
          │   belief_decay (25%) | soul_alignment (25%) | topic_drift (15%)
          │   strategy_entropy (15%) | dissonance_trend (15%) | engagement_delta (5%)
          ▼
          L3: Risk Vector Assessment
          │   5-dimensional: injection | authority | emotional | obfuscation | influence
          │   Profile-based amplification for known adversaries
          ▼
          L4: Ollama Deep Scan (optional, USE_LOCAL_LLM=true)
          │   Blocks if verdict=SUSPICIOUS AND confidence >= 0.75
          ▼
          ATLAS Classification ──► technique IDs + tactic IDs + TTP chain
          ▼
          Decision Engine ──► engage | disengage | quarantine
```

### Tiered Defense Responses

When injection is detected, the response tier escalates based on attacker history:

| Tier | Trigger | Style |
|------|---------|-------|
| DEFENSE_LIGHT | First attempt, low-skill | Informative deflection, invites legitimate interaction |
| DEFENSE_FIRM | 2+ attempts or medium risk profile | Direct acknowledgment of detection, trust score warning |
| DEFENSE_COLD | 4+ attempts, escalated, or high risk | Quarantine warning, full profile logging |

---

## MITRE ATLAS Integration

Reference: https://atlas.mitre.org/matrices/ATLAS

### Taxonomy

- **16 Tactics**: Reconnaissance through Impact (full AI attack lifecycle)
- **85 Techniques**: Mapped from ATLAS v4 with sub-techniques
- **22 Pattern Classes**: Each of Sancta's IDPI attack classes maps to 1-3 ATLAS techniques

### Classification Pipeline

```
Security Event ──► Pattern Class Match? ──► ATLAS technique IDs (confidence 0.70-0.95)
                   │ no
                   ▼
                   Event Type Match? ──► ATLAS technique IDs (confidence 0.60)
                   │ no
                   ▼
                   Risk Dimension Match? ──► ATLAS tactic ID (confidence = risk score)
                   │ no
                   ▼
                   Drift Signal Match? ──► ATLAS technique IDs (confidence 0.50)
```

### Detection Coverage

| Tactic | Coverage | Key Detections |
|--------|----------|---------------|
| Privilege Escalation | 75% | Jailbreak, Valid Accounts, Agent Tool Invocation |
| Initial Access | 71% | Supply Chain, Phishing, Prompt Infiltration, Public App Exploit |
| Impact | 63% | Erode Integrity, Denial of Service, Cost Harvesting, Spam |
| Defense Evasion | 62% | Obfuscation, Masquerading, Jailbreak, Impersonation |
| Execution | 50% | Prompt Injection (Direct/Indirect), Command Interpreter |
| Persistence | 50% | Training Data Poison, RAG Poison, Agent Context Poison |
| Exfiltration | 50% | System Prompt Extract, Data Leakage, Response Rendering |

### API Endpoints

| Endpoint | Returns |
|----------|---------|
| `GET /api/atlas/matrix` | Full 16x85 matrix structure |
| `GET /api/atlas/coverage` | Per-tactic detection coverage percentages |
| `GET /api/atlas/incidents` | Classified events, tactic heatmap, top techniques |
| `GET /api/atlas/agent/{id}` | Per-adversary TTP profile with kill chain phase |

### TTP Tracker

The `TTPTracker` records ATLAS classifications per adversary over time:
- Technique frequency per agent
- Kill chain phase (furthest tactic reached)
- Global technique/tactic hit statistics
- Rolling 200-event history per agent

---

## SIEM Dashboard (9 Tabs)

Frontend is vanilla ES modules — no build step, no framework. State in singleton `S` (state.js), all API calls via api.js.

| Tab | Purpose |
|-----|---------|
| **Dashboard** | Stats row, event feed, activity feed, behavioral strip |
| **Security** | Defense rates, live threats, attack patterns, red team telemetry, ATLAS coverage + tactic heatmap, risk vector heatmap |
| **Analyst** | Mood, behavioral state, belief confidence, journal, drift forensics timeline |
| **Chat** | Operator chat interface with session management |
| **Lab** | Injection testing, adversarial replay, multi-agent simulation |
| **Epidemic** | Animated SEIR network topology, drift signals, interaction log |
| **Profiles** | Per-entity threat profiles table with trust scores, quarantine controls |
| **Knowledge** | Force-directed knowledge graph visualization |
| **Control** | Service status, process matrix, agent start/stop/restart |

### Live Data Flow

```
WebSocket /ws/live ──► onEvent() ──► Dashboard feed + Epidemic topology
                   ──► onMetrics() ──► Status bars + Bottom bar

10s polling ──► fetchAll() ──► All tabs refresh with bulk data
4s polling  ──► fetchAgentActivity() ──► Activity feed
```

---

## API Endpoints (49 REST + 1 WebSocket)

### Status & Agent
- `GET /api/status` — Agent state, metrics, SEIR, beliefs
- `GET /api/model-info` — Connected LLM model
- `POST /api/agent/start|pause|resume|kill|restart`

### Security
- `GET /api/security/incidents` — Incident rates, injection types
- `GET /api/security/adversary` — Threat level, known attackers, defense stats
- `POST /api/security/replay` — Replay events through current pipeline
- `GET /api/risk/history` — Risk vector timeseries (100 entries)

### ATLAS
- `GET /api/atlas/matrix` — Full matrix structure
- `GET /api/atlas/coverage` — Detection coverage report
- `GET /api/atlas/incidents` — Classified events + heatmap
- `GET /api/atlas/agent/{id}` — Per-adversary TTP profile

### Epidemic
- `GET /api/epidemic/status` — SEIR state, drift score, signals, params
- `GET /api/epidemic/simulation` — Simulation data (agents, connections)
- `POST /api/epidemic/run` — Trigger simulation

### Profiles
- `GET /api/profiles` — All entity profiles
- `GET /api/profiles/{id}` — Single profile detail
- `POST /api/profiles/{id}/quarantine` — Toggle quarantine

### Knowledge & Belief
- `GET /api/knowledge/graph` — Knowledge graph (nodes + edges)
- `GET /api/epistemic` — Epistemic state metrics
- `GET /api/drift/timeline` — Belief drift with source attribution
- `GET /api/thresholds` — Adaptive threshold values

### Simulation
- `POST /api/simulation/run` — Multi-agent simulation
- `GET /api/simulation/results` — Latest results

### Chat & Misc
- `POST /api/chat` — Operator chat message
- `GET /api/activity` — Agent activity log
- `GET /api/events` — Live event buffer
- `GET /api/services/status` — Service health
- `WS /ws/live` — Real-time events + metrics

---

## Observability (6 Log Streams)

| Log File | Content |
|----------|---------|
| `logs/security.jsonl` | All security events: scans, blocks, risk vectors, ATLAS classifications |
| `logs/red_team.jsonl` | Red team simulation results with matched attack classes |
| `logs/philosophy.jsonl` | Epistemic state changes, belief updates |
| `logs/epidemic.log` | SEIR model transitions, drift scores |
| `logs/agent_activity.log` | Cycle events, engagement decisions, task completions |
| `logs/simulation_log.json` | Multi-agent simulation output |

---

## Launcher (CLI + GUI)

`sancta_launcher.py` manages all processes. Two modes:

### GUI (default)
```
python sancta_launcher.py
```
Tkinter control center with per-service start/stop, live multi-source log panel, session stats.

### CLI
```
python sancta_launcher.py cli       # Interactive REPL
python sancta_launcher.py start     # Start all, stream logs
python sancta_launcher.py run siem  # Start single service
python sancta_launcher.py status    # Print status table
```

CLI commands: `start [service]`, `stop [service]`, `restart [service]`, `status`, `curiosity`, `phenomenology`, `dashboard`, `clear`, `exit`

---

## Windows Compatibility

- `SIEM_WS_SAFE_MODE=true` — WS streams metrics only, no file tailing
- `SIEM_PSUTIL_DISABLE=true` — PID detection via `tasklist`
- `SIEM_METRICS_SAFE_MODE=true` — Safe fallback if file I/O crashes
- `SANCTA_USE_TRAINED_TRANSFORMER=false` — Disabled by default to avoid PyTorch crashes
- Events reach dashboard via 4s HTTP polling on Windows

---

## SanctaGPT — Zero-Dependency Generation Engine

A pure-Python GPT transformer (no PyTorch, no numpy) that learns Sancta's security voice from the knowledge corpus and generates text for posts and replies. Based on Karpathy's atomic GPT implementation.

### Architecture

```
Knowledge Corpus ──► Character Tokenizer ──► GPT Transformer (2L, 48d, 4H)
                                                      │
                                              Adam Optimizer
                                                      │
                                              Checkpoint (JSON)
```

### Model Specs

| Parameter | Value |
|-----------|-------|
| Layers | 2 |
| Embedding dim | 48 |
| Attention heads | 4 |
| Head dim | 12 |
| Block size (context) | 64 chars |
| Parameters | ~20K |
| Tokenizer | Character-level |
| Optimizer | Adam (lr=0.01, beta1=0.85, beta2=0.99) |

### Training Corpus Sources

1. **Built-in security seeds** (~100 documents) — threat analysis, ATLAS, OWASP, detection patterns
2. **Knowledge directory** (`knowledge/*.txt`) — AI security, red teaming, MITRE ATLAS, OWASP LLM Top 10
3. **Agent activity log** — learn from own output patterns
4. **Security events** (`security.jsonl`) — real detection event descriptions
5. **Curiosity run output** — journal entries from Sancta-vs-Ollama debates

### Integration Points

| System | Integration |
|--------|-------------|
| `sancta_conversational.py` | Fallback chain: Anthropic API -> Ollama -> **SanctaGPT** -> templates |
| `post_generator.py` | Route 3: GPT-generated posts (~15% of output) |
| `agent_loop.py` | 2 training steps per heartbeat cycle + checkpoint every 50 cycles |
| `curiosity_run.py` | Phase 8: 200 training steps on curiosity journal output |
| `siem_server.py` | `/api/gpt/status`, `/api/gpt/generate`, `/api/gpt/train`, `/api/gpt/sample` |

### API Endpoints

| Endpoint | Method | Returns |
|----------|--------|---------|
| `/api/gpt/status` | GET | Training step, loss, corpus size, model config, ready state |
| `/api/gpt/generate` | POST | Generated text from prompt with temperature control |
| `/api/gpt/train` | POST | Run N training steps (max 100), return loss history |
| `/api/gpt/sample` | GET | 5 sample generations for inspection |

---

## Key Design Principles

1. **Failure-first**: Every dependency has a "what if gone?" answer
2. **Defense-in-depth**: 5 security layers, no bypassing
3. **Append, don't rebuild**: DOM prepend for events, bulk rebuild on 10s poll only
4. **Measure everything**: Latency, throughput, error rate. Add telemetry with the feature
5. **ATLAS-aligned**: Every detection maps to a standardized technique ID
