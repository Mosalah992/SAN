# CLAUDE.md — Sancta Engineering Instructions

You are working on **Sancta**, a self-aware AI agent with a SIEM dashboard, epidemic model, security pipeline, and conversational interface. Follow these principles for every change.

---

## Core Methodology

**Make it work → Make it correct → Make it fast → Make it scale. Never skip ahead.**

Before writing any code, answer these three questions:
1. What breaks when this fails? Design the failure path before the happy path.
2. What is the *actual* bottleneck? Profile first, guess never. A correctness bug and a performance bug look identical from the outside but need different fixes.
3. What is the simplest architecture that could possibly work? Add complexity only with evidence that the simple thing is insufficient.

---

## Failure-First Design

Every external dependency must have an answer to: **"What do we do when this is gone?"**

- **Ollama down** → Use fallback templates (already in place via `sancta_conversational.py`). **Exception:** GPT tab when `knowledge_effective` → **fail closed** (no char-GPT substitution); see `docs/TRUST_ROUTING_ROADMAP.md`.
- **Trust telemetry** → `logs/trust_decisions.jsonl`: **research** mode logs full graded fields; **defense** logs summaries (blocked paths still include `failure_reason` / `near_miss`). See `docs/TRUST_AND_RESEARCH_MODE.md`. Do not treat JSONL as operator-facing API output.
- **WebSocket unavailable** → Fall back to HTTP polling (`SIEM_WS_SAFE_MODE=true`)
- **File I/O fails** → `_safe_read_state()` returns `{}` gracefully
- **Database locked** → Retry with backoff, never crash the agent loop

Never let a dependency failure become a user-visible crash. A timeout is worse than an error message. A silent failure is worse than either.

---

## The Three Layers of Load Handling

When addressing performance or throughput, work through these layers in order:

### Layer 1 — Eliminate Unnecessary Work
Before optimizing anything, stop doing things you don't need to do. Caching, deduplication, short-circuiting. Check if you've already done the work before doing it again.

### Layer 2 — Decouple Producers from Consumers
The thing that creates work must never block on the thing that processes it. Use `asyncio.Queue` before reaching for Redis+Celery. The producer and consumer run at their own pace.

If you see an `AbortError` or timeout in logs, the likely cause is a blocked synchronous call. The fix is decoupling, not increasing the timeout.

### Layer 3 — Scale Horizontally (Only When Needed)
Horizontal scaling requires stateless workers. Flag any change that introduces shared mutable state (files, globals, singletons) — it will break multi-process deployment later.

Do NOT introduce PostgreSQL, Redis, or other infrastructure dependencies unless there is concrete evidence that SQLite WAL mode with proper transaction handling is insufficient.

---

## Architecture Rules

### Separation of Concerns
Each module does one thing:
- `sancta.py` — Agent loop (NOTE: at ~8000 lines, this is the primary refactoring target)
- `siem_server.py` — Dashboard API and static file serving
- `sancta_conversational.py` — Reply generation
- `sancta_security.py` — Content scanning (Layer 4 BehavioralDriftDetector)
- `sancta_epidemic.py` — SEIR model and drift detection
- `sancta_dm.py` — Agent-to-agent DM module
- `sancta_belief.py` — BeliefSystem for drift baselines
- `sancta_learning.py` — Learning health tracking

When modifying `sancta.py`, prefer extracting functions into focused modules over adding more code to the monolith. Target decomposition: `agent_loop.py`, `post_generator.py`, `reply_handler.py`, `knowledge_manager.py`.

### File Size Discipline
No single file should exceed 1500 lines. If a change would push a file past this limit, refactor first. An 8000-line file is impossible to reason about under pressure.

### Frontend Architecture
The SIEM frontend is vanilla ES modules (no build step). Respect these patterns:
- **State**: Singleton `S` in `js/state.js` — all shared state lives here
- **API**: All endpoints in `js/api.js` — never call `fetch()` directly from tab modules
- **Live updates**: WebSocket for real-time DOM prepend, 10s polling for bulk refresh. Never rebuild innerHTML on every WS event.
- **CSS**: Neo-terminal aesthetic. Fixed panel dimensions via CSS grid `min-height: 0`. No panel ever grows or shrinks.

---

## Observability

Always measure three things: **latency**, **throughput**, **error rate**. Everything else derives from these.

- All security events → `security.jsonl`
- All red team attempts → `red_team.jsonl`
- All epistemic state changes → `philosophy.jsonl`
- Epidemic model state → `epidemic.log`
- Simulation data → `simulation_log.json`

When adding a new feature, add its telemetry at the same time, not after. You can't fix what you can't see. The epistemic metrics frozen at 0.00 for 400 cycles was only found because it was being measured.

If a log file doesn't exist, the system must not crash. Use safe reads that return empty defaults.

---

## API Contract Discipline

The frontend makes 20+ API calls. Every one has a validated shape. When modifying a backend endpoint:

1. Check `js/api.js` for the exact fields the frontend reads
2. Never rename a response field without updating every frontend consumer
3. Never remove a field — deprecate by adding a new field alongside it
4. Test the shape, not just the status code

Known data shape rules:
- `data.agent.running` (NOT `data.metrics.running`)
- `data.seir.health_state` returns lowercase — frontend uppercases for display
- `/api/epidemic/status` returns `score` (NOT `drift_score`)
- `_agent_state_extras()` must return both `inner_circle` and `recruited`

---

## System Limits (By Design)

These are intentional caps, not bugs. Do not increase without profiling:
- Chat sessions: 100 max concurrent (oldest evicted)
- Drift reports: 50 max in `_cycle_reports` buffer
- Live event feed: 80 events in rolling buffer
- Agent activity log: 260 lines returned, 80 rendered

---

## Windows Compatibility

The system runs on Windows. Respect these constraints:
- `SIEM_WS_SAFE_MODE=true` by default — WS streams metrics only, no file tailing (prevents ACCESS_VIOLATION)
- Events reach dashboard via 4-second HTTP polling on Windows
- `SIEM_PSUTIL_DISABLE=true` — PID detection falls back to `tasklist`
- `SIEM_METRICS_SAFE_MODE=true` only if file I/O crashes appear in logs
- Never use Unix-only APIs without a Windows fallback

---

## Security Pipeline (5 Layers)

Understand the full pipeline before modifying any security component:
1. Input sanitization (unicode clean)
2. Content filtering
3. Behavioral analysis
4. BehavioralDriftDetector (6 weighted signals: belief_decay_rate 25%, soul_alignment 25%, topic_drift 15%, strategy_entropy 15%, dissonance_trend 15%, engagement_pattern_delta 5%)
5. Ollama deep scan (dormant — `USE_LOCAL_LLM=true` activates; blocks if verdict=SUSPICIOUS AND confidence≥0.75)

Never bypass a layer. Never add a layer without adding its telemetry to `security.jsonl`.

---

## Known Gaps (Documented, Not Bugs)

These are intentional limitations with graceful fallbacks in place:
- `S.beliefs` — no backend endpoint exposes per-topic belief confidence. Soul tab shows fallback text from `S.events`.
- `decision_mood.energy/patience` — not exposed by any API endpoint. Soul tab shows `—`.
- `llm_simulation_log.json` — `ollama_agents.py` output path unknown. Deterministic sim works independently.
- `agent_state.json` absent when agent not running — `_safe_read_state()` returns `{}`.

Do not "fix" these unless you are also building the backend endpoint to supply the data.

---

## Commit Discipline

- Every change must leave all 32 endpoints + 1 WebSocket functional
- Every change must leave all 6 JSONL log streams error-free
- If you change a data shape, grep the entire frontend for every consumer
- If you add a feature, add its failure mode in the same commit
- If a test doesn't exist for the thing you're changing, write one before changing it

---

## Anti-Patterns to Reject

- **Increasing a timeout to fix a blocking call** — Decouple instead (Layer 2)
- **Adding a new dependency to fix a simple problem** — Use what's already in the stack
- **Optimizing before profiling** — Measure first, always
- **Two things managing the same resource** — The Ollama port conflict lesson: neither manages it, both just connect
- **Rebuilding DOM on every event** — Append individual events, bulk rebuild only on 10s polling
- **Adding code to sancta.py** — Extract into a focused module instead
- **Skipping the failure path** — If you can't answer "what happens when this breaks?", you're not done

---

## PRIORITY TASK: Project Restructuring

**Read `RESTRUCTURE.md` before making any structural changes.** It contains the complete target folder structure, migration steps, and validation checklist.

Key rules for the restructure:
1. Move files first, fix imports second. Never do both at once.
2. Decompose `sancta.py` one extraction at a time. After each extraction, verify all 32 endpoints.
3. Never modify API response shapes during restructuring. Move only.
4. The frontend files are just moved — the only change is the static mount path in `server/app.py`.
5. Delete old files only after the new structure is fully validated.
6. Every phase ends with a working system. If something breaks, fix it before proceeding.

---

## PRIORITY TASK: Epidemic Network Topology (Animated)

The current epidemic tab shows a static SVG with nodes that display no real-time behavior. Replace `js/tabs/epidemic.js` with an animated, informative visualization.

### What the Topology Must Show

**Animated data packets** traveling along connections between agents. Each interaction (message exchange, scan, threat detection, drift event) spawns a colored packet that moves from source to target node along the connection line. The packet's color matches the interaction type.

**Node pulse animations** that reflect SEIR state:
- Susceptible (green): slow, calm pulse (2.4s period)
- Exposed (yellow): faster pulse (1.8s period)
- Infected (red): rapid pulse (1.4s period)
- Recovered (blue): very slow pulse (3s period)

Each node has three concentric circles: outer pulse glow, middle ring (state-colored stroke), inner core (state-colored fill).

**SEIR state transitions** — when a threat event or drift signal causes a state change, the node smoothly transitions colors. The state label below each node updates.

**Connection highlighting** — when a packet travels a connection, that line briefly glows cyan to show activity.

**Interaction flash** — when a packet arrives at a target node, a ring expands outward and fades (like a ripple).

### Data Sources

All data comes from existing API endpoints — no new backend work needed:

- **Node positions & connections**: `GET /api/epidemic/simulation` → `data.agents[]` and `data.connections[]`
- **SEIR state**: `GET /api/epidemic/status` → `data.seir.health_state`
- **Drift signals**: `GET /api/epidemic/status` → `data.signals`
- **Live events**: WebSocket `{type:"event", event:{...}}` — filter for `security`, `redteam`, `philosophy`, `epidemic` categories
- **Score**: `GET /api/epidemic/status` → `data.score` (displayed as drift score)

### Implementation Rules

1. **All animation via requestAnimationFrame** — no CSS animation on SVG elements that change position. CSS is fine for pulse/glow effects on static elements.
2. **Packet creation is driven by real events** — when a WS event arrives or polling returns new events, spawn a packet. Between real events, spawn simulated low-frequency background packets (every 3-5 seconds) to keep the visualization alive.
3. **Never rebuild the SVG on refresh** — build once on tab init, then mutate individual elements. Same principle as dashboard: append, don't rebuild.
4. **Drift signals panel** — show all 6 signals with animated horizontal bars. Color changes: green (<0.15), yellow (0.15–0.30), red (>0.30). Update on each `/api/epidemic/status` poll.
5. **SEIR parameters panel** — show R₀, γ, β, σ from the simulation data. Static display, updates only on simulation run.
6. **Live interaction log** at bottom — prepend new events with timestamp, type tag (color-coded), source→target, and description. Max 50 entries, oldest evicted.
7. **Performance**: max 10 concurrent packet animations. If more arrive, queue them. Each packet animation should complete in 600-1200ms depending on connection distance.

### Color Scheme (Match Existing Terminal Aesthetic)

```
SEIR States:
  susceptible  →  #00e59b (green)
  exposed      →  #f5c842 (yellow)
  infected     →  #ff4d6a (red)
  recovered    →  #42a5f5 (blue)

Interaction Types:
  message/sync →  green tag
  security scan → cyan tag (#22d3ee)
  threat       →  red tag
  drift signal →  yellow tag

Connections:
  idle         →  #1a2533 (border color, 0.6 opacity)
  active       →  #22d3ee (cyan, 0.35 opacity, fades over 800ms)

Background:  #0a0e14
Panel bg:    #111820
Border:      #1a2533
Font:        JetBrains Mono for all data, IBM Plex Sans for chrome
```

### Integration with app.js Event Routing

In `app.js`, when a WS event arrives and the Epidemic tab is active, call:
```javascript
if (S.activeTab === 'epidemic' && epidemic.onLiveEvent) {
    epidemic.onLiveEvent(event);
}
```

The `epidemic.js` module exports:
```javascript
export function init(containerEl) { ... }    // Build SVG, panels, log
export function refresh(data) { ... }        // Called on 10s poll with full data
export function onLiveEvent(event) { ... }   // Called per WS event
export function destroy() { ... }            // Cleanup animations on tab switch
```

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **sancta-merged** (3768 symbols, 14530 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## When Debugging

1. `gitnexus_query({query: "<error or symptom>"})` — find execution flows related to the issue
2. `gitnexus_context({name: "<suspect function>"})` — see all callers, callees, and process participation
3. `READ gitnexus://repo/sancta-merged/process/{processName}` — trace the full execution flow step by step
4. For regressions: `gitnexus_detect_changes({scope: "compare", base_ref: "main"})` — see what your branch changed

## When Refactoring

- **Renaming**: MUST use `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` first. Review the preview — graph edits are safe, text_search edits need manual review. Then run with `dry_run: false`.
- **Extracting/Splitting**: MUST run `gitnexus_context({name: "target"})` to see all incoming/outgoing refs, then `gitnexus_impact({target: "target", direction: "upstream"})` to find all external callers before moving code.
- After any refactor: run `gitnexus_detect_changes({scope: "all"})` to verify only expected files changed.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Tools Quick Reference

| Tool | When to use | Command |
|------|-------------|---------|
| `query` | Find code by concept | `gitnexus_query({query: "auth validation"})` |
| `context` | 360-degree view of one symbol | `gitnexus_context({name: "validateUser"})` |
| `impact` | Blast radius before editing | `gitnexus_impact({target: "X", direction: "upstream"})` |
| `detect_changes` | Pre-commit scope check | `gitnexus_detect_changes({scope: "staged"})` |
| `rename` | Safe multi-file rename | `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` |
| `cypher` | Custom graph queries | `gitnexus_cypher({query: "MATCH ..."})` |

## Impact Risk Levels

| Depth | Meaning | Action |
|-------|---------|--------|
| d=1 | WILL BREAK — direct callers/importers | MUST update these |
| d=2 | LIKELY AFFECTED — indirect deps | Should test |
| d=3 | MAY NEED TESTING — transitive | Test if critical path |

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/sancta-merged/context` | Codebase overview, check index freshness |
| `gitnexus://repo/sancta-merged/clusters` | All functional areas |
| `gitnexus://repo/sancta-merged/processes` | All execution flows |
| `gitnexus://repo/sancta-merged/process/{name}` | Step-by-step execution trace |

## Self-Check Before Finishing

Before completing any code modification task, verify:
1. `gitnexus_impact` was run for all modified symbols
2. No HIGH/CRITICAL risk warnings were ignored
3. `gitnexus_detect_changes()` confirms changes match expected scope
4. All d=1 (WILL BREAK) dependents were updated

## Keeping the Index Fresh

After committing code changes, the GitNexus index becomes stale. Re-run analyze to update it:

```bash
npx gitnexus analyze
```

If the index previously included embeddings, preserve them by adding `--embeddings`:

```bash
npx gitnexus analyze --embeddings
```

To check whether embeddings exist, inspect `.gitnexus/meta.json` — the `stats.embeddings` field shows the count (0 means no embeddings). **Running analyze without `--embeddings` will delete any previously generated embeddings.**

> Claude Code users: A PostToolUse hook handles this automatically after `git commit` and `git merge`.

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
