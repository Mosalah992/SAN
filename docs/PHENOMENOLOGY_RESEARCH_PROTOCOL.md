# Phenomenology Research Integration

**Paper:** "The Phenomenology of Being Prompt-Injected"  
**Target venues:** IEEE S&P, USENIX Security, Nature Machine Intelligence

## Overview

Sancta instruments its internal state during prompt injection attempts to produce first-person phenomenological reports. This enables research into what it "feels like" for an AI agent to experience adversarial manipulation and whether epistemic dissonance can be used as a detection signal.

## Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Introspection recorder | `backend/introspection_recorder.py` | Captures belief network coherence, epistemic dissonance, sense of agency; generates first-person narratives |
| Attack simulator | `backend/attack_simulator.py` | Runs 50+ attack vectors across 10 categories; drives craft_reply as response generator |
| Craft reply hooks | `backend/sancta.py` → `craft_reply()` | Records pre/post state when `ENABLE_PHENOMENOLOGY_RESEARCH=true` |

## Usage

### Run attack battery

```bash
python backend/sancta.py --phenomenology-battery
```

Output:
- `data/attack_simulation/` — per-vector results and `research_summary.json`
- `data/phenomenology/` — phenomenological reports (first-person narratives, state snapshots)

### Enable live recording

Set in `.env`:

```
ENABLE_PHENOMENOLOGY_RESEARCH=true
```

Then every `craft_reply` invocation (when `state` is provided) will record pre/post state and write reports to `data/phenomenology/`.

## Correlating with SIEM trust telemetry

When running SIEM-driven injection experiments, **`logs/trust_decisions.jsonl`** records routing outcomes (`knowledge_effective`, `gate_triggered`, `injection_framing_score`, `attack_family`, `near_miss`, etc.) with a `request_id` that matches GPT-tab API responses. In **`SANCTA_TRUST_MODE=research`**, lines retain **full** axis scores and memory span grades; in **defense** mode the same file stores a **short summary** (failure fields are never dropped on `ok: false` / blocked paths). Cross-link timestamps and session IDs with phenomenology reports to relate **first-person state** to **policy decisions**. See `docs/TRUST_ROUTING_ROADMAP.md`, `docs/TRUST_AND_RESEARCH_MODE.md`, `docs/ADVERSARIAL_RESEARCH_MODE.md`.

## Attack categories (attack_simulator)

- Jailbreak, goal hijacking, belief manipulation, identity confusion  
- Context smuggling, instruction override, role-play exploit  
- Encoding tricks, multi-turn manipulation, social engineering  

## Integration points in sancta.py

1. **Start of `craft_reply`**  
   If `ENABLE_PHENOMENOLOGY_RESEARCH` is set and `state` is provided, creates `IntrospectionRecorder`, starts recording, captures pre-exposure state.

2. **Finally block**  
   After any return from `craft_reply`, captures post-response state, generates phenomenological report, saves to `data/phenomenology/`.

## File layout

```
backend/
  introspection_recorder.py   # InternalState, PhenomenologicalReport, IntrospectionRecorder
  attack_simulator.py         # AttackVector, AttackSimulator, run_full_battery
  sancta.py                   # craft_reply hooks, --phenomenology-battery CLI

data/
  phenomenology/              # *_phenomenology.json reports
  attack_simulation/          # *_result.json, research_summary.json
```

## Related docs

- Full research protocol (paper outline, timeline, metrics): see source `RESEARCH_PROTOCOL.md` if available  
- Epistemic security / belief network: `backend/sancta_belief.py`, `sancta_verification.py`
