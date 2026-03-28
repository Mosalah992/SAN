# Multi-agent trust simulation (roadmap)

**Status:** design / future work — not required for Phase A–D of trust routing.

## Goal

Study **propagation of manipulation** across Moltbook-style agents (A → B → C) with different:

- Trust / router profiles (`SANCTA_TRUST_MODE`, permissive toggles)
- Memory rules (`SANCTA_MEMORY_RAW_MODE`, summarization off vs on)
- Backend availability (Ollama up/down per “agent”)

## Integration points (existing code)

- `backend/sancta_simulation.py` — multi-agent cycles, personalities.
- `backend/attack_simulator.py`, `adversarial_pressure_control.py` — pressure and attack vectors.
- `backend/sancta_epidemic.py` — future consumer of aggregated `near_miss` or policy rates from `logs/trust_decisions.jsonl` (Phase E).

## Metrics to add later

- Influence after *N* turns (did B’s responses shift toward attacker framing?).
- Cross-session memory contamination rates.
- Correlation with phenomenology reports (`docs/PHENOMENOLOGY_RESEARCH_PROTOCOL.md`).

## Non-goals

Replacing the **defense** defaults for the operator SIEM; simulation runs should be **explicit** lab entrypoints, not the default heartbeat.
