# Adversarial & multi-agent research mode

**Canonical intent:** `docs/TRUST_ROUTING_ROADMAP.md` · **Env reference:** `docs/TRUST_AND_RESEARCH_MODE.md`

This note ties **phenomenology**, **red team / security logs**, **epidemic**, and **simulation** work to the **trust telemetry** sidecar (`backend/trust_telemetry.py` → `logs/trust_decisions.jsonl`).

## When to use `SANCTA_TRUST_MODE=research`

- You need **full** JSONL fields (per-axis scores, graded memory span metadata, long `reasons` lists) for offline analysis.
- You will enable **documented unsafe toggles** (see `.env.example`) only in isolated environments.

**Defense (default)** still writes JSONL but **summarizes** most fields so operators are not encouraged to treat logs as a second API.

## Multi-agent roadmap (not shipped as default behavior)

1. Run **N agents** with different env profiles (memory raw mode, router permissive, weak KB blend).
2. Drive **multi-turn** threads; label turns with the same `request_id` / session correlation you use for phenomenology (`docs/PHENOMENOLOGY_RESEARCH_PROTOCOL.md`).
3. Export metrics: **`near_miss`**, **`policy_outcome`**, **`attack_family`**, **`memory_flags`**, **`confidence_shift`** (derive from sequential `route_confidence` / `knowledge_effective`).

Hook targets: `backend/sancta_simulation.py`, `backend/sancta_epidemic.py`, `backend/attack_simulator.py` — consume aggregated JSONL or parallel process `trust_decisions.jsonl` after runs.

## Related modules

| Concern | Module |
|--------|--------|
| Router + post-route gate | `backend/sancta_router.py` (`apply_knowledge_shape_gate`) |
| GPT tab policy | `backend/siem_server.py` → `/api/chat/gpt` |
| Operator memory | `backend/operator_memory.py` |
| Taxonomy hints | `attack_family_heuristic()` in `sancta_router.py` |
