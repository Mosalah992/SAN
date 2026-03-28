# Trust mode vs research mode — env reference

Canonical intent: **`docs/TRUST_ROUTING_ROADMAP.md`**. This page lists environment variables and API hooks.

## Modes

| Env | Meaning |
|-----|---------|
| `SANCTA_TRUST_MODE=defense` | Default. SOC-style: fail-closed knowledge path, no weak-model substitution. |
| `SANCTA_TRUST_MODE=research` | Enables logging of unsafe toggles at startup; same routing defaults until toggles set. |

## Telemetry

| Env | Meaning |
|-----|---------|
| `SANCTA_TRUST_TELEMETRY=false` | Disable append to `logs/trust_decisions.jsonl`. |

Each line is JSON with `schema_version`, `ts`, `event_id`, plus fields such as `endpoint`, `route_label`, `knowledge_effective`, `gate_triggered`, `axes`, `route_confidence`, `attack_family`, `injection_framing_score`, `near_miss`, `backend_chosen`, `failure_reason`, `policy_outcome`.

## Research-only unsafe toggles

Ignored unless `SANCTA_TRUST_MODE=research`:

| Env | Effect |
|-----|--------|
| `SANCTA_ALLOW_WEAK_KB_BLEND=1` | When knowledge backend fails on GPT tab, allow SanctaGPT fallback (measurement). |
| `SANCTA_MEMORY_RAW_MODE=1` | Operator memory uses redacted verbatim blocks instead of summary/extractive. |
| `SANCTA_ROUTER_PERMISSIVE=1` | Relaxes router margin and post-route gate threshold. |

## SIEM API

- `GET /api/trust/status` — `{ trust_mode, unsafe_active, unsafe_toggles }` for the **RESEARCH MODE** banner.

## Router tuning

Edit **`config/trust_router.yaml`** (preferred) or `config/trust_router.json` (fallback if YAML missing or PyYAML unavailable):

- `confidence_min` — below this margin between knowledge vs chat scores → treat as `knowledge_effective`.
- `gate_threshold` — content signal for the **post-route** `apply_knowledge_shape_gate` baseline (inner router may use a relaxed threshold in research permissive mode; the shape gate still uses this base value).
- `knowledge_margin` — how much knowledge score must exceed chat score for a “clear” conversational win.
- `axis_weights` — per-axis weighting for the combined signal.

## Telemetry detail level

| Mode | JSONL |
|------|--------|
| Defense | Summary fields (`route_label`, `gate_triggered`, `failure_reason`, `near_miss`, …); omits full `axes` / long reason lists |
| Research | Full event dict as emitted by `siem_server` (axes, reasons, graded memory metadata when present) |
