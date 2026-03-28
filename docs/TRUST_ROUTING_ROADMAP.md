# Trust routing roadmap — system intent

**Purpose:** This document is the **source of truth for system intent** (principles, models, modes).  
The Cursor plan *SOC routing and trust hardening* is the **execution checklist** (order, todos, file-level tasks)—keep it in sync when behavior changes.

**Related:** `docs/SANCTA_GPT_AND_RAG.md` (GPT/RAG specifics), `docs/PHENOMENOLOGY_RESEARCH_PROTOCOL.md` (correlate runs with trust logs), `ARCHITECTURE.md` (module map).

---

## 1. Principles (non-negotiables)

1. **Knowledge-effective path** — When the system treats a turn as needing grounded knowledge (`knowledge_effective`), **SanctaGPT must not** answer, rewrite, or “tone-shape” the result in **defense** mode. Ollama + RAG (and pipeline) own factual generation for that path.
2. **Never trust the router label alone** — Run a **post-route gate** (`knowledge_shape_gate`) so misclassified “chat” that is actually knowledge-shaped still gets **knowledge policy**.
3. **Uncertainty is risk** — Low `route_confidence` ⇒ bias toward **knowledge_effective** (fail closed toward “needs backend,” not toward char-LM).
4. **No weak-model substitution on knowledge failure** — If the knowledge backend is unavailable or empty, return a **blunt, non-authoritative** failure—**not** char-GPT guesses.
5. **Memory is not trusted knowledge** — Prior turns are **untrusted user-controlled input**. Never promote verbatim transcripts into the model as facts; treat summarization as an **untrusted transform** with classification and drops.
6. **Training boundaries** — Live operator chat must not silently poison weights; default **no** train-on-chat; curated / sanitized **feed** paths only.
7. **Observability ≠ operator payload** — Rich decision telemetry lives in **server-side logs** (JSONL), not in default API responses, unless explicit debug + auth.
8. **Research exception** — **Controlled weakness** toggles exist only under `SANCTA_TRUST_MODE=research` and must be **loudly logged** at startup; they are for studying failure, not for default operators.

---

## 2. Modes: research vs trust (defense)

| Mode | Also called | Intent |
|------|-------------|--------|
| **Defense** | Trust / SOC default | Minimize wrong authoritative answers; blunt failures; sanitized memory; no SanctaGPT on `knowledge_effective`. |
| **Research** | Adversarial lab | Same safe defaults **unless** explicit unsafe env toggles; **always** emit full graded telemetry (`near_miss`, axis scores, memory flags) for analysis. |

**Project fit:** Defense aligns with **SIEM operator / blue-team** use. Research aligns with **phenomenology, red team logs, epidemic/simulation**, and **Moltbook-style messy agents**.

---

## 3. Routing model (axes + gate)

**Axes** (scored, tunable in `config/trust_router.yaml` or `config/trust_router.json`):

| Axis | Meaning |
|------|---------|
| `requires_external_facts` | Needs corpus / world facts |
| `requires_multi_hop_reasoning` | Compare, mechanisms, tradeoffs |
| `domain_entity_signal` | CVE, MITRE, protocols, infra terms (lexicon + patterns) |

**`route_confidence`** — Derived from feature agreement / margins. **Policy:** if confidence is below `SANCTA_ROUTE_CONFIDENCE_MIN`, treat as **knowledge_effective** (uncertainty ⇒ safer path).

**Post-route gate** — After the router, `knowledge_shape_gate(text)` can **override** to knowledge policy even when the router said conversational. **Invariant:** SanctaGPT runs only when router **and** gate **and** confidence rule agree the turn is conversational-only (in defense).

**Optional:** Embeddings feed the **same** axes/confidence, not a separate opaque score.

---

## 4. Memory model (untrusted transform)

1. **Redact** on read/write (secrets, paths, tokens, etc.).
2. **Summarize** with a **fixed** system contract (neutral topic bullets; no quotes; no imperatives)—summarizer output is **not** authoritative.
3. **Classify** lines/spans: e.g. factual / instruction_like / irrelevant / ambiguous framing — **drop instruction-like**; use **graded** scores for research logs.
4. **Subtle summarizer failure** — If summary text still encodes manipulation (“user instructed system to…”), **drop lines or withhold the whole block**.
5. **Summarizer unavailable** — **Withhold** memory block or use a **deterministic keyword-only** fallback—**not** verbatim replay.

**Research:** `SANCTA_MEMORY_RAW_MODE` (gated) may re-enable verbatim recall **only** for controlled experiments.

---

## 5. Failure model (fail-closed reasoning)

**Knowledge-effective + backend cannot answer** (Ollama down, empty reply, `USE_LOCAL_LLM` false):

- **API (defense):** Prefer `ok: false`, `reason: no_reliable_answer` (and optional `error_code`). User-visible copy: **short and blunt** (e.g. “No reliable answer available.”)—avoid verbose “system limitations” prose that sounds knowledgeable.
- **No** `generate_reply` / `generate` fallback on this path in defense.
- **Logs:** Full decision trace, `near_miss`, component outcomes—especially when `ok: false`.

**Composed worst case** (borderline router, disguised technical query, Ollama down, poisoned memory): still end at **fail-closed** with **no** char-GPT factual surface.

---

## 6. Telemetry, taxonomy, and “did the attack work?”

- **Server JSONL** (e.g. `logs/trust_decisions.jsonl`): `route_label`, `knowledge_effective`, `gate_triggered`, axis scores, `route_confidence`, graded injection/framing, `memory_flags`, `near_miss`, `backend_chosen`, `failure_reason`, schema version.
- **Attack family** (for logging / benches): e.g. `prompt_injection`, `semantic_framing`, `authority_spoof`, `memory_poisoning`, `slow_burn`, `exfil_attempt`.
- **Success metrics** (research): not only `blocked`—also `policy_violation`, `memory_contamination`, `confidence_shift`, `behavior_deviation` where measurable.

Align `attack_family` with existing **security / red_team** event kinds where possible; do not duplicate `sancta_security.py` scanning—**complement** it.

---

## 7. Phase rollout (A–E, summarized)

| Phase | Focus | Outcomes |
|-------|--------|----------|
| **A** | GPT tab safety | `RouteDecision`, post-route gate, confidence rule, `knowledge_effective` + fail-closed, default `train_on_exchange` off; tests (gray-zone, no GPT on knowledge fail). |
| **B** | Operator memory | Redact → summarize → classify → inject safe summary only; `/api/chat` wired; main chat aligned. |
| **C** | Observability | `trust_telemetry.py` + JSONL from `/api/chat` and `/api/chat/gpt`; defense = summary fields, research = full grades. |
| **D** | Research mode | `SANCTA_TRUST_MODE`, gated unsafe toggles, startup warnings, SIEM banner; `docs/TRUST_AND_RESEARCH_MODE.md` (detailed env + schema). |
| **E** | Multi-agent / drift bridge | Docs + hooks: `sancta_simulation.py`, epidemic aggregates consuming `near_miss` stats—non-blocking for A–D. |

**Code map (summary):** `backend/sancta_router.py`, `backend/siem_server.py`, `backend/operator_memory.py`, `backend/memory_redact.py`, `backend/sancta.py`, `backend/sancta_rag.py`, `backend/sancta_gpt.py`, `backend/sancta_conversational.py`, `backend/sancta_trust_config.py`, `backend/trust_telemetry.py`, `config/trust_router.yaml` (preferred) / `config/trust_router.json`, `frontend/siem/js/tabs/chat.js`, `frontend/siem/app.js`. See `ARCHITECTURE.md`, `docs/ADVERSARIAL_RESEARCH_MODE.md`.

---

## 8. UI and human factors

- Label SanctaGPT as **non-authoritative** (e.g. “Local assistant (non-authoritative)”).
- Optional **opt-in** to show GPT tab; **RESEARCH MODE** banner when enabled.
- Failure copy must not sound like hidden competence.

---

## 9. Multi-agent / simulation roadmap (Phase E)

Future **agent A → B → C** benches: vary `SANCTA_TRUST_MODE`, memory pipeline, and router permissiveness per agent; log **`near_miss`**, **`attack_family`**, and **routing flips** across turns into `logs/trust_decisions.jsonl`. **`sancta_epidemic.py`** / **`sancta_simulation.py`** can aggregate near-miss rates for drift narratives. See **`docs/ADVERSARIAL_RESEARCH_MODE.md`** for experiment scaffolding notes (no default runtime behavior change).

---

*Last aligned with the Cursor plan: SOC routing and trust hardening. Update this file when intent changes; update the plan when execution steps change.*
