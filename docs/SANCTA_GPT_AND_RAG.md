# SanctaGPT vs RAG vs Ollama — responsibilities

**Trust/routing intent (modes, gates, memory, fail-closed policy):** [`TRUST_ROUTING_ROADMAP.md`](TRUST_ROUTING_ROADMAP.md) · research env/telemetry: [`TRUST_AND_RESEARCH_MODE.md`](TRUST_AND_RESEARCH_MODE.md). This doc focuses on component roles and env knobs.

## What SanctaGPT is

`sancta_gpt.py` is a **tiny character-level** transformer (pure Python). It is a **style / fallback / offline** generator:

- **Good for:** light continuity, persona-ish blurbs when no API, toy experiments.
- **Not for:** storing facts, security reasoning, or faithful knowledge recall.

Lower training loss means **smoother character prediction**, not intelligence.

## What it should *not* learn

Avoid mixing in one undifferentiated corpus:

| Source            | Role                         | SanctaGPT default        |
|------------------|------------------------------|---------------------------|
| `User:` / `Sancta:` dialogue | Persona + turn-taking        | Use curriculum / `--light` smoke |
| Security seeds   | Tone / phrasing              | OK after dialogue phase   |
| `knowledge/`     | **Facts**                    | **Do not** “train into weights”; use RAG |
| Logs / JSONL     | Runtime telemetry            | **Excluded** unless `SANCTA_GPT_INCLUDE_LOGS=true` |

Default `build_corpus()` fills **explicit pools** (`_pool_convo`, `_pool_security`, `_pool_knowledge`, optional `_pool_telemetry`) then concatenates `_docs`. Training targets a pool via `set_training_mode("convo"|"security"|"knowledge"|"telemetry"|"all")` — **not** fragile index slicing across chunk boundaries. Legacy `set_training_doc_range` still forces `mode="all"`.

## Where intelligence lives

1. **Detection / pipeline** — `sancta_security.py`, logs, policies (strongest structured signal).
2. **Memory of facts** — `knowledge/`, `knowledge_db.json`, ingestion — retrieved at **runtime**, not memorized by SanctaGPT.
3. **Reasoning** — **Ollama / API** with `get_ollama_knowledge_context()` + **`sancta_rag`** (TF-IDF over `knowledge/` by default).
4. **SanctaGPT** — optional last-mile **tone** or **fallback** only.

## Control flow (GPT tab) — `knowledge_effective` vs conversational

`sancta_router.route_gpt_tab_decision()` computes **`knowledge_effective`** using axes (`requires_external_facts`, `requires_multi_hop_reasoning`, `domain_entity_signal`), **low route confidence** (ambiguous → bias to knowledge), and a **post-route content gate**. Legacy string route: `route_gpt_tab_message()`.

**`/api/chat/gpt`:**

- **`knowledge_effective`** — Ollama + `get_ollama_knowledge_context` only. **No SanctaGPT** on this path. If Ollama is down / empty / `USE_LOCAL_LLM` false → **`ok: false`**, `error`: “No reliable answer available.” (defense). **Research:** `SANCTA_ALLOW_WEAK_KB_BLEND=1` (with `SANCTA_TRUST_MODE=research`) allows char-GPT fallback for experiments only.
- **Conversational** — SanctaGPT first, then generic Ollama, then short fallback text.

JSON: **`route`**, **`backend`**, **`knowledge_effective`**, optional **`request_id`** (correlate with `logs/trust_decisions.jsonl`).

**Training:** Default **no** train-on-chat. Opt-in: SIEM checkbox “Learn from chat (unsafe)” or `SANCTA_GPT_TRAIN_ON_CHAT=true`. Training runs only on the **conversational** path (never on `knowledge_effective` replies). Trusted corpus: **`/api/chat/gpt/feed`**.

Disable / debug: `SANCTA_ROUTER_OFF=1`, `SANCTA_FORCE_GPT_LOCAL=1`. Tune thresholds: `config/trust_router.json`.

## Main chat memory (`operator_memory.py`)

`/api/chat` prepends **summarized or extractive** recall (redacted, instruction-like lines filtered)—not raw transcripts. **Research verbatim (redacted):** `SANCTA_MEMORY_RAW_MODE=1` with `SANCTA_TRUST_MODE=research`. Disable entirely: `SANCTA_OPERATOR_MEMORY=false`.

## RAG (`sancta_rag.py`)

- Default scoring: **TF-IDF** over chunk word counts (no extra ML deps).
- `format_rag_context` — long block for Ollama system prompts.
- `format_rag_inline` — compact `file: excerpt | …` for tight windows (e.g. char-GPT `generate_reply`).
- Legacy overlap: `SANCTA_RAG_LEXICAL=true`.

## Env quick reference

| Variable | Effect |
|----------|--------|
| `SANCTA_GPT_INCLUDE_LOGS` | Add log/JSONL lines to SanctaGPT corpus (default off). |
| `SANCTA_GPT_SHUFFLE_CORPUS` | Shuffle corpus after build (default off; breaks index curricula). |
| `SANCTA_GPT_LR` | Override learning rate. |
| `SANCTA_RAG_LEXICAL` | Use old lexical overlap instead of TF-IDF. |
| `SANCTA_RAG_TOP_K` | Max chunks per retrieval. |
| `SANCTA_TRUST_MODE` | `defense` (default) or `research` — see `docs/TRUST_AND_RESEARCH_MODE.md`. |
| `SANCTA_TRUST_TELEMETRY` | `false` disables `logs/trust_decisions.jsonl`. |
| `SANCTA_GPT_TRAIN_ON_CHAT` | `true` makes GPT-tab default to training on chat (still conversational-only). |
| `SANCTA_ROUTER_OFF` | Prefer conversational routing; strong knowledge still post-gated. |
| `SANCTA_FORCE_GPT_LOCAL` | Force conversational policy (debug). |
| `SANCTA_ALLOW_WEAK_KB_BLEND` | Research only: char-GPT fallback when knowledge backend fails. |
| `SANCTA_MEMORY_RAW_MODE` | Research only: verbatim redacted memory block. |
| `SANCTA_ROUTER_PERMISSIVE` | Research only: relax router margins. |
| `SANCTA_OPERATOR_MEMORY` | `false` disables operator memory file + injection on `/api/chat`. |

### `generate_reply` and KB

Default is **`use_retrieval=False`**: char-GPT should not reinterpret retrieved knowledge. Enable RAG-in-prompt only when you accept fidelity risk in a tiny LM.
