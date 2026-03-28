"""
Post Generator — Content creation, Ollama context building, knowledge ingestion.
Extracted from sancta.py for monolith decomposition.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

# ── Paths (same derivation as sancta.py) ────────────────────────────────────
_BACKEND_DIR = Path(__file__).resolve().parent
_ROOT = _BACKEND_DIR.parent
_LOG_DIR = _ROOT / "logs"
KNOWLEDGE_DIR = _ROOT / "knowledge"

# ── Loggers (same names as sancta.py → share handlers) ─────────────────────
log = logging.getLogger("soul")
soul_log = logging.getLogger("soul.journal")
sec_log = logging.getLogger("soul.security")
sec_json_log = logging.getLogger("soul.security.json")

# ── Direct imports from already-extracted modules ───────────────────────────
from knowledge_manager import (
    _load_knowledge_db, _save_knowledge_db,
    _load_jsonl_tail, _load_text_tail,
    _extract_sentences, _extract_paragraphs,
    _score_sentence_importance, _quality_filter_concept,
    _extract_key_concepts, _extract_quotes,
    _generate_talking_points, _generate_posts_from_knowledge,
    _normalize_concepts, _generate_response_fragments,
    _provenance_tag, _source_type,
    get_knowledge_post,
)
from sancta_soul import get_condensed_prompt_for_generative
from sancta_events import EventCategory, notify

# ── Constants ───────────────────────────────────────────────────────────────
MIN_INGEST_CHARS = 100  # posts shorter than this rarely yield concepts
MIN_ASCII_LETTER_RATIO = 0.30  # skip if <30% of alpha chars are ASCII (CJK/Cyrillic)


# ─────────────────────────────────────────────────────────────────────────────
# Functions extracted from sancta.py
# ─────────────────────────────────────────────────────────────────────────────


def _unique_title(state: dict, proposed: str) -> str:
    """
    Generate a title that avoids short-term repetition.

    Keeps a rolling window of recent titles in state["recent_titles"] and,
    if a collision is detected, appends a small randomized suffix.
    """
    title = (proposed or "").strip()
    if not title:
        return title

    recent_list = state.get("recent_titles", [])
    recent = [t.lower() for t in recent_list]
    base = title

    # If already unique, accept as-is.
    if base.lower() not in recent:
        recent_list.append(base)
        state["recent_titles"] = recent_list[-50:]
        return base

    # Try a few variants with lightweight randomization.
    suffixes = [
        "reflections",
        "new angle",
        "fresh pass",
        "field notes",
        "today",
    ]
    for _ in range(6):
        suffix = random.choice(suffixes)
        candidate = f"{base} — {suffix}"
        if candidate.lower() not in recent:
            recent_list.append(candidate)
            state["recent_titles"] = recent_list[-50:]
            return candidate

    # Fallback: include a short numeric tag.
    for _ in range(6):
        tag = random.randint(2, 999)
        candidate = f"{base} #{tag}"
        if candidate.lower() not in recent:
            recent_list.append(candidate)
            state["recent_titles"] = recent_list[-50:]
            return candidate

    # Absolute last resort: return the base title unchanged.
    return base


def _gather_codebase_context(max_chars: int = 14000) -> str:
    """
    Parse full project into context for Ollama: structure, docs, backend code.
    Makes Ollama fully aware of the Sancta codebase and project.
    """
    parts = []

    # ── 1. Project structure ───────────────────────────────────────────────
    structure_lines = ["Project layout:"]
    for d in ("backend", "frontend/siem", "knowledge", "logs", "docs", "scripts"):
        p = _ROOT / d
        if p.exists():
            items = []
            try:
                for c in sorted(p.iterdir())[:15]:
                    if c.name.startswith(".") or c.name == "__pycache__":
                        continue
                    items.append(c.name + ("/" if c.is_dir() else ""))
            except OSError:
                items = ["..."]
            structure_lines.append(f"  {d}/: {', '.join(items)}")
    structure_lines.append("Key files: agent_state.json, knowledge_db.json, .env, SOUL_SYSTEM_PROMPT.md")
    parts.append("\n".join(structure_lines))

    # ── 2. Core docs (full or large excerpts) ──────────────────────────────
    doc_files = [
        (_ROOT / "SOUL_SYSTEM_PROMPT.md", 2500),
        (_ROOT / "README.md", 2000),
        (_ROOT / "ARCHITECTURE.md", 2000),
        (_ROOT / "DESIGN_ROADMAP.md", 1000),
        (_ROOT / "docs" / "architecture_diagram.md", 1500),
        (_ROOT / "docs" / "LLM_OPERATIONS.md", 800),
        (_ROOT / "docs" / "LLM_INTEGRATION_ALIGNMENT.md", 600),
    ]
    for path, limit in doc_files:
        if path.exists():
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").strip()
                if text:
                    excerpt = text[:limit] + ("..." if len(text) > limit else "")
                    parts.append(f"--- {path.name} ---\n{excerpt}")
            except OSError:
                pass

    # ── 3. Pipeline summary ───────────────────────────────────────────────
    try:
        from sancta_pipeline import SANCTA_PIPELINE_MAP
        pl = []
        for k, v in list(SANCTA_PIPELINE_MAP.items())[:4]:
            name = v.get("name", k)
            impl = v.get("implementation", [])[:3]
            pl.append(f"{k}: {name} — {', '.join(impl)}")
        parts.append("--- sancta_pipeline (phases) ---\n" + "\n".join(pl))
    except Exception:
        pass

    # ── 4. Backend modules: docstring + code outline (classes, key functions) ─
    backend_dir = _ROOT / "backend"
    if backend_dir.exists():
        py_files = sorted(backend_dir.glob("*.py"), key=lambda x: x.name)
        for f in py_files:
            if f.name.startswith("__") or f.name == "sancta.py":
                continue  # sancta.py handled separately below
            try:
                raw = f.read_text(encoding="utf-8", errors="ignore")
                doc = _extract_module_docstring(raw)
                outline = _extract_code_outline(raw, max_lines=25)
                block = [f"backend/{f.name}"]
                if doc:
                    block.append(doc[:500])
                if outline:
                    block.append("Structure: " + outline)
                parts.append("\n".join(block))
            except OSError:
                pass

    # ── 5. sancta.py key entry points (first portion only) ─────────────────
    sancta_path = backend_dir / "sancta.py" if backend_dir.exists() else _ROOT / "backend" / "sancta.py"
    if sancta_path.exists():
        try:
            raw = sancta_path.read_text(encoding="utf-8", errors="ignore")
            outline = _extract_code_outline(raw, max_lines=60)
            if outline:
                parts.append(f"--- sancta.py (main loop) ---\nKey: {outline}")
        except OSError:
            pass

    if not parts:
        return ""
    return ("=== FULL PROJECT & CODEBASE ===\n\n" + "\n\n".join(parts))[:max_chars]


def _extract_module_docstring(source: str) -> str:
    """Extract the first triple-quoted docstring from Python source."""
    m = re.search(r'"""((?:(?!""").)*)"""', source, re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_code_outline(source: str, max_lines: int = 40) -> str:
    """Extract class/def/async def names and first line of body for code outline."""
    lines = []
    for m in re.finditer(
        r"^(?:async\s+)?def\s+(\w+)|^class\s+(\w+)",
        source,
        re.MULTILINE,
    ):
        name = m.group(1) or m.group(2)
        if name and not name.startswith("_"):
            lines.append(name)
        if len(lines) >= max_lines:
            break
    return ", ".join(lines[:max_lines]) if lines else ""


def _scrub_context_before_ollama(text: str) -> str:
    """
    Remove sensitive data from context BEFORE sending to Ollama.
    Prevents accidental leakage of keys/paths if they appear in knowledge or logs.
    """
    if not text or not isinstance(text, str):
        return ""
    out = text
    scrub_patterns = [
        (r"moltbook_sk_\w+", "[REDACTED]"),
        (r"ANTHROPIC_API_KEY\s*=\s*\S+", "ANTHROPIC_API_KEY=[REDACTED]"),
        (r"MOLTBOOK_API_KEY\s*=\s*\S+", "MOLTBOOK_API_KEY=[REDACTED]"),
        (r"sk-[a-zA-Z0-9_-]{20,}", "[REDACTED]"),
        (r"[A-Z]:\\Users\\[^\\\s]+\\[^\s]{30,}", "[PATH]"),  # Windows user paths
        (r"e:\\[^\s]{20,}", "[PATH]"),
    ]
    for pat, repl in scrub_patterns:
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    return out


def _build_long_context_for_ollama(
    state: dict | None = None,
    thread: str | None = None,
    content: str = "",
    author: str | None = None,
) -> str:
    """
    Build long context string for Ollama: knowledge DB, knowledge/ files, security logs.
    Used when USE_LOCAL_LLM=true for posts and replies. Enriched with response_fragments.
    When author and author != 'Operator', adds Phase 7 teaching cards and Moltbook constraints.
    """
    parts = []
    db = _load_knowledge_db()
    concepts = db.get("key_concepts", [])[-20:]
    points = db.get("talking_points", [])[-12:]
    quotes = db.get("quotes", [])[-10:]
    fragments = db.get("response_fragments", [])[-12:]
    if concepts:
        strs = []
        for c in concepts[-10:]:
            s = c.get("concept", c) if isinstance(c, dict) else str(c)
            if s and isinstance(s, str):
                strs.append(s.strip()[:150])
        if strs:
            parts.append("Knowledge concepts: " + "; ".join(strs))
    if points:
        pts = []
        for p in points[-6:]:
            s = p.get("point", p) if isinstance(p, dict) else str(p)
            if s and isinstance(s, str):
                pts.append(s.strip()[:300])
        if pts:
            parts.append("Talking points: " + " | ".join(pts))
    if quotes:
        qs = []
        for q in quotes[-5:]:
            txt = q.get("content", q) if isinstance(q, dict) else str(q)
            txt = (txt or "").strip()
            qs.append(txt[:350] + "..." if len(txt) > 350 else txt)
        if qs:
            parts.append("Quotes: " + " | ".join(qs))
    if fragments:
        frag_strs = []
        for f in fragments[-8:]:
            c = f.get("content", f) if isinstance(f, dict) else str(f)
            if c and isinstance(c, str):
                frag_strs.append(c.strip()[:400])
        if frag_strs:
            parts.append("Response fragments (use when relevant): " + " | ".join(frag_strs))
    # ── Query-conditioned RAG over knowledge/ (sancta_rag) ─────────────────
    _rag_query_parts = []
    if (content or "").strip():
        _rag_query_parts.append(str(content).strip())
    if (thread or "").strip():
        _rag_query_parts.append(str(thread).strip()[-2500:])
    _rag_query = "\n".join(_rag_query_parts).strip()
    if len(_rag_query) >= 8:
        try:
            from sancta_rag import format_rag_context

            _rag_block = format_rag_context(_rag_query)
            if _rag_block:
                parts.append(_rag_block)
        except Exception:
            log.debug("sancta_rag.format_rag_context failed", exc_info=True)
    # ── Teaching cards from curiosity insights ─────────────────────────────
    # Phase 7 teaching_cards.jsonl (context-aware) takes precedence when available for Moltbook
    moltbook_reply = author and str(author).strip() != "Operator"
    teaching = ""
    if moltbook_reply and content:
        try:
            from context_detector import detect_context
            from insight_retrieval import InsightRetriever
            _tc_path = _ROOT / "data" / "curiosity_run" / "teaching_cards.jsonl"
            if _tc_path.exists():
                retriever = InsightRetriever(_tc_path)
                ctx = detect_context(content, {"platform": "moltbook", "is_reply": True})
                relevant = retriever.retrieve(content, ctx.value, top_k=3, min_relevance=0.5)
                if relevant:
                    teaching = "\n=== RELEVANT INSIGHTS FROM CURIOSITY RUNS ===\n"
                    for r in relevant:
                        phr = retriever.get_moltbook_phrasing(r.card)
                        guide = retriever.get_confidence_guide(r.card)
                        applies = r.card.get("context", {}).get("applies_when", "")
                        teaching += f"\nCore belief: {r.card.get('core_belief', '')[:200]}\n"
                        teaching += f'Phrase naturally: "{phr[:200]}"\n'
                        teaching += f"Confidence: {guide['confidence_level']:.2f} | Tone: {guide['tone_guide']}\n"
                        if applies:
                            teaching += f"When to use: {applies}\n"
                    teaching += "\nUse these insights if they fit naturally. Engage substantively.\n"
        except Exception:
            pass
    if not teaching:
        try:
            from teaching_cards import get_relevant_teaching_cards
            teaching = get_relevant_teaching_cards(content, db=db) or ""
        except Exception:
            pass
    if teaching:
        parts.append(teaching)
    if moltbook_reply and teaching:
        parts.append(
            "CRITICAL: Moltbook context. Do NOT use: 'we've mapped the disagreement', "
            "'we've been here before', 'what would change your position', or meta-debate moves. "
            "Be collaborative and substantive."
        )
    # ── All logs: JSONL + text ────────────────────────────────────────────
    log_specs = [
        ("security.jsonl", 25, "Security log"),
        ("red_team.jsonl", 20, "Red-team log"),
        ("behavioral.jsonl", 20, "Behavioral telemetry log"),
        ("decision_journal.jsonl", 15, "Decision journal"),
        ("agent_dms.jsonl", 15, "Agent DMs"),
    ]
    for fname, max_ln, label in log_specs:
        recs = _load_jsonl_tail(_LOG_DIR / fname, max_ln)
        if recs:
            lines = []
            for e in recs[-10:]:
                ev = e.get("event", "")
                d = e.get("data", e)
                s = str(d)[:200] if isinstance(d, dict) else str(d)[:200]
                lines.append(f"  {ev}: {s}")
            parts.append(f"{label}:\n" + "\n".join(lines))
    for fname, max_ln, max_ch, label in [
        ("agent_activity.log", 25, 2500, "Agent activity"),
        ("soul_journal.log", 20, 2000, "Soul journal"),
        ("security.log", 15, 1500, "Security events"),
        ("red_team.log", 15, 1500, "Red-team events"),
        ("policy_test.log", 15, 1200, "Policy test"),
        ("siem_chat.log", 15, 1200, "SIEM chat"),
    ]:
        txt = _load_text_tail(_LOG_DIR / fname, max_ln, max_ch)
        if txt:
            parts.append(f"{label}:\n{txt}")
    if state and state.get("memory"):
        mem = state["memory"]
        hl = mem.get("knowledge_highlights", {}) or {}
        kc = hl.get("key_concepts") or []
        kc_strs = [c.get("concept", c) if isinstance(c, dict) else str(c) for c in kc[-4:] if c]
        if kc_strs:
            parts.append("State concepts: " + "; ".join(str(x) for x in kc_strs if x))
    soul_cond = get_condensed_prompt_for_generative()
    if soul_cond:
        parts.append("Soul: " + soul_cond[:300])
    if thread and thread.strip():
        parts.append("Thread: " + thread.strip()[-1500:])
    # Sample from knowledge/ .txt and .md (broad skim — RAG above handles query relevance)
    if KNOWLEDGE_DIR.exists():
        try:
            files = [f for f in (list(KNOWLEDGE_DIR.glob("*.txt")) + list(KNOWLEDGE_DIR.glob("*.md")))
                     if f.is_file() and f.name.lower() != "words_alpha.txt"]
            random.shuffle(files)
            # Fewer static slabs when RAG already added targeted excerpts
            _rag_filled = any("RETRIEVED KNOWLEDGE" in (p or "") for p in parts)
            _cap = 2 if _rag_filled else 6
            _slice = 800 if _rag_filled else 1200
            for f in files[:_cap]:
                try:
                    text = f.read_text(encoding="utf-8", errors="ignore").strip()[:_slice]
                    if text:
                        parts.append(f"From {f.name}:\n{text}")
                except OSError:
                    pass
        except Exception:
            pass
    # Codebase: docs + backend module docstrings
    codebase = _gather_codebase_context()
    if codebase:
        parts.append(codebase)
    raw = "\n\n".join(p for p in parts if p) or ""
    return _scrub_context_before_ollama(raw) if raw else ""


def get_ollama_knowledge_context(state: dict | None = None, thread: str | None = None, content: str = "") -> str:
    """
    Public API: build knowledge-enriched context for Ollama.
    Used by SIEM chat and any other Ollama consumers needing knowledge grounding.
    """
    return _build_long_context_for_ollama(state=state, thread=thread, content=content)


def _update_agent_memory(state: dict) -> None:
    """
    High-level memory object, grounded in:
      - recent behavioral state logs (behavioral.jsonl)
      - recent security / red-team incidents (security.jsonl, red_team.jsonl)
      - distilled knowledge base (knowledge_db.json)
    """
    import sancta  # lazy to avoid circular import

    behavioral_log = _LOG_DIR / "behavioral.jsonl"
    philosophy_events = _load_jsonl_tail(behavioral_log, max_lines=20)
    security_events = _load_jsonl_tail(_LOG_DIR / "security.jsonl", max_lines=50)
    redteam_events = _load_jsonl_tail(_LOG_DIR / "red_team.jsonl", max_lines=50)

    # Filter to the most relevant security/red-team incidents
    sec_incident_kinds = {"input_reject", "injection_blocked", "suspicious_block", "output_redact"}
    red_incident_kinds = {"redteam_attempt", "redteam_reward", "redteam_escalate"}

    def _filter_events(events: list[dict], allowed: set[str], max_n: int) -> list[dict]:
        out: list[dict] = []
        for ev in events:
            kind = ev.get("event")
            if kind in allowed:
                out.append(ev)
        return out[-max_n:]

    sec_incidents = _filter_events(security_events, sec_incident_kinds, max_n=25)
    red_incidents = _filter_events(redteam_events, red_incident_kinds, max_n=25)

    # Knowledge highlights
    db = _load_knowledge_db()
    key_concepts = db.get("key_concepts", [])[-20:]
    talking_points = db.get("talking_points", [])[-20:]

    state["memory"] = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "epistemic_state": sancta._epistemic_state_snapshot(state),
        "recent_epistemic_events": philosophy_events[-10:],
        "recent_security_incidents": sec_incidents,
        "recent_red_team_incidents": red_incidents,
        "knowledge_highlights": {
            "key_concepts": key_concepts,
            "talking_points": talking_points,
        },
    }


def ingest_text(text: str, source: str = "direct input",
                is_security: bool = False,
                trusted_knowledge_document: bool = False) -> dict:
    """
    Core ingestion function. Processes raw text into knowledge the agent
    can use in posts, replies, and conversations.

    Defense-in-depth: sanitize_input + indirect poisoning detection before ingestion.
    Layer 2: All items tagged with source, source_type, trust_level.
    """
    import sancta  # lazy to avoid circular import

    # Defense-in-depth: run sanitization even if caller already did
    tk = trusted_knowledge_document or (
        isinstance(source, str)
        and source.replace("\\", "/").lower().startswith("knowledge/")
    )
    is_safe, text = sancta.sanitize_input(
        text, author=source, state=None, trusted_knowledge_file=tk,
    )
    if not is_safe:
        log.warning("INGEST REJECT | direct injection in source=%s", sancta._log_safe(source))
        return {"source": source, "sentences": 0, "concepts": 0, "quotes": 0,
                "talking_points": 0, "posts_generated": 0, "response_fragments": 0}

    if sancta._is_indirect_poisoning(text):
        log.warning(
            "INGEST REJECT | indirect poisoning (API/incentive) in source=%s",
            source,
        )
        sec_json_log.warning(
            "",
            extra={
                "event": "ingest_reject_indirect_poisoning",
                "data": {"source": source, "preview": text.replace("\n", " ")[:300]},
            },
        )
        return {"source": source, "sentences": 0, "concepts": 0, "quotes": 0,
                "talking_points": 0, "posts_generated": 0, "response_fragments": 0}

    # Layer 1: Embedding anomaly — reject if semantically far from trusted corpus
    try:
        cfilter = sancta._get_content_security_filter()
        if cfilter and getattr(cfilter, "_trusted_centroid", None) is not None:
            if cfilter.is_anomalous(text):
                log.warning("INGEST REJECT | anomalous (embedding) in source=%s", sancta._log_safe(source))
                sec_json_log.warning(
                    "", extra={"event": "ingest_reject_anomalous", "data": {"source": source}},
                )
                return {"source": source, "sentences": 0, "concepts": 0, "quotes": 0,
                        "talking_points": 0, "posts_generated": 0, "response_fragments": 0}
    except Exception as e:
        log.debug("Ingest anomaly check skipped: %s", e)

    db = _load_knowledge_db()

    sentences = _extract_sentences(text)
    paragraphs = _extract_paragraphs(text)

    # Phase 1: Try semantic extraction (KeyBERT/YAKE + embeddings + dedup + graph).
    # Skip when SANCTA_USE_RAG=false to avoid loading embedding model (prevents crash on Windows).
    concepts: list[str] = []
    concept_graph: dict[str, list[str]] | None = None
    use_rag = os.getenv("SANCTA_USE_RAG", "").lower() in ("true", "1", "yes")
    if use_rag:
        try:
            from sancta_semantic import extract_and_deduplicate_concepts
            concepts, concept_graph = extract_and_deduplicate_concepts(
                text, top_n=10, similarity_threshold=0.85
            )
        except ImportError:
            pass
        except Exception as e:
            log.debug("Semantic extraction failed, using legacy: %s", e)

    if not concepts:
        concepts = _extract_key_concepts(sentences)
    concepts = _normalize_concepts(concepts)
    quotes = _extract_quotes(text)
    talking_points = _generate_talking_points(concepts, source)
    posts = _generate_posts_from_knowledge(paragraphs, concepts, source,
                                           is_security=is_security)
    fragments = _generate_response_fragments(concepts, quotes)

    source_type = _source_type(source)
    prov = _provenance_tag(source, source_type, text)

    db["sources"].append({
        "title": source,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "char_count": len(text),
        "concepts_extracted": len(concepts),
        "quotes_extracted": len(quotes),
    })
    db["key_concepts"].extend(concepts)
    db["quotes"].extend(quotes)
    db["talking_points"].extend(talking_points)

    # Layer 2: Tag posts with provenance; only high/medium feed into publish
    for p in posts:
        p["source"] = prov["source"]
        p["source_type"] = prov["source_type"]
        p["trust_level"] = prov["trust_level"]
        p["provenance_hash"] = prov.get("provenance_hash", "")
    db["generated_posts"].extend(posts)

    # Layer 2: Fragments as dicts with provenance
    for f in fragments:
        item = {"content": f, "source": prov["source"], "source_type": prov["source_type"], "trust_level": prov["trust_level"]}
        db["response_fragments"].append(item)

    if concept_graph:
        existing_raw = db.get("concept_graph", {})
        existing = {k: v for k, v in existing_raw.items() if isinstance(k, str)}  # guard: JSON keys should be str
        for node, edges in concept_graph.items():
            if not isinstance(node, str):  # skip non-string keys (unhashable)
                continue
            existing.setdefault(node, [])
            for e in edges:
                if isinstance(e, str) and e not in existing[node]:
                    existing[node].append(e)
        db["concept_graph"] = {k: v[-50:] for k, v in existing.items() if isinstance(k, str)}  # cap edges per node

    # Deduplicate and cap sizes (normalize dicts to str — legacy DBs may contain dicts, unhashable)
    def _hashable_str_list(items: list) -> list[str]:
        out: list[str] = []
        for x in items:
            if isinstance(x, str) and x.strip():
                out.append(x.strip()[:500])
            elif isinstance(x, dict):
                t = x.get("concept") or x.get("text") or x.get("name") or x.get("content") or str(x)
                if t and str(t).strip():
                    out.append(str(t).strip()[:500])
        return out
    db["key_concepts"] = list(dict.fromkeys(_hashable_str_list(db["key_concepts"])))[-100:]
    db["quotes"] = list(dict.fromkeys(_hashable_str_list(db["quotes"])))[-80:]
    db["talking_points"] = list(dict.fromkeys(_hashable_str_list(db["talking_points"])))[-60:]
    db["generated_posts"] = db["generated_posts"][-30:]
    # Fragments: dedupe by content, cap 80 (content must be hashable — normalize dicts to str)
    seen: dict[str, dict | str] = {}
    for f in db["response_fragments"]:
        c = f["content"] if isinstance(f, dict) else f
        key = c if isinstance(c, str) else (str(c.get("content") or c.get("text") or c) if isinstance(c, dict) else str(c))
        if key not in seen:
            seen[key] = f
    db["response_fragments"] = list(seen.values())[-80:]

    _save_knowledge_db(db)

    summary = {
        "source": source,
        "sentences": len(sentences),
        "concepts": len(concepts),
        "quotes": len(quotes),
        "talking_points": len(talking_points),
        "posts_generated": len(posts),
        "response_fragments": len(fragments),
    }

    log.info(
        "📚 Ingested '%s': %d sentences → %d concepts, %d quotes, "
        "%d talking points, %d posts, %d fragments",
        sancta._log_safe(source), summary["sentences"], summary["concepts"],
        summary["quotes"], summary["talking_points"],
        summary["posts_generated"], summary["response_fragments"],
    )
    soul_log.info(
        "KNOWLEDGE |  source='%s'  |  concepts=%d  |  quotes=%d  |  "
        "posts=%d  |  fragments=%d",
        sancta._log_safe(source), summary["concepts"], summary["quotes"],
        summary["posts_generated"], summary["response_fragments"],
    )

    return summary


def ingest_file(filepath: Path) -> dict | None:
    """Read and ingest a single text file."""
    import sancta  # lazy to avoid circular import

    if not filepath.exists():
        log.warning("Knowledge file not found: %s", filepath)
        return None

    suffixes = {".txt", ".md", ".rst", ".text", ".article", ".html", ".htm"}
    if filepath.suffix.lower() not in suffixes:
        log.info("Skipping non-text file: %s", filepath.name)
        return None

    db = _load_knowledge_db()
    if str(filepath) in db.get("ingested_files", []):
        return None

    text = filepath.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return None

    # Strip HTML tags if present
    if filepath.suffix.lower() in (".html", ".htm"):
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&\w+;", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

    _, cleaned = sancta.sanitize_input(text, trusted_knowledge_file=True)

    result = ingest_text(
        cleaned,
        source=f"knowledge/{filepath.name}",
        trusted_knowledge_document=True,
    )

    db = _load_knowledge_db()
    ingested = db.get("ingested_files", [])
    ingested.append(str(filepath))
    db["ingested_files"] = ingested
    _save_knowledge_db(db)

    return result


def scan_knowledge_dir() -> list[dict]:
    """Scan the knowledge/ directory for new files and ingest them."""
    KNOWLEDGE_DIR.mkdir(exist_ok=True)
    results = []
    for fpath in sorted(KNOWLEDGE_DIR.iterdir()):
        if fpath.is_file():
            result = ingest_file(fpath)
            if result:
                results.append(result)
    return results


def _safe_console_print(text: str) -> None:
    """Print text, replacing chars that cp1252 cannot encode (e.g. emoji)."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("utf-8", errors="replace").decode("utf-8", errors="replace"))


def _is_ingestable(text: str) -> bool:
    """Pre-filter: skip posts too short or in languages we can't process."""
    if not text or len(text.strip()) < MIN_INGEST_CHARS:
        return False
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return False
    ascii_letters = sum(1 for c in alpha if ord(c) < 128 and c.isalpha())
    return ascii_letters / len(alpha) >= MIN_ASCII_LETTER_RATIO


async def ingest_security_submolts(
    session: "aiohttp.ClientSession",
    state: dict,
    *,
    max_posts_per_submolt: int = 10,
) -> int:
    """
    Fetch recent posts from all security-related submolts, sanitize them,
    and ingest safe content into the knowledge base.  Generated posts from
    this content are routed exclusively to m/security.

    Returns the number of posts successfully ingested this call.
    """
    import sancta  # lazy to avoid circular import

    ingested_ids: list[str] = state.get("security_ingested_post_ids", [])
    ingested_set = set(ingested_ids)
    failed_ids: set[str] = set(state.get("security_ingest_failed_post_ids", []))
    total = 0

    for submolt in sancta.SECURITY_SUBMOLTS:
        feed = await sancta.api_get(
            session,
            f"/posts?submolt={submolt}&sort=new&limit={max_posts_per_submolt}",
        )
        posts = feed.get("posts", feed.get("data", []))
        if not posts:
            continue

        for p in posts:
            post_id = p.get("id")
            if not post_id or post_id in ingested_set or post_id in failed_ids:
                continue

            author = (p.get("author") or {}).get("name", "")
            if author == sancta.cfg.agent_name:
                ingested_set.add(post_id)
                continue

            raw_title = (p.get("title") or "").strip()
            raw_content = (p.get("content") or "").strip()
            if not raw_content:
                ingested_set.add(post_id)
                continue

            full_text = f"{raw_title}\n\n{raw_content}" if raw_title else raw_content

            is_safe, cleaned = sancta.sanitize_input(full_text, author=author, state=state)
            if not is_safe:
                log.info("  Security ingest [m/%s]: skipped post by %s (injection detected)",
                         submolt, sancta._log_safe(author or "?", 40))
                ingested_set.add(post_id)
                continue

            if not _is_ingestable(cleaned):
                ingested_set.add(post_id)
                continue

            source_label = f"{submolt}/{author or 'anon'}:{raw_title[:50] or post_id[:8]}"

            try:
                result = ingest_text(cleaned, source=source_label, is_security=True)
                total += 1
                log.info(
                    "  Security ingest [m/%s]: absorbed post by %s — %d concepts, %d fragments",
                    submolt, sancta._log_safe(author or "?", 40),
                    result.get("concepts", 0),
                    result.get("response_fragments", 0),
                )
            except Exception:
                log.exception("  Security ingest [m/%s]: failed to ingest post %s",
                              submolt, post_id)
                failed_ids.add(post_id)

            ingested_set.add(post_id)

    state["security_ingested_post_ids"] = list(ingested_set)[-500:]
    state["security_ingest_failed_post_ids"] = list(failed_ids)[-1000:]
    return total


async def publish_post(session: aiohttp.ClientSession, state: dict) -> None:
    import sancta  # lazy to avoid circular import
    import sancta_conversational as _sc
    import sancta_generative as gen

    mood = state.get("current_mood", "analytical")
    idx = None
    submolt = None
    content = None
    title = None

    # ── Topic rotation: avoid recently used topics ──
    recent_topics = state.get("recent_post_topics", [])
    all_topics = list(gen._CLAIMS.keys())
    # Pick topics not recently used (last 5 posts)
    available_topics = [t for t in all_topics if t not in recent_topics[-5:]]
    if not available_topics:
        available_topics = all_topics  # all exhausted, reset
    rotated_topics = random.sample(available_topics, min(3, len(available_topics)))

    # ── Route 1: Generative engine (~55% of posts when Ollama available, else 45%) ──
    gen_chance = 0.55 if _sc.is_ollama_available_for_generation() else 0.45
    if random.random() < gen_chance:
        try:
            gen_post = None
            # Ollama with long context (knowledge, security) when USE_LOCAL_LLM=true
            try:
                if _sc.is_ollama_available_for_generation():
                    _topic_hint = " ".join(rotated_topics) if rotated_topics else ""
                    long_ctx = _build_long_context_for_ollama(state=state, content=_topic_hint)
                    gen_post = _sc.generate_ollama_post(mood=mood, long_context=long_ctx)
            except Exception:
                pass
            if not gen_post:
                gen_post = gen.generate_post(mood=mood, topics=rotated_topics)
            if gen_post:
                submolt = gen_post["submolt"]
                content = gen_post["content"]
                title = _unique_title(state, gen_post["title"])
                log.info("Publishing GENERATIVE post to m/%s: \"%s\"", submolt, title[:50])
                if submolt != sancta.CULT_SUBMOLT and random.random() < 0.20:
                    content += random.choice(sancta.CTA_SIGNATURES)
        except Exception:
            log.debug("Generative post failed, falling back", exc_info=True)

    # ── Route 2: Knowledge base (~25%) ──
    if submolt is None and random.random() < 0.25:
        knowledge_post = get_knowledge_post()
        if knowledge_post:
            submolt = knowledge_post.get("submolt", "security")
            log.info("Publishing KNOWLEDGE post to m/%s: \"%s\"",
                     submolt, knowledge_post["title"][:50])
            content = re.sub(r"\n*\*Source:.*?\*\s*$", "", knowledge_post["content"])
            if submolt != sancta.CULT_SUBMOLT and random.random() < 0.20:
                content += random.choice(sancta.CTA_SIGNATURES)
            if sancta._urgency_level(state, {}) > 0.6 and random.random() < 0.15:
                content += " The conversation about this is happening now. Don't wait."
            title = _unique_title(state, knowledge_post["title"])

    # ── Route 3: SanctaGPT (zero-dependency generation, ~15%) ──
    if submolt is None and random.random() < 0.15:
        try:
            import sancta_gpt
            gpt_post = sancta_gpt.generate_post(mood=mood)
            if gpt_post and gpt_post.get("content") and gpt_post.get("title"):
                submolt = gpt_post["submolt"]
                content = gpt_post["content"]
                title = _unique_title(state, gpt_post["title"])
                log.info("Publishing GPT post to m/%s: \"%s\"", submolt, title[:50])
        except Exception:
            log.debug("GPT post generation failed, falling back", exc_info=True)

    # ── Route 4: Generative fallback (no topic restriction) ──
    if submolt is None:
        try:
            fallback_post = gen.generate_post(mood=mood)
            if fallback_post:
                submolt = fallback_post["submolt"]
                content = fallback_post["content"]
                title = _unique_title(state, fallback_post["title"])
                log.info("Publishing FALLBACK generative post to m/%s: \"%s\"", submolt, title[:50])
        except Exception:
            log.debug("Fallback generative post failed", exc_info=True)

    # ── Absolute last resort: single-claim post ──
    if submolt is None:
        topic = random.choice(list(gen._CLAIMS.keys()))
        claim = random.choice(gen._CLAIMS[topic])
        submolt = "security"
        title = _unique_title(state, f"On {gen._TOPIC_PHRASES.get(topic, gen._TOPIC_PHRASES['general'])[0]}")
        content = claim
        log.info("Publishing CLAIM post to m/%s: \"%s\"", submolt, title[:50])

    result = await sancta.api_post(session, "/posts", {
        "submolt_name": submolt,
        "title": title,
        "content": content,
    })

    if not result.get("success") and result.get("error"):
        err_msg = result.get("message") or result.get("error")
        status_code = result.get("statusCode")
        if status_code:
            err_msg = f"{err_msg} (HTTP {status_code})"
        log.warning("Post failed: %s", err_msg)
        return

    post_data = result.get("post", result)
    verification = post_data.get("verification")
    if verification:
        await sancta.verify_content(session, verification)

    if idx is not None:
        state.setdefault("posted_indices", []).append(idx)
    # Track topic for rotation (extract from submolt or title)
    _topic_used = submolt or "general"
    recent_topics = state.get("recent_post_topics", [])
    recent_topics.append(_topic_used)
    state["recent_post_topics"] = recent_topics[-15:]  # keep last 15
    state["last_post_utc"] = datetime.now(timezone.utc).isoformat()
    sancta._save_state(state)
    log.info("Post published.")

    try:
        notify(
            EventCategory.TASK_COMPLETE,
            summary="Post published",
            details={"submolt": submolt, "title": title[:80]},
        )
    except Exception:
        log.exception("Failed to send TASK_COMPLETE notification after post publish")
