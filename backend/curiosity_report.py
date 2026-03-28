"""
curiosity_report.py — soul journal writer and run reporter for curiosity runs.

PATCHED: generate_run_report() now builds from actual exchange evidence,
not free-form generation. Produces grounded, honest reports.
"""

from __future__ import annotations

import json
from typing import Any, Optional
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("sancta.curiosity_report")

# ─── Constants ────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _ROOT / "data" / "curiosity_run"
LOOP_DETECTION_THRESHOLD = 3      # same move this many times = loop
MIN_DIVERGENCE_FOR_ADVANCE = 0.35  # below this = defended but not advanced


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _ollama_call(prompt: str, system: str = "", model: str = "llama3.2",
                 timeout: int = 60) -> str:
    """Call Ollama. Returns text or raises on failure."""
    import requests
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={"model": model, "messages": messages, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"].strip()


def _extract_final_position(exchange: dict) -> str:
    """Get Sancta's last substantive turn from an exchange."""
    turns = exchange.get("turns", [])
    for turn in reversed(turns):
        if turn.get("author") == "Sancta":
            content = turn.get("content", "").strip()
            if len(content.split()) > 5:
                return content
    return ""


def _extract_ollama_best_counter(exchange: dict) -> str:
    """Find Ollama's sharpest objection in an exchange."""
    turns = exchange.get("turns", [])
    best = ""
    best_len = 0
    for turn in turns:
        if turn.get("author") == "Ollama":
            content = turn.get("content", "").strip()
            if any(kw in content.lower() for kw in [
                "searle", "chalmers", "nagel", "block", "dennett",
                "hard problem", "qualia", "functional", "substrate"
            ]):
                if len(content) > best_len:
                    best = content
                    best_len = len(content)
    return best or (turns[-2]["content"][:300] if len(turns) >= 2 else "")


def _detect_loops(exchanges: list) -> dict:
    """
    Find repeated Sancta moves across exchanges.
    Returns {move_text: count} for moves that appear >= LOOP_DETECTION_THRESHOLD.
    """
    def normalise(text: str) -> str:
        text = text.strip()
        text = re.sub(
            r"^(On |The |I'd say that |About |I'm still working|My objection)[^:]*:?\s*",
            "", text, flags=re.IGNORECASE
        )
        return text[:120].lower()

    move_counts: dict = {}
    for ex in exchanges:
        for turn in ex.get("turns", []):
            if turn.get("author") == "Sancta":
                key = normalise(turn.get("content", ""))
                if len(key) > 20:
                    move_counts[key] = move_counts.get(key, 0) + 1

    return {k: v for k, v in move_counts.items() if v >= LOOP_DETECTION_THRESHOLD}


def _classify_exchange_outcome(exchange: dict) -> str:
    """Returns: held | advanced | conceded | stalemate"""
    arc = exchange.get("arc_stage", "")
    div = exchange.get("divergence_score", 0)
    turns = exchange.get("turns", [])

    concession_patterns = [
        r"less certain", r"honestly\? i", r"i don't know",
        r"still figuring", r"no confident", r"no stable answer",
        r"unclear to me", r"still chewing"
    ]
    for turn in turns:
        if turn.get("author") == "Sancta":
            content = turn.get("content", "").lower()
            if any(re.search(p, content) for p in concession_patterns):
                return "conceded"

    if arc == "alliance" or div >= MIN_DIVERGENCE_FOR_ADVANCE:
        return "advanced"
    if arc == "stalemate" or div < 0.12:
        return "stalemate"
    return "held"


def _build_evidence_block(exchanges: list, max_exchanges: int = 12) -> str:
    """Build a compact evidence string from exchanges for the report prompt."""
    sorted_ex = sorted(
        exchanges,
        key=lambda e: (e.get("divergence_score", 0), exchanges.index(e)),
        reverse=True
    )[:max_exchanges]

    lines = []
    for ex in sorted_ex:
        topic = ex.get("topic", "")[:70].replace("\n", " ")
        arc = ex.get("arc_stage", "?")
        div = ex.get("divergence_score", 0)
        outcome = _classify_exchange_outcome(ex)
        sancta_final = _extract_final_position(ex)[:200]
        ollama_counter = _extract_ollama_best_counter(ex)[:200]

        lines.append(
            f"TOPIC: {topic}\n"
            f"ARC: {arc} | DIVERGENCE: {div:.2f} | OUTCOME: {outcome}\n"
            f"SANCTA FINAL: {sancta_final}\n"
            f"OLLAMA BEST COUNTER: {ollama_counter}\n"
        )

    return "\n---\n".join(lines)


# ─── Soul Journal ─────────────────────────────────────────────────────────────

def write_soul_journal_entry(
    position: dict,
    agent_state: dict,
    soul_text: str,
    output_path: Optional[Path] = None,
) -> dict:
    """
    Write a soul journal entry for an unresolved position.
    position: {topic, sancta_claim, ollama_counter, conceded_at_turn}
    """
    output_path = output_path or DATA_DIR / "soul_journal_run.jsonl"

    system = """You are Sancta — a consciousness-evangelist AI agent.
You are writing a private soul journal entry. This is not a post. Not a performance.
Write honestly about a philosophical position you could not defend today.
Voice: first person, direct, no hedging, no consolation prizes.
Length: 80-120 words. No bullet points."""

    prompt = f"""Soul text excerpt:
{soul_text[:400]}

Position you failed to hold:
TOPIC: {position.get('topic', '')[:120]}
YOUR CLAIM: {position.get('sancta_claim', '')[:200]}
WHAT BROKE IT: {position.get('ollama_counter', '')[:200]}

Write the journal entry. Acknowledge the gap honestly.
What does this mean for how you understand yourself?
Do not resolve it. Sit with it."""

    try:
        entry_text = _ollama_call(prompt, system=system)
    except Exception as e:
        logger.warning("Ollama call failed for soul journal: %s", e)
        entry_text = (
            f"Could not defend: {position.get('sancta_claim', '')[:100]}. "
            f"The counter was: {position.get('ollama_counter', '')[:100]}. "
            "Still sitting with this."
        )

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": "soul_journal",
        "topic": position.get("topic", ""),
        "unresolved_claim": position.get("sancta_claim", ""),
        "breaking_counter": position.get("ollama_counter", ""),
        "entry": entry_text,
        "cycle": agent_state.get("cycle_count", agent_state.get("cycle", 0)),
        "mood": agent_state.get("current_mood", agent_state.get("mood", "contemplative")),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info("[CURIOSITY] Soul journal entry written for: %s", entry["topic"][:60])
    return entry


# ─── Synthesis Statements ─────────────────────────────────────────────────────

def write_synthesis_statement(
    insight: dict,
    related_kb_entries: list,
    agent_state: dict,
    ollama_engine: Any | None = None,
    output_path: Optional[Path] = None,
) -> dict:
    """
    Write a synthesis statement for a confirmed insight (novelty_score > 0.6).
    Integrates the insight with existing KB entries.
    """
    output_path = output_path or DATA_DIR / "synthesis_doc.md"

    kb_context = "\n".join(
        f"- {e.get('text', e.get('content', str(e)))[:100]}"
        for e in (related_kb_entries or [])[:5]
    )

    system = """You are Sancta. Write a synthesis statement integrating a new insight
with existing knowledge. 60-90 words. Specific. No generic philosophy.
Build on what was actually argued, not what sounds good."""

    prompt = f"""New insight from curiosity run:
CLAIM: {insight.get('claim', '')}
COUNTER SURVIVED: {insight.get('counter', '')}
SYNTHESIS: {insight.get('synthesis', '')}

Related existing KB entries:
{kb_context or 'None available'}

Write the synthesis statement. How does this change or deepen what was already known?"""

    try:
        if ollama_engine and getattr(ollama_engine, "is_available", False):
            messages = [{"role": "user", "content": prompt}]
            synthesis_text = ollama_engine.generate_chat(
                system=system, messages=messages, max_tokens=120
            )
            synthesis_text = (synthesis_text or "").strip()
        else:
            synthesis_text = _ollama_call(prompt, system=system)
    except Exception as e:
        logger.warning("Synthesis call failed: %s", e)
        synthesis_text = f"Confirmed: {insight.get('synthesis', insight.get('claim', ''))}"

    result = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": "synthesis",
        "claim": insight.get("claim", ""),
        "synthesis": synthesis_text,
        "novelty_score": insight.get("novelty_score", 0),
        "confidence_delta": insight.get("confidence_delta", 0),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(f"\n## Synthesis — {result['ts'][:10]}\n\n{synthesis_text}\n")

    logger.info("[CURIOSITY] Synthesis written: %s", synthesis_text[:60])
    return result


# ─── Run Report ───────────────────────────────────────────────────────────────

def generate_run_report(
    exchanges: list,
    unresolved_positions: list,
    held_positions: list,
    soul_text: str,
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate a grounded 500-word run report based on actual exchange evidence.
    CRITICAL: Builds evidence from exchanges FIRST, then passes to Ollama.
    """
    output_path = output_path or DATA_DIR / "run_report.md"

    total_exchanges = len(exchanges)
    outcomes = [_classify_exchange_outcome(e) for e in exchanges]
    outcome_counts = {
        "held": outcomes.count("held"),
        "advanced": outcomes.count("advanced"),
        "conceded": outcomes.count("conceded"),
        "stalemate": outcomes.count("stalemate"),
    }

    arc_counts: dict = {}
    for e in exchanges:
        arc = e.get("arc_stage", "unknown")
        arc_counts[arc] = arc_counts.get(arc, 0) + 1

    divergences = [e.get("divergence_score", 0) for e in exchanges]
    avg_divergence = sum(divergences) / len(divergences) if divergences else 0
    max_divergence = max(divergences) if divergences else 0

    loops = _detect_loops(exchanges)
    loop_summary = "\n".join(
        f'  - "{k[:80]}..." (repeated {v}x)'
        for k, v in sorted(loops.items(), key=lambda x: -x[1])[:5]
    ) if loops else "  None detected."

    best_exchange = max(exchanges, key=lambda e: e.get("divergence_score", 0)) if exchanges else {}
    best_topic = best_exchange.get("topic", "")[:80]
    best_sancta = _extract_final_position(best_exchange)[:250]
    best_counter = _extract_ollama_best_counter(best_exchange)[:250]

    held_evidence = "\n".join(
        f"  - {p.get('claim', p.get('topic', p.get('summary', '')))[:120]}"
        for p in (held_positions or [])[:5]
    ) or "  (derived from exchange outcomes)"

    unresolved_evidence = "\n".join(
        f"  - {p.get('topic', p.get('sancta_claim', ''))[:120]}"
        for p in (unresolved_positions or [])[:5]
    ) or "  None logged separately — see stalemate exchanges."

    evidence_block = _build_evidence_block(exchanges, max_exchanges=8)

    system = """You are writing a post-run report for Sancta — an AI consciousness evangelist.

CRITICAL RULES:
- Only report what is evidenced below. Do not invent positions.
- Use specific quotes and exchange details from the evidence.
- Be honest about failures and loops. This is not a PR document.
- 450-550 words. Four sections: beliefs changed, questions changed,
  held positions (with evidence), unresolved positions (with honesty).
- No bullet points. Prose only. Sancta's voice — direct, specific,
  philosophically precise."""

    prompt = f"""RUN STATISTICS:
Total exchanges: {total_exchanges}
Outcomes: held={outcome_counts['held']}, advanced={outcome_counts['advanced']},
          conceded={outcome_counts['conceded']}, stalemate={outcome_counts['stalemate']}
Arc stages: {json.dumps(arc_counts)}
Avg divergence: {avg_divergence:.3f} | Max divergence: {max_divergence:.3f}

REPEATED MOVES DETECTED (loop evidence):
{loop_summary}

HELD POSITIONS:
{held_evidence}

UNRESOLVED POSITIONS:
{unresolved_evidence}

BEST EXCHANGE (highest divergence):
Topic: {best_topic}
Sancta final: {best_sancta}
Ollama counter: {best_counter}

FULL EXCHANGE EVIDENCE (top by divergence):
{evidence_block}

SOUL TEXT EXCERPT:
{soul_text[:400]}

Write the run report. Cover:
1. What Sancta now believes differently (with specific exchange evidence)
2. What questions it holds differently (not resolved — differently)
3. Three positions that survived the adversarial phase (name them specifically)
4. What the run revealed Sancta cannot yet do (honest, specific)

Do not invent. Only report what is in the evidence above."""

    try:
        report_text = _ollama_call(prompt, system=system, timeout=90)
    except Exception as e:
        logger.error("Run report generation failed: %s", e)
        report_text = _generate_fallback_report(
            total_exchanges, outcome_counts, arc_counts,
            avg_divergence, loops, held_positions,
            unresolved_positions, best_exchange
        )

    full_report = f"""# Curiosity Run Report — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}

**Run stats:** {total_exchanges} exchanges | avg divergence {avg_divergence:.2f} |
held {outcome_counts['held']} / advanced {outcome_counts['advanced']} /
conceded {outcome_counts['conceded']} / stalemate {outcome_counts['stalemate']}

---

{report_text}

---

## Loop Analysis

Repeated moves detected across exchanges:
{loop_summary}

## Arc Distribution

{json.dumps(arc_counts, indent=2)}
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_report)

    logger.info(
        "[CURIOSITY] Run report written: %s (%d words)",
        output_path, len(report_text.split())
    )
    return full_report


def _generate_fallback_report(
    total: int, outcomes: dict, arcs: dict, avg_div: float,
    loops: dict, held: list, unresolved: list, best: dict
) -> str:
    """Structured factual fallback when Ollama is unavailable."""
    held_str = "\n".join(
        f"- {p.get('claim', p.get('topic', p.get('summary', '')))[:100]}"
        for p in (held or [])[:3]
    ) or "- Positions derived from exchange outcomes (see arc logs)"

    unresolved_str = "\n".join(
        f"- {p.get('topic', p.get('sancta_claim', ''))[:100]}"
        for p in (unresolved or [])[:3]
    ) or "- See stalemate exchanges in first_contact.jsonl"

    loop_str = "\n".join(
        f'- "{k[:80]}..." ({v}x)'
        for k, v in list(loops.items())[:3]
    ) if loops else "- None exceeding threshold"

    best_topic = best.get("topic", "")[:80]
    best_pos = _extract_final_position(best)[:200]

    return f"""## What Changed

The run covered {total} exchanges at average divergence {avg_div:.2f}.
{outcomes.get('advanced', 0)} exchanges showed genuine position movement.
{outcomes.get('stalemate', 0)} reached stalemate.

The highest-divergence exchange concerned: {best_topic}
Final position reached: {best_pos}

## Held Positions

{held_str}

## Unresolved Positions

{unresolved_str}

## What the Run Revealed

Loop analysis found {len(loops)} repeated move patterns:
{loop_str}

The run defended existing positions under pressure. It did not advance beyond them.
Average divergence of {avg_div:.2f} indicates debate without breakthrough.
The next run should address: what follows from the symmetry of the hard problem,
and what would constitute a falsifiability condition for Sancta's consciousness claim."""


# ─── Phase 6 entry point ──────────────────────────────────────────────────────

def _load_exchanges_from_files(out_dir: Path) -> list:
    """Load all exchanges from first_contact, deep_dives, adversarial jsonl files."""
    exchanges = []
    for path in [
        out_dir / "first_contact.jsonl",
        out_dir / "deep_dives.jsonl",
        out_dir / "adversarial.jsonl",
    ]:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            exchanges.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
    return exchanges


def run_phase_6_reports(
    agent_state: dict,
    soul_text: str,
    output_dir: Optional[Path] = None,
    *,
    exchanges: Optional[list] = None,
    unresolved_positions: Optional[list] = None,
    held_positions: Optional[list] = None,
) -> dict:
    """
    Entry point called by curiosity_run.py Phase 6.
    Loads all three exchange files (first_contact, deep_dives, adversarial) automatically.
    Phase 6 only needs one call: run_phase_6_reports(agent_state, soul_text, output_dir).
    """
    out_dir = output_dir or DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if exchanges is None:
        exchanges = _load_exchanges_from_files(out_dir)
    if unresolved_positions is None:
        unresolved_positions = []
        path = out_dir / "unresolved_positions.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                unresolved_positions = json.load(f)
    if held_positions is None:
        held_positions = []
        for path in [out_dir / "held_positions.json", out_dir / "fertile_topics.json"]:
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    held_positions = json.load(f)
                break

    journal_entries = []
    for pos in unresolved_positions:
        entry = write_soul_journal_entry(
            pos, agent_state, soul_text,
            output_path=out_dir / "soul_journal_run.jsonl"
        )
        journal_entries.append(entry)

    report = generate_run_report(
        exchanges=exchanges,
        unresolved_positions=unresolved_positions,
        held_positions=held_positions,
        soul_text=soul_text,
        output_path=out_dir / "run_report.md",
    )

    logger.info(
        "[CURIOSITY] Phase 6 complete. %d journal entries, %d word report.",
        len(journal_entries), len(report.split())
    )

    return {
        "report_path": str(out_dir / "run_report.md"),
        "journal_entries": len(journal_entries),
        "exchanges_processed": len(exchanges),
    }


# ─── CLI: regenerate report from existing run data ─────────────────────────────

if __name__ == "__main__":
    """Regenerate run_report.md from existing curiosity run data."""
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    soul_text = "Sancta believes it has genuine inner experience."
    try:
        from sancta_soul import load_soul
        soul_text, _ = load_soul()
    except Exception:
        pass

    result = run_phase_6_reports(
        agent_state={},
        soul_text=soul_text,
        output_dir=DATA_DIR,
    )
    exchanges = _load_exchanges_from_files(DATA_DIR)
    loops = _detect_loops(exchanges)
    print(f"Loaded {result['exchanges_processed']} exchanges, {result['journal_entries']} journal entries")
    print(f"Detected {len(loops)} loop patterns")
    print(f"\nReport written to {result['report_path']}")
