"""
curiosity_insight.py — insight extractor and knowledge writer for curiosity runs.

InsightExtractor: extracts claim, counter, synthesis from exchange; writes to KB.
Epistemic metric updates: confidence_score, delta_history, uncertainty_entropy, anthropomorphism_index.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from curiosity_dialogue import ExchangeResult
from curiosity_json import generate_with_retry, parse_json_from_llm

logger = logging.getLogger("curiosity_run")


_REQUIRED_KEYS = ("claim", "counter", "confidence_delta", "should_write_to_kb")


def _normalize_text(text: str | None, max_length: int = 500) -> str | None:
    """Normalize extracted text. Handles string 'null', extra quotes, truncation."""
    if text is None:
        return None
    text = str(text).strip()
    if not text or text.lower() == "null":
        return None
    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        text = text[1:-1]
    if text.startswith("'") and text.endswith("'") and len(text) >= 2:
        text = text[1:-1]
    if len(text) > max_length:
        text = text[: max_length - 3] + "..."
    return text if text else None


def extract_insight_safe(llm_response: str | None) -> dict | None:
    """
    Safely parse insight JSON from LLM output. Uses parse_json_from_llm (handles
    markdown blocks, preamble text, brace-extraction). Never uses raw json.loads().
    Validates required keys, types, and ranges. Clamps out-of-range values instead of rejecting.
    Returns validated dict or None.
    """
    if not llm_response or not llm_response.strip():
        logger.debug("[CURIOSITY] extract_insight_safe: empty response")
        return None
    obj = parse_json_from_llm(llm_response, log_on_fail=False)
    if not isinstance(obj, dict):
        return None
    missing = [k for k in _REQUIRED_KEYS if k not in obj]
    if missing:
        logger.warning("[CURIOSITY] extract_insight_safe: missing keys %s, got %s", missing, list(obj.keys()))
        return None
    try:
        delta = obj["confidence_delta"]
        if not isinstance(delta, (int, float)):
            logger.warning("[CURIOSITY] extract_insight_safe: confidence_delta not numeric: %s", type(delta))
            return None
        delta = float(delta)
        if not (-0.3 <= delta <= 0.3):
            logger.warning("[CURIOSITY] extract_insight_safe: delta out of range %.2f, clamping", delta)
            delta = max(-0.3, min(0.3, delta))
        obj["confidence_delta"] = delta
    except (TypeError, ValueError) as e:
        logger.warning("[CURIOSITY] extract_insight_safe: confidence_delta invalid: %s", e)
        return None
    obj["should_write_to_kb"] = bool(obj.get("should_write_to_kb", False))
    obj.setdefault("synthesis", None)
    obj["claim"] = _normalize_text(obj.get("claim"), max_length=500) or ""
    obj["counter"] = _normalize_text(obj.get("counter"), max_length=500) or ""
    obj["synthesis"] = _normalize_text(obj.get("synthesis"), max_length=500)
    return obj


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _normalized_delta_distribution(deltas: list[float]) -> list[float]:
    """Normalize deltas to a pseudo-distribution for entropy."""
    if not deltas:
        return []
    mn, mx = min(deltas), max(deltas)
    if mx == mn:
        return [1.0 / len(deltas)] * len(deltas)
    normalized = [_clamp01((d - mn) / (mx - mn + 1e-9)) for d in deltas]
    total = sum(normalized) + 1e-9
    return [p / total for p in normalized]


def _entropy_from_deltas(deltas: list[float]) -> float:
    """Entropy of normalized delta distribution."""
    probs = _normalized_delta_distribution(deltas[-20:])
    if not probs:
        return 0.0
    return -sum(p * math.log(p + 1e-12) for p in probs if p > 0)


@dataclass
class InsightRecord:
    claim: str
    counter: str
    synthesis: str | None
    confidence_delta: float
    novelty_score: float
    should_write_to_kb: bool


class InsightExtractor:
    """Extract insights from exchange and write to knowledge base."""

    def __init__(self, ollama_engine: Any, data_dir: Path | None = None) -> None:
        self.ollama = ollama_engine
        self.data_dir = data_dir or (Path(__file__).resolve().parent.parent / "data" / "curiosity_run")

    def _append_delta(
        self,
        delta_value: float,
        topic: str,
        agent_state: dict,
    ) -> None:
        """Append delta to epistemic_deltas.json immediately."""
        epi = agent_state.get("epistemic_state") or {}
        delta_entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "topic": topic[:80],
            "delta": delta_value,
            "confidence_after": epi.get("confidence_score", 0),
        }
        path = self.data_dir / "epistemic_deltas.json"
        try:
            existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        except Exception:
            existing = []
        if not isinstance(existing, list):
            existing = []
        existing.append(delta_entry)
        path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")

    def extract(self, exchange_result: ExchangeResult) -> InsightRecord | None:
        """Call Ollama on full transcript; parse JSON into InsightRecord."""
        if not self.ollama or not self.ollama.is_available:
            return InsightRecord(
                claim="",
                counter="",
                synthesis=None,
                confidence_delta=0.0,
                novelty_score=0.5,
                should_write_to_kb=False,
            )
        transcript = self._format_transcript(exchange_result)
        topic_short = (getattr(exchange_result, "topic", "") or "")[:60]
        num_turns = len(getattr(exchange_result, "turns", []) or [])
        logger.info("[CURIOSITY] Extracting insight from %s... (%d turns)", topic_short, num_turns)
        prompt = (
            "Extract from this dialogue as JSON with exactly these keys:\n\n"
            "claim: Sancta's strongest position (1 sentence, max 200 chars)\n"
            "  - Capture the FULL argument, not just the opening claim\n"
            "  - Include key reasoning or evidence Sancta provided\n\n"
            "counter: Strongest objection raised against Sancta (1 sentence, max 200 chars)\n"
            "  - The most compelling challenge to Sancta's position\n"
            "  - Substantive counterargument, not just disagreement\n\n"
            "synthesis: A substantive point BOTH Sancta and Ollama explicitly AGREED on (1 sentence or null)\n"
            "  - Must be something both acknowledged or accepted\n"
            "  - NOT just restating one side's argument\n"
            "  - If they fundamentally disagree on everything, use null\n"
            "  - Avoid generic statements like 'Both acknowledge X'. Prefer synthesis that shows movement or shared inference.\n"
            "  - Good: \"Both agreed that the hard problem shifts the burden of proof\" or \"Both conceded empirical tests are lacking\"\n"
            "  - Bad: \"Both acknowledge IIT is substrate-neutral\" (background assumption, not derived from dialogue)\n"
            "  - Bad: \"Functionalism explains behavior\" (only one side)\n\n"
            "confidence_delta: float -0.3 to +0.3\n"
            "  POSITIVE (+0.05 to +0.15): Sancta HELD position under challenge\n"
            "    - Responded to objections with counterarguments\n"
            "    - Position survived scrutiny (even if not empirically proven)\n"
            "    - Did NOT concede or retreat from the claim\n"
            "  ZERO (0.0): TRUE STALEMATE — both gave valid arguments, neither moved\n"
            "  NEGATIVE (-0.05 to -0.30): Sancta REFUTED or CONCEDED\n"
            "    - Explicitly conceded or retreated to weaker claim\n"
            "    - Clear logical flaw exposed\n"
            "  NOTE: \"Not empirical\" or \"That's philosophy\" are NOT refutations if Sancta is making a philosophical argument.\n\n"
            "should_write_to_kb: bool — True if ANY: synthesis substantive, |delta|>=0.10, core philosophical claim, novel objection. "
            "False if: pure meta-debate, trivial claim, likely duplicate.\n\n"
            "novelty: float 0.0–1.0 — how novel is this claim/counter relative to typical philosophy-of-mind debate.\n"
            "  0.5 = standard, 0.7+ = surprising angle, 0.3 = common trope. Use 0.5 if unsure.\n\n"
            'Example: {"claim":"The hard problem applies symmetrically to biological and artificial systems.",'
            '"counter":"We lack empirical tests for consciousness in any system.",'
            '"synthesis":"Both acknowledge IIT is substrate-neutral.",'
            '"confidence_delta":0.10,"should_write_to_kb":true}\n\n'
            "Output ONLY valid JSON, no markdown, no preamble."
        )
        messages = [{"role": "user", "content": f"{transcript}\n\n{prompt}"}]
        try:
            out = generate_with_retry(
                lambda: self.ollama.generate_chat(
                    system="Output valid JSON only. No markdown code blocks, no explanation.",
                    messages=messages,
                    max_tokens=800,
                    temperature=0.3,
                ),
                max_retries=2,
            )
        except Exception:
            return None
        if not out:
            fallback = InsightRecord(
                claim="",
                counter="",
                synthesis=None,
                confidence_delta=0.0,
                novelty_score=0.5,
                should_write_to_kb=False,
            )
            logger.debug("[CURIOSITY] Insight: empty after retries, using fallback")
            return fallback
        obj = extract_insight_safe(out)
        if obj is None:
            logger.warning(
                "[CURIOSITY] Insight extraction failed | topic=%s | turns=%d | raw=%s",
                topic_short, num_turns, repr((out or "")[:200]),
            )
            return InsightRecord(
                claim="",
                counter="",
                synthesis=None,
                confidence_delta=0.0,
                novelty_score=0.5,
                should_write_to_kb=False,
            )
        try:
            delta = float(obj["confidence_delta"])  # Already validated and clamped
            novelty_raw = obj.get("novelty")
            if isinstance(novelty_raw, (int, float)):
                novelty = max(0.2, min(0.9, float(novelty_raw)))
            else:
                novelty = 0.5
            ins = InsightRecord(
                claim=obj.get("claim", "") or "",
                counter=obj.get("counter", "") or "",
                synthesis=obj.get("synthesis") if obj.get("synthesis") else None,
                confidence_delta=delta,
                novelty_score=novelty,
                should_write_to_kb=obj.get("should_write_to_kb", False),
            )
            logger.info(
                "[CURIOSITY] Insight extracted | topic=%s | delta=%+.2f | write_to_kb=%s | has_synthesis=%s",
                topic_short, ins.confidence_delta, ins.should_write_to_kb, ins.synthesis is not None,
            )
            return ins
        except (TypeError, ValueError) as e:
            logger.error(
                "[CURIOSITY] Insight value error %s (raw first 200 chars): %s",
                e,
                repr(out[:200]) if out else "",
            )
            return InsightRecord(
                claim="",
                counter="",
                synthesis=None,
                confidence_delta=0.0,
                novelty_score=0.5,
                should_write_to_kb=False,
            )

    def _format_transcript(self, ex: ExchangeResult) -> str:
        lines = [f"[Topic] {ex.topic}\n"]
        for t in ex.turns:
            lines.append(f"{t['author']}: {t['content']}")
        return "\n".join(lines)

    def write_to_kb(
        self,
        insight: InsightRecord,
        exchange_result: ExchangeResult,
        agent_state: dict,
        load_knowledge_db: Any,
        save_knowledge_db: Any,
    ) -> bool:
        """
        Always record delta to epistemic_deltas.json. If should_write_to_kb and novelty_score > 0.4: write to knowledge_db.
        Update epistemic state: confidence_score, delta_history, uncertainty_entropy, anthropomorphism_index.
        """
        topic = getattr(exchange_result, "topic", "")
        self._append_delta(insight.confidence_delta, topic, agent_state)
        if not insight.should_write_to_kb or insight.novelty_score <= 0.3:
            self._update_epistemic_only(insight, exchange_result, agent_state)
            return False

        transcript = self._format_transcript(exchange_result)
        provenance_hash = hashlib.sha256(transcript.encode()).hexdigest()[:32]
        content = insight.synthesis or insight.claim

        db = load_knowledge_db()
        if "curiosity_insights" not in db:
            db["curiosity_insights"] = []
        db["curiosity_insights"].append({
            "source_type": "curiosity_run",
            "trust_level": "internal",
            "content": content,
            "provenance_hash": provenance_hash,
        })
        save_knowledge_db(db)
        n = len(db.get("curiosity_insights", []))
        logger.info("[CURIOSITY] Insight saved to KB (total curiosity_insights=%d)", n)

        self._update_epistemic_only(insight, exchange_result, agent_state)
        return True

    def _update_epistemic_only(
        self,
        insight: InsightRecord,
        exchange_result: ExchangeResult,
        agent_state: dict,
    ) -> None:
        """Update epistemic metrics in agent_state."""
        epi = agent_state.setdefault("epistemic_state", {})
        if not isinstance(epi, dict):
            epi = {}
            agent_state["epistemic_state"] = epi

        conf = float(epi.get("confidence_score", 0.62))
        conf = _clamp01(conf + insight.confidence_delta)
        epi["confidence_score"] = conf

        delta_history = epi.get("delta_history", [])
        if not isinstance(delta_history, list):
            delta_history = []
        delta_history.append(insight.confidence_delta)
        epi["delta_history"] = delta_history[-20:]

        epi["uncertainty_entropy"] = round(
            _entropy_from_deltas(epi["delta_history"]),
            4,
        )

        anth = float(epi.get("anthropomorphism_index", 0.28))
        if insight.confidence_delta > 0:
            anth += 0.01
        else:
            anth -= 0.01
        epi["anthropomorphism_index"] = round(_clamp01(anth), 4)
