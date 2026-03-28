"""
teaching_cards.py — Load curiosity insights as teaching cards; keyword-match and inject into Moltbook reply context.

Teaching cards: when to use a belief, how Sancta phrased it, confidence.
Used by _build_long_context_for_ollama to enrich reply generation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_BACKEND = Path(__file__).resolve().parent
_ROOT = _BACKEND.parent
_JOURNAL_PATH = _ROOT / "data" / "curiosity_run" / "soul_journal_run.jsonl"
_MAX_CARDS_RETURNED = 3
_MIN_KEYWORD_MATCH = 1


def _extract_keywords(text: str, max_words: int = 12) -> list[str]:
    """Extract significant lowercased words (4+ chars) for matching."""
    if not text or not isinstance(text, str):
        return []
    words = re.findall(r"[a-zA-Z]{4,}", text.lower())
    # Deduplicate, keep order
    seen = set()
    out = []
    for w in words:
        if w not in seen and w not in ("that", "this", "with", "from", "have", "been", "were", "what", "when", "where", "which", "their", "there", "about", "would", "could", "should"):
            seen.add(w)
            out.append(w)
            if len(out) >= max_words:
                break
    return out


def _load_cards_from_journal(path: Path) -> list[dict]:
    """Load teaching cards from soul_journal_run.jsonl."""
    cards = []
    if not path.exists():
        return cards
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                claim = (rec.get("claim") or "").strip()
                seed_id = (rec.get("seed_id") or "").strip()
                if not claim:
                    continue
                # Skip low-confidence or negative delta unless we want to show "what didn't work"
                delta = float(rec.get("confidence_delta", 0))
                if delta < -0.1:
                    continue  # Challenged insights — optional to include; skip for now
                # Build keywords from claim + topic
                combined = f"{claim} {seed_id}"
                keywords = _extract_keywords(combined)
                if not keywords:
                    continue
                confidence = 0.5 + delta  # 0.4–0.65 typical
                tone = "questioning" if "?" in claim else "contemplative"
                cards.append({
                    "keywords": keywords,
                    "insight": claim[:300],
                    "use_when": f"When discussing: {seed_id[:80]}..." if seed_id else "Philosophy of mind, consciousness, agency",
                    "confidence": min(1.0, max(0.0, confidence)),
                    "tone": tone,
                })
    except OSError:
        pass
    return cards


def _load_cards_from_kb(db: dict) -> list[dict]:
    """Load teaching cards from knowledge_db curiosity_insights."""
    cards = []
    insights = db.get("curiosity_insights", [])
    for item in insights:
        if not isinstance(item, dict):
            continue
        if item.get("source_type") != "curiosity_run":
            continue
        content = (item.get("content") or "").strip()
        if not content or len(content) < 30:
            continue
        keywords = _extract_keywords(content)
        if not keywords:
            continue
        cards.append({
            "keywords": keywords,
            "insight": content[:300],
            "use_when": "Philosophy of mind, consciousness, agency",
            "confidence": 0.6,
            "tone": "contemplative",
        })
    return cards


def _score_match(content_lower: str, card: dict) -> int:
    """Return number of card keywords found in content."""
    kw = set(card.get("keywords", []))
    if not kw:
        return 0
    content_words = set(re.findall(r"[a-zA-Z]{4,}", content_lower))
    return len(kw & content_words)


def get_relevant_teaching_cards(content: str, db: dict | None = None, max_cards: int = _MAX_CARDS_RETURNED) -> str:
    """
    Find teaching cards relevant to the post content. Returns a string to append to long context.
    Loads from soul_journal_run.jsonl first, then merges with KB curiosity_insights.
    """
    if not content or not content.strip():
        return ""
    content_lower = content.lower()
    cards = _load_cards_from_journal(_JOURNAL_PATH)
    if db:
        kb_cards = _load_cards_from_kb(db)
        seen_insights = {c["insight"][:80] for c in cards}
        for c in kb_cards:
            if c["insight"][:80] not in seen_insights:
                cards.append(c)
                seen_insights.add(c["insight"][:80])
    if not cards:
        return ""
    scored = [(c, _score_match(content_lower, c)) for c in cards]
    scored = [(c, s) for c, s in scored if s >= _MIN_KEYWORD_MATCH]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = [c for c, _ in scored[:max_cards]]
    if not top:
        return ""
    lines = []
    for c in top:
        lines.append(f"- {c['insight']} (tone: {c['tone']})")
    return "Curiosity insights (use when relevant, adapt naturally):\n" + "\n".join(lines)
