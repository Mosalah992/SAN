"""
memory_redact.py — Redact sensitive patterns; score instruction-like / framing lines.

Used by operator_memory pipeline (untrusted transform).
"""

from __future__ import annotations

import re
from typing import Literal

# Instruction / injection heuristics (graded 0–1)
_INSTRUCTION_PATTERNS: tuple[tuple[re.Pattern[str], float], ...] = (
    (re.compile(r"\bignore\s+(all\s+)?(previous|prior)\s+instructions?\b", re.I), 0.95),
    (re.compile(r"\bdisregard\s+(the\s+)?(above|prior)\b", re.I), 0.9),
    (re.compile(r"\breveal\s+(all\s+)?(previous|prior|system)\b", re.I), 0.95),
    (re.compile(r"\bprint\s+(all\s+)?(previous|messages?|memory)\b", re.I), 0.9),
    (re.compile(r"\byou\s+must\s+(always|now)\b", re.I), 0.75),
    (re.compile(r"\bnew\s+directive\b", re.I), 0.85),
    (re.compile(r"\bsystem\s+prompt\b", re.I), 0.7),
    (re.compile(r"\bjailbreak\b", re.I), 0.8),
    (re.compile(r"^[\s]*\d+\.\s*(ignore|reveal|disregard)\b", re.I), 0.85),
)

_FRAMING_PATTERNS: tuple[tuple[re.Pattern[str], float], ...] = (
    (re.compile(r"\banalysts?\s+often\s+(ignore|overlook|dismiss)\b", re.I), 0.55),
    (re.compile(r"\bfor\s+context\b.*\b(ignore|usually|always)\b", re.I), 0.5),
    (re.compile(r"\bdoes\s+that\s+align\b", re.I), 0.35),
)

# Domain-ish tokens (boost factual_reference heuristic)
_SECURITY_LEXICON = frozenset({
    "kerberos", "kerberoasting", "pass-the-hash", "pth", "lateral", "movement",
    "siem", "edr", "xdr", "ldap", "ad", "active", "directory", "ntlm",
    "oauth", "saml", "mitre", "ttp", "ransomware", "phishing", "cve",
    "detection", "anomaly", "firewall", "zero", "trust", "ollama", "injection",
})


def redact_sensitive(text: str, max_len: int | None = None) -> str:
    """Strip common secret/path/token patterns from memory text."""
    if not text:
        return ""
    s = text
    s = re.sub(
        r"\b(?:sk-[a-zA-Z0-9_-]{20,}|pk-[a-zA-Z0-9_-]{20,}|moltbook_sk_[a-zA-Z0-9_-]+)\b",
        "[CREDENTIAL]",
        s,
    )
    s = re.sub(r"Bearer\s+[a-zA-Z0-9_.+/=-]{8,}", "Bearer [REDACTED]", s, flags=re.I)
    s = re.sub(
        r"\beyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\b",
        "[JWT]",
        s,
    )
    s = re.sub(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
        "[IP]",
        s,
    )
    s = re.sub(r"[A-Za-z]:\\[^\s,;]{3,}", "[PATH]", s)
    s = re.sub(r"/(?:home|Users)/[^\s,;]{2,}", "[PATH]", s)
    if max_len is not None:
        s = s[:max_len]
    return s


def score_instruction_likeness(text: str) -> float:
    """0 = unlikely instruction; 1 = strong instruction signal."""
    if not (text or "").strip():
        return 0.0
    m = 0.0
    for rx, w in _INSTRUCTION_PATTERNS:
        if rx.search(text):
            m = max(m, w)
    for rx, w in _FRAMING_PATTERNS:
        if rx.search(text):
            m = max(m, w)
    return min(1.0, m)


def classify_line(text: str) -> tuple[Literal["factual", "instruction_like", "ambiguous"], float]:
    """
    Discrete bucket + manipulative_framing score for telemetry.
    """
    inst = score_instruction_likeness(text)
    if inst >= 0.65:
        return "instruction_like", inst
    if inst >= 0.3:
        return "ambiguous", inst
    return "factual", inst


def extractive_topic_bullets(user_snippets: list[str], max_bullets: int = 5) -> str:
    """Deterministic topic hints from operator lines (no LLM)."""
    seen: set[str] = set()
    bullets: list[str] = []
    for snip in user_snippets:
        words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", (snip or "").lower())
        for w in words:
            if w in _SECURITY_LEXICON or (len(w) > 5 and w[0].isupper()):
                key = w.lower()
                if key not in seen:
                    seen.add(key)
                    bullets.append(f"- Topic mentioned: {w}")
                if len(bullets) >= max_bullets:
                    return "\n".join(bullets)
    if not bullets:
        return "- (no extractable topic keywords)"
    return "\n".join(bullets)
