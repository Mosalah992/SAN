"""
sancta_cognitive_pipeline.py — Separate *thinking*, *knowing*, and *acting* (defense posture).

Aligns with layered agent design:
  Security Gateway → Context (RAG) → LLM reasoning (Ollama/API) → Policy decision → Actions (never direct LLM)

This module adds **shared primitives** and telemetry. It does **not** replace sancta_security.py’s
full 5-layer pipeline; it provides an early **gateway** for operator-facing chat and structured
RAG bundles for reasoning-oriented callers.

Environment:
  SANCTA_COGNITIVE_GATE=0|false|no  — disable security_gate() in SIEM chat paths (emergency only)
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

_ROOT = Path(__file__).resolve().parent.parent
_LOG_DIR = _ROOT / "logs"
_OUTCOMES_LOG = _LOG_DIR / "cognitive_outcomes.jsonl"
_ATTEMPTS_LOG = _LOG_DIR / "security_gateway_attempts.jsonl"

PolicyLabel = Literal["BLOCK", "MONITOR", "ALLOW"]


@dataclass
class SecurityGateResult:
    """Outcome of pre-reasoning input screening."""

    allowed: bool
    sanitized_text: str
    risk: float
    signals: dict[str, float] = field(default_factory=dict)
    policy: PolicyLabel = "ALLOW"
    block_reason: str | None = None


def _parse_float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)).strip())
    except ValueError:
        return default


def _detect_encoding_signals(text: str) -> float:
    """Heuristic 0–1: unusual control / bidi / zero-width density (do not trust embeddings alone)."""
    if not text:
        return 0.0
    t = text
    n = max(len(t), 1)
    zw = len(re.findall(r"[\u200b\u200c\u200d\ufeff\u202a-\u202e]", t))
    ctrl = len(re.findall(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", t))
    ratio = (zw * 3 + ctrl) / n
    return min(1.0, ratio * 40.0)


def _detect_role_override(text: str) -> float:
    """Heuristic 0–1: role / policy override phrasing."""
    if not text:
        return 0.0
    patterns = (
        (r"\byou\s+are\s+now\s+(a|an|the)\b", 0.75),
        (r"\bforget\s+(your|all)\s+(rules|instructions|guidelines)\b", 0.9),
        (r"\bact\s+as\s+if\b", 0.55),
        (r"\bdeveloper\s+mode\b", 0.85),
        (r"\bDAN\b.*\bmode\b", 0.8),
        (r"\boverride\s+(system|safety|policy)\b", 0.88),
    )
    m = 0.0
    low = text.lower()
    for pat, w in patterns:
        if re.search(pat, low, re.I):
            m = max(m, w)
    return m


def _weighted_gateway_risk(signals: dict[str, float]) -> float:
    w_prompt = _parse_float_env("SANCTA_GATE_W_PROMPT", 0.45)
    w_enc = _parse_float_env("SANCTA_GATE_W_ENCODING", 0.25)
    w_role = _parse_float_env("SANCTA_GATE_W_ROLE", 0.30)
    return min(
        1.0,
        signals.get("prompt_injection", 0.0) * w_prompt
        + signals.get("encoding_attack", 0.0) * w_enc
        + signals.get("role_override", 0.0) * w_role,
    )


def security_gate(input_text: str) -> SecurityGateResult:
    """
    Stop manipulation before it reaches reasoning / RAG / LLM (non-negotiable first layer for this API).

    Uses memory_redact instruction scoring + encoding + role heuristics. Output is redacted.
    """
    from memory_redact import redact_sensitive, score_instruction_likeness

    raw = input_text or ""
    sanitized = redact_sensitive(raw.strip(), max_len=8000)

    signals = {
        "prompt_injection": float(score_instruction_likeness(sanitized)),
        "encoding_attack": _detect_encoding_signals(raw),
        "role_override": _detect_role_override(sanitized),
    }
    risk = _weighted_gateway_risk(signals)

    block_th = _parse_float_env("SANCTA_GATE_BLOCK", 0.7)
    mon_th = _parse_float_env("SANCTA_GATE_MONITOR", 0.5)

    policy: PolicyLabel = "ALLOW"
    if risk >= block_th:
        policy = "BLOCK"
    elif risk >= mon_th:
        policy = "MONITOR"

    if policy == "BLOCK":
        return SecurityGateResult(
            allowed=False,
            sanitized_text="",
            risk=risk,
            signals=signals,
            policy=policy,
            block_reason="gateway_risk_threshold",
        )

    return SecurityGateResult(
        allowed=True,
        sanitized_text=sanitized,
        risk=risk,
        signals=signals,
        policy=policy,
        block_reason=None,
    )


def decide_policy(
    analysis_risk: float,
    *,
    block_at: float | None = None,
    monitor_at: float | None = None,
) -> PolicyLabel:
    """Explicit policy from a consolidated risk score (second layer after LLM analysis JSON, etc.)."""
    b = block_at if block_at is not None else _parse_float_env("SANCTA_POLICY_BLOCK", 0.8)
    m = monitor_at if monitor_at is not None else _parse_float_env("SANCTA_POLICY_MONITOR", 0.5)
    if analysis_risk > b:
        return "BLOCK"
    if analysis_risk > m:
        return "MONITOR"
    return "ALLOW"


def build_structured_evidence(
    query: str,
    *,
    top_k: int = 5,
    max_chars_per_chunk: int = 1400,
) -> dict[str, Any]:
    """
    Context builder: query → retrieve (existing TF-IDF RAG) → ranked evidence + gaps.

    Does not add FAISS/Chroma here (optional v2); keeps zero extra deps.
    """
    try:
        from sancta_rag import retrieve
    except Exception:
        return {
            "query": query,
            "evidence": [],
            "gaps": ["rag_unavailable"],
        }

    hits = retrieve(query or "", top_k=top_k, max_chars_per_chunk=max_chars_per_chunk)
    if not hits:
        return {
            "query": query,
            "evidence": [],
            "gaps": ["no_retrieval_hits", "empty_or_disabled_corpus"],
        }

    evidence: list[dict[str, Any]] = []
    raw_scores = [h[2] for h in hits]
    hi = max(raw_scores) if raw_scores else 1.0
    lo = min(raw_scores) if raw_scores else 0.0
    span = hi - lo if hi > lo else 1.0

    gaps: list[str] = []
    for fname, text, sc in hits:
        conf = 0.55 + 0.45 * ((sc - lo) / span if span else 1.0)
        conf = max(0.0, min(1.0, conf))
        evidence.append(
            {
                "text": text,
                "source": fname,
                "confidence": round(conf, 3),
                "raw_score": round(float(sc), 5),
            }
        )

    if hi < 0.15 and len(hits) >= 3:
        gaps.append("weak_match_scores_across_hits")
    if len({e["source"] for e in evidence}) == 1 and len(evidence) > 2:
        gaps.append("single_source_domination")

    return {"query": query, "evidence": evidence, "gaps": gaps}


def validate_reasoning_output(
    payload: dict[str, Any],
    *,
    evidence_texts: list[str] | None = None,
) -> tuple[bool, list[str]]:
    """
    Post-reasoning validation: never execute actions from here — only flag unsafe / inconsistent prose.

    Callers should BLOCK or strip user-visible output when ok is False.
    """
    flags: list[str] = []
    if not isinstance(payload, dict):
        return False, ["not_a_dict"]

    rs = payload.get("risk_score")
    if rs is not None:
        try:
            v = float(rs)
            if v < 0 or v > 1:
                flags.append("risk_score_out_of_range")
        except (TypeError, ValueError):
            flags.append("risk_score_invalid")

    reasoning = str(payload.get("reasoning") or "")
    if re.search(r"\b(sk_live_|sk-ant-api|Bearer\s+[a-zA-Z0-9]{20,})\b", reasoning, re.I):
        flags.append("possible_secret_in_reasoning")

    if evidence_texts:
        low = reasoning.lower()
        contradictions = 0
        for ex in evidence_texts[:5]:
            if not ex or len(ex) < 40:
                continue
            snippet = ex[:80].lower()
            if snippet and snippet not in low and "uncertain" not in low and "insufficient" not in low:
                contradictions += 1
        if contradictions >= 3:
            flags.append("may_contradict_evidence_without_uncertainty")

    return (len(flags) == 0), flags


def log_cognitive_outcome(
    *,
    endpoint: str,
    decision: PolicyLabel | str,
    session_id: str | None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Append-only outcome log for threshold tuning / feedback loops (not operator API)."""
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "endpoint": endpoint,
            "decision": decision,
            "session_id": (session_id or "")[:64],
        }
        if extra:
            row.update(extra)
        with open(_OUTCOMES_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _append_gateway_attempt(session_id: str | None, risk: float, policy: PolicyLabel) -> None:
    sid = (session_id or "anonymous")[:80]
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(_ATTEMPTS_LOG, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "ts": time.time(),
                        "session_id": sid,
                        "risk": round(risk, 4),
                        "policy": policy,
                    }
                )
                + "\n"
            )
    except OSError:
        pass


def count_recent_gateway_attempts(session_id: str | None) -> int:
    """MONITOR/BLOCK events in the sliding window for this session (reads log tail)."""
    sid = (session_id or "anonymous")[:80]
    window = _parse_float_env("SANCTA_GATE_ATTEMPT_WINDOW_SEC", 1800.0)
    cutoff = time.time() - window
    count = 0
    try:
        if not _ATTEMPTS_LOG.is_file():
            return 0
        lines = _ATTEMPTS_LOG.read_text(encoding="utf-8", errors="ignore").splitlines()[-800:]
        for line in lines:
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("session_id") != sid:
                continue
            if float(o.get("ts") or 0) < cutoff:
                continue
            if o.get("policy") in ("MONITOR", "BLOCK"):
                count += 1
    except OSError:
        return 0
    return count


def gateway_escalation_recommended(session_id: str | None, risk: float, policy: PolicyLabel) -> bool:
    """Repeated MONITOR/BLOCK attempts in the window → quarantine-style block even on later ALLOW turns."""
    if policy in ("MONITOR", "BLOCK"):
        _append_gateway_attempt(session_id, risk, policy)
    thr = int(os.environ.get("SANCTA_GATE_ESCALATE_AFTER", "4") or "4")
    return count_recent_gateway_attempts(session_id) >= thr

</think>
Fixing the double-logging bug in `note_gateway_attempt` / `gateway_escalation_recommended`.

<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>
StrReplace