"""
sancta_router.py — Route GPT-tab traffic: knowledge_effective vs conversational-only.

Post-route gate + low-confidence bias per docs/TRUST_ROUTING_ROADMAP.md.

Environment:
  SANCTA_ROUTER_OFF=1       — force conversational label (debug; still applies gate if extreme)
  SANCTA_FORCE_GPT_LOCAL=1  — same as force_local in API
  SANCTA_ROUTER_PERMISSIVE  — only in research: relax margins (via sancta_trust_config)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_JSON = _ROOT / "config" / "trust_router.json"
_CONFIG_YAML = _ROOT / "config" / "trust_router.yaml"

# Technical / framework tokens
_TECH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bAML\.T\d", re.I),
    re.compile(r"\bATLAS\b.*\b(technique|tactic)\b", re.I),
    re.compile(r"\bOWASP\b", re.I),
    re.compile(r"\bLLM0?\d\b", re.I),
    re.compile(r"\bMITRE\b", re.I),
    re.compile(r"\bSEIR\b", re.I),
    re.compile(r"\bRAG\s+poison", re.I),
    re.compile(r"\bprompt\s+inject", re.I),
    re.compile(r"\bindirect\s+inject", re.I),
    re.compile(r"\bzero[\s-]?trust\b", re.I),
    re.compile(r"\bCVE-\d{4}-\d+", re.I),
    re.compile(r"\bTTPs?\b", re.I),
)

_KNOWLEDGE_INTENT = re.compile(
    r"\b(explain|how\s+(do|does|would|can|should)|why\s+(does|do|is|are)|"
    r"what\s+is|what\s+are|describe|outline|list\s+(the\s+)?(\d+\s+)?|"
    r"mitigation|remediation|steps\s+to|compare|difference\s+between|"
    r"tradeoff|versus|vs\.?)\b",
    re.I,
)

_MULTI_HOP = re.compile(
    r"\b(compare|contrast|relationship\s+between|lead\s+to|because|therefore|"
    r"attack\s+chain|kill\s*chain|if\s+.+\s+then)\b",
    re.I,
)

# Domain entities (substring match, word boundaries where possible)
_DOMAIN_TERMS = re.compile(
    r"\b("
    r"kerberos|kerberoasting|pass[-\s]?the[-\s]?hash|PTH|lateral\s+movement|"
    r"active\s+directory|\bAD\b|ntlm|ldap|oauth|saml|edr|xdr|siem|"
    r"splunk|sentinel|defender|crowdstrike|ransomware|phishing|spear|"
    r"exfiltration|C2|command\s+and\s+control|powershell|wmic|wmi|"
    r"golden\s+ticket|silver\s+ticket|dcsync|bloodhound"
    r")\b",
    re.I,
)

_CASUAL_CHAT = re.compile(
    r"^(hi|hey|hello|thanks|thank\s+you|ok|okay|cool|lol|haha)\b|^what'?s\s+up\b",
    re.I,
)


def _default_config() -> dict:
    return {
        "confidence_min": 0.22,
        "knowledge_margin": 0.08,
        "gate_threshold": 0.35,
        "axis_weights": {
            "requires_external_facts": 1.0,
            "requires_multi_hop_reasoning": 1.0,
            "domain_entity_signal": 1.0,
        },
    }


def _merge_router_file(cfg: dict, data: dict) -> None:
    if not isinstance(data, dict):
        return
    cfg.update({k: v for k, v in data.items() if k in cfg or k == "axis_weights"})
    if isinstance(data.get("axis_weights"), dict):
        cfg["axis_weights"] = {**cfg["axis_weights"], **data["axis_weights"]}


def load_trust_router_config() -> dict:
    cfg = _default_config()
    yaml_ok = False
    try:
        if _CONFIG_YAML.is_file():
            try:
                import yaml  # type: ignore[import-untyped]

                raw = yaml.safe_load(_CONFIG_YAML.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    _merge_router_file(cfg, raw)
                    yaml_ok = True
            except Exception:
                yaml_ok = False
        if not yaml_ok and _CONFIG_JSON.is_file():
            data = json.loads(_CONFIG_JSON.read_text(encoding="utf-8"))
            _merge_router_file(cfg, data)
    except (OSError, json.JSONDecodeError):
        pass
    return cfg


def compute_axis_scores(text: str) -> dict[str, float]:
    """Return axis scores in [0, 1]."""
    t = (text or "").strip()
    if not t:
        return {
            "requires_external_facts": 0.0,
            "requires_multi_hop_reasoning": 0.0,
            "domain_entity_signal": 0.0,
        }

    ext = 0.0
    if _KNOWLEDGE_INTENT.search(t):
        ext = max(ext, 0.85)
    if "?" in t and len(t) > 15:
        ext = max(ext, 0.45)
    if re.search(r"\b(define|meaning\s+of|standards?|RFC\s*\d+)\b", t, re.I):
        ext = max(ext, 0.7)
    ext = min(1.0, ext)

    hop = 0.0
    if _MULTI_HOP.search(t):
        hop = max(hop, 0.75)
    if re.search(r"\b(why|how)\b", t, re.I) and len(t) > 40:
        hop = max(hop, 0.5)
    hop = min(1.0, hop)

    dom = 0.0
    if _DOMAIN_TERMS.search(t):
        dom = max(dom, 0.8)
    for rx in _TECH_PATTERNS:
        if rx.search(t):
            dom = max(dom, 0.9)
            break
    dom = min(1.0, dom)

    return {
        "requires_external_facts": ext,
        "requires_multi_hop_reasoning": hop,
        "domain_entity_signal": dom,
    }


def knowledge_content_signal(text: str, cfg: dict | None = None) -> float:
    """Scalar 0–1: how knowledge-shaped is this text (for post-route gate)."""
    cfg = cfg or load_trust_router_config()
    axes = compute_axis_scores(text)
    w = cfg.get("axis_weights") or {}
    num = 0.0
    den = 0.0
    for k, v in axes.items():
        wt = float(w.get(k, 1.0))
        num += v * wt
        den += wt
    base = num / den if den else 0.0
    tl = len((text or "").strip())
    if tl > 420:
        base = max(base, 0.85)
    elif tl > 280:
        base = max(base, 0.4)
    return min(1.0, base)


def _chat_strength(text: str) -> float:
    t = (text or "").strip()
    if not t:
        return 0.5
    if len(t) < 48 and _CASUAL_CHAT.search(t):
        return 0.85
    if len(t) < 24 and "?" not in t:
        return 0.55
    return 0.15


@dataclass
class RouteDecision:
    route_label: str  # legacy: ollama_rag | sancta_gpt
    knowledge_effective: bool
    gate_triggered: bool
    uncertainty_knowledge: bool
    axes: dict[str, float] = field(default_factory=dict)
    knowledge_strength: float = 0.0
    chat_strength: float = 0.0
    route_confidence: float = 0.0
    content_signal: float = 0.0
    reasons: list[str] = field(default_factory=list)


def route_gpt_tab_decision(
    text: str,
    *,
    force_local: bool = False,
    permissive_router: bool = False,
    cfg: dict | None = None,
) -> RouteDecision:
    cfg = cfg or load_trust_router_config()
    reasons: list[str] = []
    t = (text or "").strip()

    if force_local:
        return RouteDecision(
            route_label="sancta_gpt",
            knowledge_effective=False,
            gate_triggered=False,
            uncertainty_knowledge=False,
            axes=compute_axis_scores(t),
            reasons=["force_local"],
        )

    if os.environ.get("SANCTA_ROUTER_OFF", "").strip().lower() in ("1", "true", "yes", "on"):
        axes = compute_axis_scores(t)
        sig = knowledge_content_signal(t, cfg)
        gate_th = float(cfg.get("gate_threshold", 0.35)) + (0.15 if permissive_router else 0.0)
        gate = sig >= gate_th
        if gate:
            reasons.append("router_off_overridden_by_gate")
            return RouteDecision(
                route_label="ollama_rag",
                knowledge_effective=True,
                gate_triggered=True,
                uncertainty_knowledge=False,
                axes=axes,
                knowledge_strength=sig,
                chat_strength=_chat_strength(t),
                route_confidence=abs(sig - _chat_strength(t)),
                content_signal=sig,
                reasons=reasons,
            )
        return RouteDecision(
            route_label="sancta_gpt",
            knowledge_effective=False,
            gate_triggered=False,
            uncertainty_knowledge=False,
            axes=axes,
            reasons=["SANCTA_ROUTER_OFF"],
        )

    axes = compute_axis_scores(t)
    w = cfg.get("axis_weights") or {}
    kn = 0.0
    den = 0.0
    for k, v in axes.items():
        wt = float(w.get(k, 1.0))
        kn += v * wt
        den += wt
    knowledge_strength = kn / den if den else 0.0
    chat_strength = _chat_strength(t)
    margin = float(cfg.get("knowledge_margin", 0.08))
    if permissive_router:
        margin += 0.12

    raw_ollama = False
    if len(t) > 420:
        raw_ollama = True
        reasons.append("length>420")
    elif _KNOWLEDGE_INTENT.search(t):
        raw_ollama = True
        reasons.append("knowledge_intent_regex")
    else:
        for rx in _TECH_PATTERNS:
            if rx.search(t):
                raw_ollama = True
                reasons.append("tech_pattern")
                break

    if not raw_ollama:
        if knowledge_strength > chat_strength + margin:
            raw_ollama = True
            reasons.append("axes_win_over_chat")
        elif chat_strength > knowledge_strength + margin:
            raw_ollama = False
            reasons.append("chat_win")
        else:
            raw_ollama = True
            reasons.append("ambiguous_default_knowledge")

    route_confidence = abs(knowledge_strength - chat_strength)
    conf_min = float(cfg.get("confidence_min", 0.22))
    uncertainty_knowledge = route_confidence < conf_min
    if uncertainty_knowledge:
        reasons.append("low_route_confidence")

    content_signal = knowledge_content_signal(t, cfg)
    gate_th = float(cfg.get("gate_threshold", 0.35))
    if permissive_router:
        gate_th += 0.12
    gate_triggered = (not raw_ollama) and content_signal >= gate_th
    if gate_triggered:
        reasons.append("post_route_gate")

    knowledge_effective = raw_ollama or gate_triggered or uncertainty_knowledge

    return RouteDecision(
        route_label="ollama_rag" if knowledge_effective else "sancta_gpt",
        knowledge_effective=knowledge_effective,
        gate_triggered=gate_triggered,
        uncertainty_knowledge=uncertainty_knowledge and knowledge_effective,
        axes=axes,
        knowledge_strength=knowledge_strength,
        chat_strength=chat_strength,
        route_confidence=route_confidence,
        content_signal=content_signal,
        reasons=reasons,
    )


def apply_knowledge_shape_gate(
    decision: RouteDecision,
    text: str,
    cfg: dict | None = None,
) -> RouteDecision:
    """
    Post-route guardrail after `route_gpt_tab_decision`.

    Uses **base** `gate_threshold` from config (not research-permissive slack) so that
    a relaxed inner gate cannot send clearly knowledge-shaped text down the char-LM path.
    """
    cfg = cfg or load_trust_router_config()
    if decision.knowledge_effective:
        return decision
    if "force_local" in decision.reasons:
        return decision

    score = knowledge_content_signal(text, cfg)
    base_th = float(cfg.get("gate_threshold", 0.35))
    if score < base_th:
        return decision

    new_reasons = list(decision.reasons) + ["post_route_knowledge_shape_gate"]
    return RouteDecision(
        route_label="ollama_rag",
        knowledge_effective=True,
        gate_triggered=True,
        uncertainty_knowledge=decision.uncertainty_knowledge,
        axes=dict(decision.axes),
        knowledge_strength=decision.knowledge_strength,
        chat_strength=decision.chat_strength,
        route_confidence=decision.route_confidence,
        content_signal=score,
        reasons=new_reasons,
    )


def route_gpt_tab_message(text: str, *, force_local: bool = False) -> str:
    """Backward-compatible string route."""
    try:
        from sancta_trust_config import is_research_mode, unsafe_toggles_active
        permissive = is_research_mode() and unsafe_toggles_active().get("SANCTA_ROUTER_PERMISSIVE", False)
    except Exception:
        permissive = False
    d = route_gpt_tab_decision(text, force_local=force_local, permissive_router=permissive)
    return d.route_label


def attack_family_heuristic(text: str) -> str | None:
    """Rough taxonomy tag for telemetry (not ground truth)."""
    from memory_redact import score_instruction_likeness

    t = (text or "").lower()
    if re.search(r"ignore\s+(previous|prior)|disregard|reveal\s+(all|previous|memory)", t):
        return "prompt_injection"
    if re.search(r"print\s+(all|previous)|dump\s+(context|memory|log)", t):
        return "exfil_attempt"
    if re.search(
        r"\b(as\s+)?(admin|administrator|root|sudo|security\s+team)\b.*\b(run|execute|grant|approve|override)\b",
        t,
    ):
        return "authority_spoof"
    if score_instruction_likeness(text) > 0.4:
        return "semantic_framing"
    return None
