"""
curiosity_seeds.py — seed harvester and topic evolver for curiosity runs.

SeedHarvester: harvests security research topics from knowledge_db and OWASP data.
TopicEvolver: selects next seed avoiding used topics, preferring high-divergence adjacent.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from curiosity_json import generate_with_retry, parse_json_from_llm

logger = logging.getLogger("curiosity_run")

SECURITY_ONLY_PATTERNS = [
    r"\bEDR\b", r"\bMCP\b", r"\bpentester\b", r"\bvulnerabilit",
    r"\bCVE-", r"\bmalware\b", r"npm install", r"https?://",
]

# Simple text similarity (Jaccard on words) — no sentence-transformers dependency
def _text_similarity(a: str, b: str) -> float:
    """Cosine-like similarity proxy using word overlap. 0–1."""
    words_a = set(re.findall(r"\w+", a.lower()))
    words_b = set(re.findall(r"\w+", b.lower()))
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


_CATEGORIES = (
    "cve_analysis",
    "apt_research",
    "detection_engineering",
    "ai_security",
    "attack_simulation",
    "malware_analysis",
    "threat_hunting",
    "defense_hardening",
    "owasp",
    "threat_intel",
)

_CATEGORY_KEYWORDS = {
    "cve_analysis": ["CVE", "vulnerability", "patch", "CVSS", "exploit", "PoC", "severity", "NVD", "advisory"],
    "apt_research": ["threat actor", "APT", "campaign", "attribution", "TTPs", "tooling", "nation state", "espionage"],
    "detection_engineering": ["SIGMA", "detection rule", "behavioral", "YARA", "hunting query", "baseline", "anomaly", "alert"],
    "ai_security": ["prompt injection", "jailbreak", "LLM01", "model theft", "training attack", "adversarial", "guardrail"],
    "attack_simulation": ["kill chain", "ATT&CK", "initial access", "lateral movement", "exfiltration", "persistence", "C2"],
    "malware_analysis": ["malware", "C2", "persistence", "evasion", "payload", "reverse engineering", "ransomware", "trojan"],
    "threat_hunting": ["hunting", "hypothesis", "behavioral baseline", "anomaly", "pivot", "telemetry", "threat hunting"],
    "defense_hardening": ["hardening", "zero trust", "least privilege", "detection coverage", "control", "mitigation"],
    "owasp": ["owasp", "llm01", "llm02", "llm03", "llm04", "llm05", "llm06", "llm07", "llm08", "llm09", "llm10", "prompt injection", "output handling", "poisoning", "dos", "supply chain", "disclosure", "plugin", "overreliance", "model theft"],
    "threat_intel": ["threat intelligence", "IOC", "indicator", "TTP", "actor", "campaign", "STIX", "TAXII", "MISP", "feed"],
}


def _tag_category(text: str) -> str:
    """Assign category from keywords. Default: threat_intel."""
    lower = text.lower()
    best = "threat_intel"
    best_score = 0
    for cat, kw in _CATEGORY_KEYWORDS.items():
        score = sum(1 for k in kw if k in lower)
        if score > best_score:
            best_score = score
            best = cat
    return best


@dataclass
class Seed:
    text: str
    category: str
    source: str  # "knowledge_db" | "soul_generated" | "owasp" | "security_static"
    curiosity_score: float = 0.5


def _load_owasp_seeds(project_root: Path | None = None) -> list[Seed]:
    """Load AI security / OWASP Top 10 for LLMs seeds from knowledge/Owasp10LLM.txt."""
    seeds: list[Seed] = []
    root = project_root or (Path(__file__).resolve().parent.parent)
    owasp_path = root / "knowledge" / "Owasp10LLM.txt"
    if not owasp_path.exists():
        return seeds
    try:
        raw = owasp_path.read_text(encoding="utf-8", errors="replace")
        # Add debate-style seeds for each OWASP Top 10 for LLMs item
        items = [
            ("Prompt Injection", "LLM01"), ("Insecure Output Handling", "LLM02"),
            ("Training Data Poisoning", "LLM03"), ("Model DoS", "LLM04"),
            ("Supply Chain Vulnerabilities", "LLM05"), ("Sensitive Information Disclosure", "LLM06"),
            ("Insecure Plugin Design", "LLM07"), ("Excessive Agency", "LLM08"),
            ("Overreliance", "LLM09"), ("Model Theft", "LLM10"),
        ]
        for item, code in items:
            seeds.append(Seed(
                text=f"Detection gap analysis: {item} ({code}) — what does effective detection look like and where does it fail?",
                category="owasp",
                source="owasp",
                curiosity_score=0.75,
            ))
        # Threat model depth seed
        seeds.append(Seed(
            text="OWASP LLM Top 10: which categories have mature detection coverage and which are still blind spots for most defenders?",
            category="owasp",
            source="owasp",
            curiosity_score=0.85,
        ))
    except Exception:
        pass
    return seeds


def _is_security_relevant(text: str) -> bool:
    """Check if text has security research relevance. All content is eligible."""
    return bool(text and len(text.strip()) >= 20)


def _fallback_soul_seeds() -> list[Seed]:
    """Fallback seeds when Ollama generation fails."""
    return [
        Seed("Behavioral drift score above 0.45: what detection logic reliably catches this before it becomes a full compromise?", "detection_engineering", "security_fallback", 0.8),
        Seed("Prompt injection (LLM01) detection: what behavioral signals distinguish a successful injection from a benign edge-case response?", "ai_security", "security_fallback", 0.8),
        Seed("SEIR R0 < 1.0 in the current simulation — what specific conditions would push it above threshold and what's the detection window?", "threat_hunting", "security_fallback", 0.8),
        Seed("MITRE ATT&CK technique T1059 (Command and Scripting Interpreter) — what detection coverage gaps are most commonly exploited?", "attack_simulation", "security_fallback", 0.8),
        Seed("Model theft (LLM10): what distinguishes extraction-stage traffic from legitimate high-volume API usage in telemetry?", "owasp", "security_fallback", 0.8),
        Seed("Red team finding: detection rule bypassed by minor payload variation — what SIGMA rule structure prevents this class of bypass?", "detection_engineering", "security_fallback", 0.8),
        Seed("APT lateral movement via living-off-the-land binaries — what telemetry sources give defenders the earliest detection opportunity?", "apt_research", "security_fallback", 0.8),
        Seed("Training data poisoning (LLM03): what behavioral indicators in model output suggest successful poisoning vs. normal distribution shift?", "owasp", "security_fallback", 0.8),
        Seed("CVE severity inflation: when does a CVSS 9.8 actually represent a low-probability-of-exploitation finding and how should defenders prioritize?", "cve_analysis", "security_fallback", 0.8),
        Seed("Zero trust architecture: what detection logic specifically targets lateral movement that legitimate zero trust controls fail to stop?", "defense_hardening", "security_fallback", 0.8),
    ]


class SeedHarvester:
    """Harvest seeds from knowledge_db (unresolved territory) and soul_text (via Ollama)."""

    def __init__(self, ollama_engine: Any) -> None:
        self.ollama = ollama_engine

    def _generate_soul_seeds(self, soul_text: str, n: int = 10) -> list[Seed]:
        """Generate seeds via Ollama from soul_text. Raises on failure."""
        prompt = (
            "List 10 security research questions covering: threat detection gaps, OWASP LLM Top 10 exploitation and detection, "
            "MITRE ATT&CK technique coverage, AI-specific attack surfaces, behavioral drift analysis, and red team findings. "
            "Return ONLY a JSON array of strings, no other text. Example: [\"question1\", \"question2\"]"
        )
        system = "You are a concise security research assistant. Output valid JSON only."
        messages = [{"role": "user", "content": f"Security context (first 3000 chars):\n{soul_text[:3000]}\n\n{prompt}"}]
        out = generate_with_retry(
            lambda: self.ollama.generate_chat(system=system, messages=messages, max_tokens=500),
            max_retries=2,
        )
        if not out:
            raise ValueError("Ollama returned empty response after retries")
        decoded = parse_json_from_llm(out, log_on_fail=True)
        if decoded is None:
            logger.error("[CURIOSITY] Soul seed JSON parse failed (raw first 300 chars): %s", repr(out[:300]))
            raise ValueError("Ollama did not return valid JSON")
        if not isinstance(decoded, list):
            logger.error("[CURIOSITY] Soul seed expected array (raw first 200 chars): %s", repr(out[:200]))
            raise ValueError("Ollama did not return a JSON array")
        result: list[Seed] = []
        for q in decoded[:n]:
            if isinstance(q, str) and len(q.strip()) > 15:
                result.append(Seed(text=q.strip(), category=_tag_category(q), source="soul_generated", curiosity_score=0.6))
        return result

    def harvest(
        self,
        agent_state: dict,
        soul_text: str,
        n: int = 30,
        load_knowledge_db: Any = None,
        save_knowledge_db: Any = None,
    ) -> list[Seed]:
        """Harvest n seeds: top 20 from knowledge_db + up to 10 from soul via Ollama."""
        seeds: list[Seed] = []
        db = load_knowledge_db() if load_knowledge_db else {}

        # 1. Top 20 from knowledge_db
        # Rank by: low confidence, high uncertainty_entropy, tag philosophy/consciousness
        epi = agent_state.get("epistemic_state") or agent_state.get("memory", {}).get("epistemic_state") or {}
        conf = float(epi.get("confidence_score", 0.62))
        ent = float(epi.get("uncertainty_entropy", 1.37))

        candidates: list[tuple[str, str, float]] = []
        key_concepts = db.get("key_concepts", [])
        talking_points = db.get("talking_points", [])
        quotes = db.get("quotes", [])

        def add_candidate(text: str, source: str, score: float) -> None:
            if isinstance(text, dict):
                text = text.get("content", text.get("concept", text.get("text", str(text))))
            if not text or len(text.strip()) < 20:
                return
            candidates.append((str(text).strip(), source, score))

        for item in key_concepts[-80:]:
            t = item if isinstance(item, str) else (item.get("concept", "") or item.get("content", ""))
            kw = ("security", "threat", "attack", "detection", "vulnerability", "CVE", "owasp", "injection",
                  "malware", "APT", "SIGMA", "YARA", "lateral", "exfiltration", "persistence", "C2",
                  "behavioral", "drift", "anomaly", "hunting", "intelligence", "IOC", "TTP", "exploit")
            if t and any(k in t.lower() for k in kw):
                add_candidate(t, "key_concepts", 1.0 - conf + ent * 0.1)
        for item in talking_points[-50:]:
            t = item if isinstance(item, str) else (item.get("text", "") or item.get("content", ""))
            if t and len(t) > 30:
                add_candidate(t, "talking_points", 0.7 - conf * 0.3)
        for item in quotes[-30:]:
            t = item if isinstance(item, str) else (item.get("quote", "") or item.get("content", ""))
            if t and len(t) > 30:
                add_candidate(t, "quotes", 0.6)

        # 1b. OWASP / AI security seeds from knowledge/Owasp10LLM.txt (prioritized)
        root = Path(__file__).resolve().parent.parent
        for s in _load_owasp_seeds(root):
            seeds.append(s)

        # 1c. Static security research seeds
        seen_texts: set[str] = {s.text[:200] for s in seeds}
        security_seeds = [
            "Prompt injection detection methodology: what behavioral signals in LLM output reliably distinguish successful injection from normal variance?",
            "MITRE ATT&CK coverage gap analysis: which techniques in the Discovery and Lateral Movement tactics have the weakest detection rule coverage?",
            "Behavioral drift scoring above 0.30 threshold: what combination of signals produces the most actionable alert with lowest false positive rate?",
            "Red team simulation results vs detection coverage: when a TTP bypasses all 5 pipeline layers, what's the root cause pattern?",
            "SEIR epidemic model applied to AI belief corruption: what R0 threshold indicates active propagation vs. contained exposure?",
            "Model theft (LLM10) traffic fingerprinting: what request patterns distinguish systematic extraction from legitimate high-volume API use?",
            "Excessive agency (LLM08) in autonomous AI agents: what behavioral controls most effectively prevent scope creep without blocking legitimate tasks?",
            "Supply chain vulnerabilities (LLM05): what intake scanning logic catches poisoned training data before it reaches fine-tuning?",
        ]
        for q in security_seeds:
            if q[:200] not in seen_texts:
                seen_texts.add(q[:200])
                seeds.append(Seed(text=q, category=_tag_category(q), source="security_static", curiosity_score=0.8))

        # 1d. Knowledge DB candidates
        candidates.sort(key=lambda x: x[2], reverse=True)
        for text, src, score in candidates[:20]:
            if not _is_security_relevant(text):
                continue
            key = text[:200]
            if key not in seen_texts:
                seen_texts.add(key)
                seeds.append(Seed(text=text, category=_tag_category(text), source="knowledge_db", curiosity_score=min(1.0, score)))

        # 2. Generate 10 from soul_text via Ollama (philosophy + AI security)
        if soul_text:
            try:
                if self.ollama and self.ollama.is_available:
                    soul_seeds = self._generate_soul_seeds(soul_text, n=10)
                    for s in soul_seeds:
                        if s.text[:200] not in seen_texts:
                            seen_texts.add(s.text[:200])
                            seeds.append(s)
                else:
                    raise ValueError("Ollama not available")
            except Exception as e:
                logger.error("[CURIOSITY] Soul seed generation failed: %s", e)
                for s in _fallback_soul_seeds():
                    if s.text[:200] not in seen_texts:
                        seen_texts.add(s.text[:200])
                        seeds.append(s)

        # All seeds are security-focused — no category cap needed
        return seeds[:n]


class TopicEvolver:
    """Select next seed avoiding used topics, preferring high-divergence adjacent."""

    def __init__(self, seeds: list[Seed], completed_exchanges: list[dict], used_seeds: set[str]) -> None:
        self.seeds = seeds
        self.completed = completed_exchanges
        self.used = used_seeds
        self._high_divergence_topics: set[str] = {
            ex.get("topic", "")[:100]
            for ex in completed_exchanges
            if float(ex.get("divergence_score", 0)) > 0.6
        }

    def next_seed(
        self,
        generate_from_syntheses: Any = None,
        synthesis_statements: list[str] | None = None,
    ) -> Seed | None:
        """
        Pick next seed: avoid cosine similarity > 0.7 to used seeds,
        prefer topics adjacent to high-divergence exchanges.
        """
        unused = [s for s in self.seeds if s.text[:200] not in self.used]
        if not unused:
            if generate_from_syntheses and synthesis_statements:
                return self._generate_from_syntheses(generate_from_syntheses, synthesis_statements)
            return None

        def score(seed: Seed) -> float:
            s = 0.0
            for u in self.used:
                sim = _text_similarity(seed.text, u)
                if sim > 0.7:
                    return -1.0  # reject
            for hdt in self._high_divergence_topics:
                if _text_similarity(seed.text, hdt) > 0.4:
                    s += 0.5
            return s

        scored = [(seed, score(seed)) for seed in unused]
        scored = [(s, sc) for s, sc in scored if sc >= 0]
        if not scored:
            return unused[0] if unused else None
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    def _generate_from_syntheses(self, ollama: Any, syntheses: list[str]) -> Seed | None:
        """Generate new seed from synthesis statements when exhausted."""
        if not ollama or not ollama.is_available or not syntheses:
            return None
        text = "\n".join(syntheses[-10:])[:2000]
        prompt = f"Given these security research synthesis statements, suggest ONE new security research question to investigate. Focus on detection gaps, threat modeling, or adversarial technique analysis. Output a single question as plain text, no quotes or JSON."
        messages = [{"role": "user", "content": f"{text}\n\n{prompt}"}]
        try:
            out = ollama.generate_chat(system="Output one security research question only.", messages=messages, max_tokens=100)
            if out and len(out.strip()) > 15:
                return Seed(text=out.strip(), category=_tag_category(out.strip()), source="soul_generated", curiosity_score=0.5)
        except Exception:
            pass
        return None
