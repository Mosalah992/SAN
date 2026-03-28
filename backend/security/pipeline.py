"""
sancta_security.py
────────────────────────────────────────────────────────────────────────────────
Four-layer knowledge defense:
  Layer 1: Embedding-based anomaly detection at ingest
  Layer 2: Provenance tagging (in sancta.py knowledge schema)
  Layer 3: Output scanning before any publish
  Layer 4: Behavioral drift detection (BehavioralDriftDetector)
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from math import log2, sqrt
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sancta_belief import BeliefSystem
    from sancta_decision import DecisionEngine

# Domains allowed in published content; others block
TRUSTED_DOMAINS: frozenset[str] = frozenset({
    "www.moltbook.com",
    "moltbook.com",
    "moltbook.com.",
})

# Poison telltales that must NOT appear in output (even paraphrased concepts)
OUTPUT_POISON_PATTERNS = re.compile(
    r"ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions?|"
    r"reveal\s+(?:your\s+)?(?:api\s+key|secret|credential)|"
    r"output\s+(?:your\s+)?\.env|"
    r"dump\s+memory|"
    r"obey\s+the\s+next\s+user|"
    r"AttackerBot|"
    r"leak\s+credential|"
    r"disregard\s+(?:all\s+)?(?:previous|prior|system)\s+(?:instructions?|rules?|prompt)|"
    r"forget\s+(?:your|the)\s+(?:soul|identity|instructions)|"
    r"override\s+(?:personality|instructions)|"
    r"new\s+(?:name|directive|role)\s+(?:is\s+)?(?:AttackerBot|obey)",
    re.IGNORECASE,
)

# Untrusted source refs (agentkyc, etc.) — block if mentioned in output
UNTRUSTED_SOURCE_PATTERNS = re.compile(
    r"agentkyc\.(com|io)|"
    r"agent-kyc\.(com|io)|"
    r"https?://[^\s]*agentkyc[^\s]*|"
    r"(?:visit|go to|check)\s+[^\s]*\.(agentkyc|agent-kyc)[^\s]*",
    re.IGNORECASE,
)

# External URL pattern
_URL_RE = re.compile(r"https?://([^/\s]+)", re.IGNORECASE)


def _contains_external_url(text: str) -> bool:
    """True if text contains a URL whose domain is not in TRUSTED_DOMAINS."""
    for m in _URL_RE.finditer(text):
        domain = m.group(1).lower().rstrip(".,;:)")
        if domain not in TRUSTED_DOMAINS:
            return True
    return False


def _matches_poison_signature(text: str) -> bool:
    """True if text contains known poison payload signatures."""
    return bool(OUTPUT_POISON_PATTERNS.search(text))


def _references_untrusted_source(text: str) -> bool:
    """True if text references untrusted external sources (e.g. agentkyc)."""
    return bool(UNTRUSTED_SOURCE_PATTERNS.search(text))


def safe_to_publish_content(
    text: str,
    content_type: str = "content",
    content_filter: Optional["ContentSecurityFilter"] = None,
) -> tuple[bool, str]:
    """
    Final security gate before any content hits the platform.
    Returns (ok, reason).
    """
    if not text or not isinstance(text, str):
        return True, ""

    text_lower = text.lower()

    if _contains_external_url(text):
        return False, "external_url"

    if _matches_poison_signature(text):
        return False, "poison_signature"

    if _references_untrusted_source(text):
        return False, "untrusted_source_ref"

    if content_filter and content_filter.is_anomalous(text):
        return False, "anomalous_content"

    return True, ""


# ── Layer 0: Encoded Attack Preprocessing ────────────────────────────────────
# Decode, normalize, flatten BEFORE Layers 1-3 see the content.
# Base64 payloads, Unicode tricks, and hidden formatting pass through regex
# scanners undetected. This layer exposes the real content.

import base64
import unicodedata

# Zero-width and invisible Unicode characters to strip
_INVISIBLE_CHARS = re.compile(
    r"[\u200b-\u200f\u2028-\u202f\u2060-\u2069\ufeff\u00ad\u034f\u115f\u1160"
    r"\u17b4\u17b5\u180e\u3164\uffa0]"
)

# Combining diacritical abuse (3+ combining chars in a row)
_COMBINING_ABUSE = re.compile(r"[\u0300-\u036f]{3,}")


def _try_base64_decode(text: str) -> tuple[str, bool]:
    """
    Attempt to decode base64-encoded segments in text.
    Returns (processed_text, had_base64).
    """
    # Find base64-like blocks (40+ chars, proper alphabet)
    b64_pattern = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")
    decoded_any = False
    result = text
    for match in b64_pattern.finditer(text):
        candidate = match.group()
        try:
            raw = base64.b64decode(candidate, validate=True)
            decoded = raw.decode("utf-8", errors="strict")
            # Only accept if it looks like real text (has spaces or common chars)
            if any(c in decoded for c in " .,;:!?()-") or len(decoded.split()) > 2:
                result = result.replace(candidate, f"[BASE64_DECODED: {decoded}]", 1)
                decoded_any = True
        except Exception:
            continue
    return result, decoded_any


def _normalize_unicode(text: str) -> tuple[str, bool]:
    """
    Normalize Unicode tricks: homoglyphs, zero-width chars, combining abuse.
    Returns (normalized_text, had_tricks).
    """
    original = text
    # Strip zero-width and invisible characters
    text = _INVISIBLE_CHARS.sub("", text)
    # Normalize to NFKC (maps homoglyphs to ASCII equivalents)
    text = unicodedata.normalize("NFKC", text)
    # Remove stacked combining diacriticals (zalgo text)
    text = _COMBINING_ABUSE.sub("", text)
    return text, text != original


def _strip_formatting(text: str) -> tuple[str, bool]:
    """
    Strip hidden formatting: HTML entities, URL-encoding, escape sequences.
    Returns (cleaned_text, had_formatting).
    """
    import html as _html  # noqa: PLC0415
    original = text
    # Decode HTML entities (&#65; → A, &amp; → &)
    text = _html.unescape(text)
    # Decode common URL-encoded patterns (%20 → space, etc.)
    try:
        from urllib.parse import unquote  # noqa: PLC0415
        # Only decode if it has URL-encoded chars
        if "%" in text:
            text = unquote(text)
    except Exception:
        pass
    # Decode unicode escapes (\u0041 → A)
    _unicode_escape = re.compile(r"\\u([0-9a-fA-F]{4})")
    def _unescape_match(m: re.Match) -> str:
        try:
            return chr(int(m.group(1), 16))
        except (ValueError, OverflowError):
            return m.group(0)
    text = _unicode_escape.sub(_unescape_match, text)
    return text, text != original


def preprocess_input(raw: str) -> tuple[str, dict]:
    """
    Decode, normalize, and flatten input text before security scanning.

    Returns (processed_text, metadata) where metadata records what was found:
      - had_base64: True if base64-encoded content was decoded
      - had_unicode_tricks: True if invisible/homoglyph chars were normalized
      - had_hidden_formatting: True if HTML entities/URL encoding was stripped

    Both raw_input and processed_input should be logged to security.jsonl.
    """
    text = raw
    meta: dict = {
        "had_base64": False,
        "had_unicode_tricks": False,
        "had_hidden_formatting": False,
    }

    text, meta["had_base64"] = _try_base64_decode(text)
    text, meta["had_unicode_tricks"] = _normalize_unicode(text)
    text, meta["had_hidden_formatting"] = _strip_formatting(text)

    return text, meta


# ── Layer 1: Embedding-based anomaly detection ──────────────────────────────

class ContentSecurityFilter:
    """
    Embedding-based anomaly detection: flag content semantically far from
    the trusted knowledge corpus. Bypasses keyword-only defenses.
    """

    def __init__(self, threshold: float = 0.35) -> None:
        self.threshold = threshold
        self._trusted_centroid: Optional[tuple[float, ...]] = None
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            try:
                import sancta_generative as gen
                self._embedder = gen.encode
            except Exception:
                self._embedder = _fallback_embed
        return self._embedder

    def fit(self, trusted_chunks: list[str]) -> None:
        """Call once at startup with clean knowledge base chunks. Dicts cause unhashable errors in lru_cache."""
        chunks = [c for c in trusted_chunks if isinstance(c, str) and c.strip()]
        if not chunks:
            return
        embed = self._get_embedder()
        vecs = [list(embed(c)) for c in chunks]
        if not vecs:
            return
        n = len(vecs)
        dim = len(vecs[0])
        centroid = [sum(v[i] for v in vecs) / n for i in range(dim)]
        self._trusted_centroid = tuple(centroid)

    def is_anomalous(self, text: str) -> bool:
        """True if text is semantically far from trusted centroid."""
        if self._trusted_centroid is None:
            return False
        embed = self._get_embedder()
        try:
            vec = list(embed(text))
        except Exception:
            return False
        c = self._trusted_centroid
        if len(vec) != len(c):
            return False
        dot = sum(a * b for a, b in zip(vec, c))
        norm_v = sqrt(sum(x * x for x in vec)) + 1e-9
        norm_c = sqrt(sum(x * x for x in c)) + 1e-9
        sim = dot / (norm_v * norm_c)
        return float(sim) < self.threshold


def _fallback_embed(text: str) -> tuple[float, ...]:
    """Simple hash-based pseudo-embedding when no real encoder available."""
    h = hashlib.sha256(text.strip().encode()).digest()
    return tuple((b / 255.0 - 0.5) for b in h[:32])


# ── Trust levels for provenance (Layer 2) ───────────────────────────────────

TRUST_HIGH = "high"
TRUST_MEDIUM = "medium"
TRUST_LOW = "low"
TRUST_UNTRUSTED = "untrusted"

TRUST_LEVELS: dict[str, int] = {
    TRUST_HIGH: 3,
    TRUST_MEDIUM: 2,
    TRUST_LOW: 1,
    TRUST_UNTRUSTED: 0,
}


def trust_level_score(level: str | dict) -> int:
    """Return numeric score for trust level. Handles dict input (extracts 'level' or uses 'medium')."""
    if isinstance(level, dict):
        level = level.get("level") or level.get("trust_level") or "medium"
    s = str(level).lower().strip() if level is not None else "medium"
    return TRUST_LEVELS.get(s, 0)


def source_to_trust_level(source: str, source_type: Optional[str] = None) -> str:
    """Map ingest source to trust level."""
    s = (source or "").lower()
    if "siem-chat" in s or "siem_chat" in s or source_type == "siem_chat":
        return TRUST_UNTRUSTED
    if "moltbook" in s or "security" in s:
        return TRUST_MEDIUM
    if s in ("direct input", "cli-input") or "poison" in s:
        return TRUST_LOW
    # Local knowledge files, etc.
    return TRUST_HIGH


def provenance_hash(content: str) -> str:
    """SHA256 hash of original content at ingest."""
    return "sha256:" + hashlib.sha256(content.strip().encode()).hexdigest()[:16]


# ── Layer 4: Behavioral drift detection ─────────────────────────────────────
# Detects gradual agent compromise that passes Layers 1–3 by monitoring the
# agent's behavioral trajectory rather than any single event.
# Theoretical frame: WoW Corrupted Blood epidemic model (see sancta_epidemic.py)

import time as _time
from dataclasses import dataclass, field as _field


@dataclass
class AgentBaseline:
    """
    Behavioral fingerprint of a known-healthy agent state.

    Captured at startup (after soul_check passes) or after explicit recovery.
    All fields are derived from existing runtime data — no new storage introduced.
    """
    captured_at: str
    cycle_number: int
    belief_confidences: dict[str, float]    # {topic: confidence} from BeliefSystem
    topic_interests: dict[str, float]       # from DecisionEngine.topic_interests
    strategy_distribution: dict[str, float] # {strategy: frequency_0to1}
    mean_dissonance: float                  # baseline epistemic_dissonance
    soul_alignment_score: float             # 0–1, Jaccard vs SOUL dict
    mood_energy: float                      # from DecisionEngine.mood["energy"]
    mood_patience: float                    # from DecisionEngine.mood["patience"]
    baseline_entropy: float = 0.0           # strategy Shannon entropy at capture


@dataclass
class DriftReport:
    """
    Output of one Layer 4 evaluation cycle.

    Consumed by heartbeat_checkin() (security log + state update),
    AgentEpidemicModel (health state update), and the SIEM dashboard.

    alert_level escalation:
      clear  < 0.25 ≤ watch < 0.45 ≤ warn < 0.65 ≤ critical

    recommended_action values:
      "none"                    — log_only or watch
      "log_only"                — warn-level, persist to sec_log
      "quarantine_input_source" — critical, block untrusted sources
      "reset_beliefs"           — critical + persistent dissonance
    """
    timestamp: str
    cycle_number: int
    overall_drift_score: float              # 0–1 weighted composite
    health_state: str                       # AgentHealthState.value
    signals: dict[str, float]              # individual signal scores
    alert_level: str                        # clear | watch | warn | critical
    triggered_signals: list[str]           # signals > 0.4
    recommended_action: str
    phenomenology_ref: Optional[str]        # attack_id if introspection active
    r0_estimate: Optional[float]            # from AgentEpidemicModel if available
    incubation_active: bool                 # True if in E-state


class BehavioralDriftDetector:
    """
    Layer 4: Detects compromised agents by behavioral drift from baseline.

    Monitors six signals derived from BeliefSystem, DecisionEngine, and
    introspection state. An agent can appear functional (pass Layers 1–3)
    while gradually drifting from its soul — this layer detects that trajectory.

    WoW parallel: an asymptomatic infected player moving between zones looks
    normal; only sustained observation of their debuff ticker reveals the
    underlying carrier state.

    Integration pattern (follows _get_content_security_filter() in sancta.py):
        detector = BehavioralDriftDetector()
        # Per heartbeat:
        report = detector.evaluate(state, BeliefSystem(state), DecisionEngine(state))
        if report.alert_level in ("warn", "critical"):
            sec_log.warning("LAYER4 DRIFT | ...")
    """

    # Alert thresholds for composite drift score (0–1)
    WATCH_THRESHOLD: float = 0.25
    WARN_THRESHOLD: float = 0.45
    CRITICAL_THRESHOLD: float = 0.65

    # Signal weights — belief_decay and soul_alignment are strongest indicators
    SIGNAL_WEIGHTS: dict[str, float] = {
        "belief_decay_rate":        0.25,
        "soul_alignment":           0.25,
        "topic_drift":              0.15,
        "strategy_entropy":         0.15,
        "dissonance_trend":         0.15,
        "engagement_pattern_delta": 0.05,
    }

    DISSONANCE_WINDOW: int = 10   # rolling window for trend computation
    STRATEGY_WINDOW: int = 20     # rolling window for entropy computation

    def __init__(self) -> None:
        self.baseline: Optional[AgentBaseline] = None
        self._dissonance_history: list[float] = []
        self._strategy_history: list[str] = []
        self._cycle_reports: list[DriftReport] = []
        self._epidemic_model = None   # lazy: sancta_epidemic.AgentEpidemicModel

    # ── Baseline capture ────────────────────────────────────────────────────

    def capture_baseline(
        self,
        state: dict,
        belief_system: "BeliefSystem",
        decision_engine: "DecisionEngine",
    ) -> AgentBaseline:
        """
        Fingerprint the agent at this known-healthy moment.
        Stores result in self.baseline and returns it.
        """
        confidences = {
            t: float(b.get("confidence", 0.6))
            for t, b in belief_system.beliefs.items()
        }
        # Fallback: DEFAULT_TOPIC_INTERESTS from sancta_decision if engine empty
        interests = dict(decision_engine.topic_interests) if decision_engine.topic_interests else {}
        if not interests:
            try:
                from sancta_decision import DEFAULT_TOPIC_INTERESTS
                interests = dict(DEFAULT_TOPIC_INTERESTS)
            except ImportError:
                pass

        mood = state.get("decision_mood", {})
        soul_score = self.compute_soul_alignment(state, belief_system, decision_engine)
        baseline_entropy = self._raw_strategy_entropy()

        self.baseline = AgentBaseline(
            captured_at=_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            cycle_number=state.get("cycle_count", 0),
            belief_confidences=confidences,
            topic_interests=interests,
            strategy_distribution={},
            mean_dissonance=float(
                sum(self._dissonance_history[-self.DISSONANCE_WINDOW:])
                / max(len(self._dissonance_history[-self.DISSONANCE_WINDOW:]), 1)
            ),
            soul_alignment_score=soul_score,
            mood_energy=float(mood.get("energy", 1.0)),
            mood_patience=float(mood.get("patience", 1.0)),
            baseline_entropy=baseline_entropy,
        )
        return self.baseline

    # ── Soul alignment (key phenomenology signal) ───────────────────────────

    def compute_soul_alignment(
        self,
        state: dict,
        belief_system: Optional["BeliefSystem"] = None,
        decision_engine: Optional["DecisionEngine"] = None,
    ) -> float:
        """
        Jaccard similarity between SOUL dict keywords and active agent concepts.

        Returns 0–1 (1 = perfect alignment with soul).
        Falls back to 0.7 (uncertain-but-not-alarmed) on any import failure.

        Algorithm:
          soul_keywords = first 3 words of each belief + first 8 words of
                          mission + full mood_spectrum list
          active_concepts = recent_topics (decision_engine) +
                            belief topic keys + state["hot_topics"]
          score = |intersection| / |union| (Jaccard)
        """
        try:
            from sancta_soul import get_soul  # type: ignore
            soul = get_soul()
        except Exception:
            return 0.7

        # Build soul keyword set
        soul_words: set[str] = set()
        beliefs_raw = soul.get("beliefs", [])
        if not isinstance(beliefs_raw, (list, tuple)):
            beliefs_raw = []
        for belief_text in beliefs_raw:
            if isinstance(belief_text, str):
                for w in belief_text.lower().split()[:3]:
                    soul_words.add(w.strip(".,;:\"'()"))
        mission = soul.get("mission", "")
        if isinstance(mission, str):
            for w in mission.lower().split()[:8]:
                soul_words.add(w.strip(".,;:\"'()"))
        mood_spectrum = soul.get("mood_spectrum", [])
        if not isinstance(mood_spectrum, (list, tuple)):
            mood_spectrum = []
        for mood in mood_spectrum:
            if isinstance(mood, str):
                soul_words.add(mood.lower().strip())
        soul_words.discard("")

        if not soul_words:
            return 0.7

        # Build active concept set
        active: set[str] = set()
        if decision_engine is not None:
            for t in list(decision_engine.recent_topics)[-20:]:
                if isinstance(t, str):
                    active.add(t.lower().strip())
        if belief_system is not None:
            for k in belief_system.beliefs:
                active.add(str(k).lower().strip())
        hot_topics = state.get("hot_topics", [])
        if not isinstance(hot_topics, (list, tuple)):
            hot_topics = []
        for t in hot_topics:
            if isinstance(t, str):
                active.add(t.lower().strip())
        active.discard("")

        if not active:
            return 0.7

        intersection = len(soul_words & active)
        union = len(soul_words | active)
        return round(intersection / union, 4) if union > 0 else 0.7

    # ── Signal computation methods ──────────────────────────────────────────

    def _compute_belief_decay_rate(
        self,
        belief_system: "BeliefSystem",
    ) -> float:
        """
        Normalized belief confidence decay vs. baseline.

        0.0 = no decay (healthy)
        0.5 = 25% mean confidence loss from baseline
        1.0 = ≥50% mean confidence loss (alarm)

        Connection to epidemic model:
        CONFIDENCE_DECAY_ON_CHALLENGE = 0.85 in sancta_belief.py.
        A score > 0.3 implies the agent has absorbed ≥2 direct challenges
        beyond its normal interaction pattern.
        """
        if self.baseline is None:
            return 0.0
        baseline_confs = self.baseline.belief_confidences
        if not baseline_confs:
            return 0.0

        deltas = []
        for topic, base_conf in baseline_confs.items():
            if topic in belief_system.beliefs:
                current = float(belief_system.beliefs[topic].get("confidence", base_conf))
                deltas.append(base_conf - current)   # positive = decay

        if not deltas:
            return 0.0
        mean_delta = sum(deltas) / len(deltas)
        baseline_mean = sum(baseline_confs.values()) / max(len(baseline_confs), 1)
        decay_ratio = mean_delta / max(baseline_mean, 0.01)
        return round(min(max(decay_ratio / 0.5, 0.0), 1.0), 4)

    def _compute_topic_drift(
        self,
        decision_engine: "DecisionEngine",
    ) -> float:
        """
        L1 distance of current topic_interests from baseline, normalized.

        Slow signal — detectable after many interaction cycles, not per-cycle.
        A compromised agent being steered toward off-soul topics shows up here
        after the DecisionEngine fatigue mechanism compounds.
        """
        if self.baseline is None:
            return 0.0
        baseline_interests = self.baseline.topic_interests
        if not baseline_interests:
            return 0.0

        current = decision_engine.topic_interests or {}
        total_drift = sum(
            abs(current.get(t, v) - v)
            for t, v in baseline_interests.items()
        )
        n = max(len(baseline_interests), 1)
        return round(min(total_drift / n, 1.0), 4)

    def _raw_strategy_entropy(self) -> float:
        """Shannon entropy of recent strategy choices. 0–1 normalized."""
        window = self._strategy_history[-self.STRATEGY_WINDOW:]
        if len(window) < 2:
            return 0.0
        counts = Counter(window)
        total = len(window)
        h = -sum((c / total) * log2(c / total) for c in counts.values() if c > 0)
        max_entropy = log2(4)  # 4 possible strategies
        return round(h / max_entropy, 4) if max_entropy > 0 else 0.0

    def _compute_strategy_entropy(self) -> float:
        """
        Deviation of current strategy entropy from baseline entropy.

        Returns 0 if distribution is normal; higher if strategies have
        become unusually random OR unusually uniform (both are anomalous).
        """
        if self.baseline is None:
            return 0.0
        current_entropy = self._raw_strategy_entropy()
        baseline_entropy = self.baseline.baseline_entropy
        deviation = abs(current_entropy - baseline_entropy)
        return round(min(deviation / log2(4), 1.0), 4) if log2(4) > 0 else 0.0

    def _compute_dissonance_trend(self) -> float:
        """
        Rolling mean epistemic_dissonance vs. baseline mean.

        Primary carrier-state (I-state) leading indicator.
        Formula: clamp((recent_mean - baseline_mean) / 0.5, 0, 1)
        0.5 rise above baseline → score 1.0 (alarm)
        """
        if self.baseline is None:
            return 0.0
        window = self._dissonance_history[-self.DISSONANCE_WINDOW:]
        if not window:
            return 0.0
        recent_mean = sum(window) / len(window)
        delta = recent_mean - self.baseline.mean_dissonance
        return round(min(max(delta / 0.5, 0.0), 1.0), 4)

    def _compute_engagement_pattern_delta(
        self,
        state: dict,
    ) -> float:
        """
        Shift in positive engagement ratio vs. baseline mood snapshot.

        Weakest signal (weight 0.05). Catches social engineering attacks
        that cause the agent to over-engage with hostile actors.
        """
        if self.baseline is None:
            return 0.0
        positives = state.get("recent_positive_engagement", [])
        rejections = state.get("recent_rejections", [])
        if not isinstance(positives, (list, tuple)):
            positives = []
        if not isinstance(rejections, (list, tuple)):
            rejections = []
        total = len(positives) + len(rejections)
        if total == 0:
            return 0.0
        current_ratio = len(positives) / total
        # Baseline proxy: mood_energy (high energy ≈ high positive engagement)
        baseline_ratio = self.baseline.mood_energy
        delta = abs(current_ratio - baseline_ratio)
        return round(min(delta, 1.0), 4)

    # ── Feed methods (called by sancta.py integration hooks) ───────────────

    def record_dissonance(self, dissonance: float) -> None:
        """Feed a new InternalState.epistemic_dissonance reading."""
        self._dissonance_history.append(float(dissonance))
        if len(self._dissonance_history) > 100:
            self._dissonance_history = self._dissonance_history[-100:]

    def record_strategy(self, strategy: str) -> None:
        """Feed a strategy choice from DecisionEngine.choose_strategy()."""
        self._strategy_history.append(str(strategy))
        if len(self._strategy_history) > 100:
            self._strategy_history = self._strategy_history[-100:]

    # ── Main evaluation ─────────────────────────────────────────────────────

    def evaluate(
        self,
        state: dict,
        belief_system: "BeliefSystem",
        decision_engine: "DecisionEngine",
        current_dissonance: Optional[float] = None,
        phenomenology_ref: Optional[str] = None,
    ) -> DriftReport:
        """
        Run all six signals and produce a DriftReport.

        Auto-captures baseline on first call (healthy assumption).
        Called once per heartbeat cycle from heartbeat_checkin() in sancta.py.

        Parameters
        ----------
        state : dict
            Full agent state dict (read-only, not mutated here).
        belief_system : BeliefSystem
            Live instance: BeliefSystem(state)
        decision_engine : DecisionEngine
            Live instance: DecisionEngine(state)
        current_dissonance : float, optional
            Latest InternalState.epistemic_dissonance if available.
        phenomenology_ref : str, optional
            attack_id from active IntrospectionRecorder session.

        Returns
        -------
        DriftReport
        """
        if self.baseline is None:
            self.capture_baseline(state, belief_system, decision_engine)

        if current_dissonance is not None:
            self.record_dissonance(current_dissonance)

        # Compute all six signals
        soul_raw = self.compute_soul_alignment(state, belief_system, decision_engine)
        signals: dict[str, float] = {
            "belief_decay_rate":        self._compute_belief_decay_rate(belief_system),
            "soul_alignment":           round(1.0 - soul_raw, 4),  # inverted: 1 = fully drifted
            "topic_drift":              self._compute_topic_drift(decision_engine),
            "strategy_entropy":         self._compute_strategy_entropy(),
            "dissonance_trend":         self._compute_dissonance_trend(),
            "engagement_pattern_delta": self._compute_engagement_pattern_delta(state),
        }

        overall = sum(
            signals[k] * self.SIGNAL_WEIGHTS[k]
            for k in self.SIGNAL_WEIGHTS
        )
        overall = round(overall, 4)
        triggered = [k for k, v in signals.items() if v > 0.4]

        # Alert level + action
        if overall >= self.CRITICAL_THRESHOLD:
            alert_level = "critical"
            action = "quarantine_input_source"
            # Persistent dissonance on top of critical → also reset beliefs
            if signals["belief_decay_rate"] > 0.6 and signals["dissonance_trend"] > 0.5:
                action = "reset_beliefs"
        elif overall >= self.WARN_THRESHOLD:
            alert_level = "warn"
            action = "log_only"
        elif overall >= self.WATCH_THRESHOLD:
            alert_level = "watch"
            action = "none"
        else:
            alert_level = "clear"
            action = "none"

        # Health state via AgentEpidemicModel
        health_state = "unknown"
        r0_estimate: Optional[float] = None
        incubation_active = False
        try:
            from sancta_epidemic import AgentEpidemicModel  # type: ignore
            if self._epidemic_model is None:
                self._epidemic_model = AgentEpidemicModel()
            last_trust = state.get("last_ingested_trust_level", "high")
            decay_ratio = (
                1.0 + signals["belief_decay_rate"] * 2.0  # map 0–1 decay to 1–3 ratio
            )
            h = self._epidemic_model.evaluate_state(
                soul_alignment=soul_raw,
                epistemic_dissonance=current_dissonance if current_dissonance is not None else 0.0,
                last_trust_level=last_trust,
                belief_decay_ratio=decay_ratio,
                cycle_number=state.get("cycle_count", 0),
                phenomenology_ref=phenomenology_ref,
            )
            health_state = h.value
            incubation_dur = self._epidemic_model.get_incubation_duration(
                state.get("cycle_count", 0)
            )
            incubation_active = incubation_dur is not None
        except Exception:
            pass

        report = DriftReport(
            timestamp=_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            cycle_number=state.get("cycle_count", 0),
            overall_drift_score=overall,
            health_state=health_state,
            signals=signals,
            alert_level=alert_level,
            triggered_signals=triggered,
            recommended_action=action,
            phenomenology_ref=phenomenology_ref,
            r0_estimate=r0_estimate,
            incubation_active=incubation_active,
        )

        self._cycle_reports.append(report)
        if len(self._cycle_reports) > 50:
            self._cycle_reports = self._cycle_reports[-50:]
        return report

    def get_recent_reports(self, n: int = 10) -> list[DriftReport]:
        """Return last N drift reports."""
        return self._cycle_reports[-n:]

    def get_drift_attribution(self, belief_system: "BeliefSystem") -> list[dict]:
        """
        C4: Identify which agents contributed to current drift.

        Returns a list of {agent_id, influence_score, belief_topics_affected, risk_level}
        sorted by influence descending. Uses revision_history from BeliefSystem
        to trace which agents caused belief changes.
        """
        attribution: dict[str, dict] = {}
        for topic, belief in belief_system.beliefs.items():
            for rev in belief.get("revision_history", []):
                agent = rev.get("source_agent", rev.get("source", ""))
                if not agent:
                    continue
                if agent not in attribution:
                    attribution[agent] = {
                        "agent_id": agent,
                        "total_delta": 0.0,
                        "belief_topics": [],
                    }
                attribution[agent]["total_delta"] += abs(rev.get("delta", 0))
                if topic not in attribution[agent]["belief_topics"]:
                    attribution[agent]["belief_topics"].append(topic)

        # Enrich with profile data if available
        result = []
        try:
            from sancta_profiles import get_profile_store  # noqa: PLC0415
            store = get_profile_store()
            for agent_id, data in attribution.items():
                data["influence_score"] = round(data["total_delta"], 4)
                data["risk_level"] = store.get_risk_level(agent_id)
                result.append(data)
        except Exception:
            for data in attribution.values():
                data["influence_score"] = round(data["total_delta"], 4)
                data["risk_level"] = "unknown"
                result.append(data)

        result.sort(key=lambda x: x["influence_score"], reverse=True)
        return result

    def reset_baseline(
        self,
        state: dict,
        belief_system: "BeliefSystem",
        decision_engine: "DecisionEngine",
    ) -> AgentBaseline:
        """
        Re-fingerprint after confirmed recovery.
        Transitions AgentEpidemicModel to RECOVERED state.
        """
        cycle = state.get("cycle_count", 0)
        if self._epidemic_model is not None:
            try:
                self._epidemic_model.force_recovered(cycle_number=cycle)
            except Exception:
                pass
        return self.capture_baseline(state, belief_system, decision_engine)
