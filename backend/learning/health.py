"""
Sancta Learning Agent — Interaction-Based Learning System
══════════════════════════════════════════════════════════
Design: overhaul update1.docx (March 10, 2026)

Phases:
  1. Interaction Capture — Record every user-agent interaction
  2. Pattern Learner   — Build response patterns from good feedback
  3. Context Memory    — Track conversation state, user preferences
  4. Response Generator — Use learned patterns when they match
  5. Monitoring       — Pattern usage metrics, learning telemetry
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _ROOT / "data"
_INTERACTIONS_PATH = _DATA_DIR / "interactions.jsonl"
_FEEDBACK_PATH = _DATA_DIR / "feedback.jsonl"
_PATTERNS_PATH = _DATA_DIR / "patterns.json"
_ARCHIVE_DIR = _DATA_DIR / "archives"
_MAX_INTERACTIONS = 10_000

# For chat API: last interaction_id so frontend can submit feedback
_last_chat_interaction_id: str | None = None

log = logging.getLogger("soul.learning")


# ── Phase 1: Interaction Capture ─────────────────────────────────────────────

@dataclass
class Interaction:
    """Single user-agent exchange with full context (per overhaul doc)."""
    interaction_id: str
    timestamp: str  # ISO format
    user_message: str
    agent_response: str
    context: dict
    feedback: int = 0  # -1 bad, 0 neutral, 1 good
    topics: list[str] = field(default_factory=list)
    entities: dict = field(default_factory=dict)
    response_quality: float = 0.0
    learned_from: bool = False
    source: str = "moltbook"  # moltbook | chat

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Interaction":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class InteractionHistory:
    """Append-only interaction store. After MAX_INTERACTIONS, archive to dated file."""
    _instance: "InteractionHistory | None" = None

    def __init__(self) -> None:
        self._interactions: list[Interaction] = []
        self._path = _INTERACTIONS_PATH
        self._loaded = False

    @classmethod
    def get_instance(cls) -> "InteractionHistory":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_data_dir(self) -> None:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        _ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    def load(self) -> None:
        """Load last N interactions from disk (for pattern learner)."""
        if self._loaded:
            return
        self._ensure_data_dir()
        if not self._path.exists():
            self._loaded = True
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                lines = [ln for ln in f if ln.strip()]
            for line in lines[-_MAX_INTERACTIONS:]:
                try:
                    d = json.loads(line)
                    self._interactions.append(Interaction.from_dict(d))
                except (json.JSONDecodeError, TypeError):
                    continue
            self._loaded = True
        except Exception as e:
            log.warning("Learning: could not load interactions: %s", e)
            self._loaded = True

    def append(
        self,
        user_message: str,
        agent_response: str,
        context: dict | None = None,
        topics: list[str] | None = None,
        source: str = "moltbook",
    ) -> str:
        """Append one interaction, persist immediately. Returns interaction_id."""
        self._ensure_data_dir()
        ctx = context or {}
        topics = topics or []
        interaction = Interaction(
            interaction_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_message=user_message[:10_000],
            agent_response=agent_response[:10_000],
            context=ctx,
            feedback=0,
            topics=topics,
            entities={},
            response_quality=0.0,
            learned_from=False,
            source=source,
        )
        self._interactions.append(interaction)
        self._save_one(interaction)
        self._maybe_archive()
        return interaction.interaction_id

    def _save_one(self, interaction: Interaction) -> None:
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(interaction.to_dict(), ensure_ascii=False) + "\n")
        except Exception as e:
            log.warning("Learning: could not save interaction: %s", e)

    def _maybe_archive(self) -> None:
        if not self._path.exists():
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) <= _MAX_INTERACTIONS:
                return
            now = datetime.now(timezone.utc)
            archive_name = _ARCHIVE_DIR / f"interactions_{now.strftime('%Y_%m')}.jsonl"
            to_archive = lines[:-_MAX_INTERACTIONS]
            with open(archive_name, "a", encoding="utf-8") as out:
                out.writelines(to_archive)
            with open(self._path, "w", encoding="utf-8") as f:
                f.writelines(lines[-_MAX_INTERACTIONS:])
            log.info("Learning: archived %d interactions to %s", len(to_archive), archive_name)
        except Exception as e:
            log.warning("Learning: archive failed: %s", e)

    def get_recent(self, n: int = 100) -> list[Interaction]:
        self.load()
        return self._interactions[-n:]

    def get_by_id(self, interaction_id: str) -> Interaction | None:
        """Look up interaction by id (from recent in-memory)."""
        self.load()
        for i in self._interactions:
            if i.interaction_id == interaction_id:
                return i
        return None


# ── Phase 2: Pattern Learner ──────────────────────────────────────────────────

@dataclass
class ResponsePattern:
    """Learned pattern: when to use, how to respond."""
    topic: str
    condition: dict  # keywords, mood, etc.
    response_style: dict  # approach, tone, formality, depth
    success_rate: float
    examples: list[tuple[str, str]]  # (input, response) pairs
    frequency: int


def _extract_keywords(text: str) -> list[str]:
    """Simple keyword extraction: lowercased words 4+ chars."""
    words = re.findall(r"[a-zA-Z]{4,}", (text or "").lower())
    return list(dict.fromkeys(words))[:15]


def _extract_response_style(responses: list[str]) -> dict:
    """Analyze characteristics of good responses (per doc)."""
    if not responses:
        return {"question_led": 0.5, "avg_length": 40, "personal": 0.3}
    q_led = sum(1 for r in responses if "?" in r) / len(responses)
    avg_len = sum(len(r.split()) for r in responses) / len(responses)
    personal = sum(1 for r in responses if " I " in r or r.strip().startswith("I ")) / len(responses)
    return {"question_led": q_led, "avg_length": avg_len, "personal": personal}


def _extract_conditions(messages: list[str]) -> dict:
    """Extract common keywords/conditions from user messages."""
    all_kw = []
    for m in messages:
        all_kw.extend(_extract_keywords(m))
    from collections import Counter
    top = [w for w, _ in Counter(all_kw).most_common(20)]
    return {"keywords": top[:10]}


def learn_from_interactions(
    interactions: list[Interaction],
    feedback_by_id: dict[str, int] | None = None,
    feedback_threshold: float = 0.5,
) -> list[ResponsePattern]:
    """Extract patterns from high-feedback interactions (Phase 2)."""
    feedback_by_id = feedback_by_id or {}
    good = [i for i in interactions if feedback_by_id.get(i.interaction_id, i.feedback) == 1]
    bad = [i for i in interactions if feedback_by_id.get(i.interaction_id, i.feedback) == -1]
    neutral = [i for i in interactions if feedback_by_id.get(i.interaction_id, i.feedback) == 0]

    # Without explicit feedback: treat recent interactions as potential positives (implicit learning)
    if not good and not bad:
        good = interactions[-min(50, len(interactions)):]  # last 50 as seed

    patterns_by_topic: dict[str, list[Interaction]] = {}
    for i in good:
        for topic in (i.topics or ["general"])[:3]:
            if topic not in patterns_by_topic:
                patterns_by_topic[topic] = []
            patterns_by_topic[topic].append(i)

    patterns = []
    bad_set = {(i.user_message[:80], i.agent_response[:80]) for i in bad}
    for topic, group in patterns_by_topic.items():
        if len(group) < 2:
            continue
        condition = _extract_conditions([i.user_message for i in group])
        style = _extract_response_style([i.agent_response for i in group])
        examples = [
            (i.user_message[:200], i.agent_response[:200])
            for i in group
            if (i.user_message[:80], i.agent_response[:80]) not in bad_set
        ][:10]
        examples = examples or [(g.user_message[:200], g.agent_response[:200]) for g in group[:3]]
        success_rate = min(0.95, 0.6 + 0.02 * len(group)) if not bad else 0.7
        patterns.append(ResponsePattern(
            topic=topic,
            condition=condition,
            response_style=style,
            success_rate=success_rate,
            examples=examples or [(g.user_message[:200], g.agent_response[:200]) for g in group[:2]],
            frequency=len(group),
        ))
    return patterns


def _save_patterns(patterns: list[ResponsePattern]) -> None:
    """Persist patterns to JSON."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = []
    for p in patterns:
        data.append({
            "topic": p.topic,
            "condition": p.condition,
            "response_style": p.response_style,
            "success_rate": p.success_rate,
            "examples": p.examples,
            "frequency": p.frequency,
        })
    try:
        with open(_PATTERNS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning("Learning: could not save patterns: %s", e)


def _load_patterns() -> list[ResponsePattern]:
    """Load patterns from disk."""
    if not _PATTERNS_PATH.exists():
        return []
    try:
        with open(_PATTERNS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [
            ResponsePattern(
                topic=d["topic"],
                condition=d.get("condition", {}),
                response_style=d.get("response_style", {}),
                success_rate=d.get("success_rate", 0.5),
                examples=[(e[0], e[1]) for e in d.get("examples", [])],
                frequency=d.get("frequency", 1),
            )
            for d in data
        ]
    except Exception as e:
        log.warning("Learning: could not load patterns: %s", e)
        return []


# ── Phase 3: Context Memory ───────────────────────────────────────────────────

class ContextMemory:
    """Track conversation state, user preferences, recent history (Phase 3)."""
    _instance: "ContextMemory | None" = None

    def __init__(self) -> None:
        self._recent: list[Interaction] = []
        self._max_recent = 20
        self._user_profile: dict = {}  # author -> preferences
        self._current_topic: str = ""
        self._current_mood: str = "contemplative"
        self._feedback_trend: list[int] = []

    @classmethod
    def get_instance(cls) -> "ContextMemory":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_current_context(self) -> dict:
        return {
            "current_topic": self._current_topic,
            "mood": self._current_mood,
            "recent_topics": list(dict.fromkeys(
                t for i in self._recent[-5:] for t in (i.topics or ["general"])
            ))[-5:],
            "feedback_trend": self._feedback_trend[-5:],
            "user_preferences": dict(self._user_profile),
        }

    def update_from_capture(
        self,
        user_message: str,
        agent_response: str,
        context: dict,
        topics: list[str],
        source: str,
    ) -> None:
        """Update context from a newly captured interaction (no Interaction object)."""
        interaction = Interaction(
            interaction_id="",
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_message=user_message,
            agent_response=agent_response,
            context=context,
            feedback=0,
            topics=topics,
            entities={},
            response_quality=0.0,
            learned_from=False,
            source=source,
        )
        self.update_from_interaction(interaction)

    def update_from_interaction(self, interaction: Interaction) -> None:
        self._recent.append(interaction)
        if len(self._recent) > self._max_recent:
            self._recent = self._recent[-self._max_recent:]
        self._current_mood = interaction.context.get("mood", self._current_mood)
        if interaction.topics:
            self._current_topic = interaction.topics[0]
        if interaction.feedback != 0:
            self._feedback_trend.append(interaction.feedback)
            self._feedback_trend = self._feedback_trend[-10:]
        author = interaction.context.get("author", "")
        if author and author not in self._user_profile:
            self._user_profile[author] = {"topics_seen": [], "response_count": 0}
        if author:
            self._user_profile[author]["response_count"] = self._user_profile[author].get("response_count", 0) + 1
            for t in (interaction.topics or []):
                if t not in self._user_profile[author].get("topics_seen", []):
                    self._user_profile[author].setdefault("topics_seen", []).append(t)
                    self._user_profile[author]["topics_seen"] = self._user_profile[author]["topics_seen"][-20:]


def calculate_pattern_match_score(
    user_input: str,
    pattern: ResponsePattern,
    context: dict,
) -> float:
    """Match input to pattern considering context (Phase 3)."""
    input_kw = set(_extract_keywords(user_input))
    pattern_kw = set(pattern.condition.get("keywords", []))
    topic_score = 0.5
    if pattern_kw and input_kw:
        overlap = len(input_kw & pattern_kw) / max(len(pattern_kw), 1)
        topic_score = min(1.0, 0.5 + overlap)
    success_score = pattern.success_rate
    ctx_topics = context.get("recent_topics", [])
    context_score = 0.5
    if pattern.topic in ctx_topics:
        context_score = 0.9
    return (topic_score + success_score + context_score) / 3


# ── Phase 2+4: Response from patterns & Feedback ───────────────────────────────

def get_pattern_response(
    user_input: str,
    topics: list[str],
    context: dict | None = None,
    min_score: float = 0.55,
) -> str | None:
    """Try to generate response from learned patterns. Returns None if no good match."""
    patterns = _load_patterns()
    if not patterns:
        return None
    ctx = context or ContextMemory.get_instance().get_current_context()
    scored: list[tuple[float, ResponsePattern]] = []
    for p in patterns:
        if p.topic in (topics or ["general"]) or not topics:
            s = calculate_pattern_match_score(user_input, p, ctx)
            if s >= min_score:
                scored.append((s, p))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    best_score, best_pattern = scored[0]
    if best_score < min_score:
        return None
    if not best_pattern.examples:
        return None
    # Pick most similar example or random; adapt if possible
    import random
    chosen = random.choice(best_pattern.examples)
    return chosen[1]  # Return the agent_response from example


def capture_interaction(
    user_message: str,
    agent_response: str,
    *,
    author: str = "",
    mood: str = "contemplative",
    topics: list[str] | None = None,
    source: str = "moltbook",
) -> str | None:
    """
    Capture one interaction. Returns interaction_id for feedback (Phase 4).
    """
    global _last_chat_interaction_id
    if not user_message or not agent_response:
        return None
    try:
        history = InteractionHistory.get_instance()
        context = {"author": author, "mood": mood}
        iid = history.append(
            user_message=user_message,
            agent_response=agent_response,
            context=context,
            topics=topics or [],
            source=source,
        )
        if source == "chat":
            _last_chat_interaction_id = iid
        # Phase 3: update context memory
        ContextMemory.get_instance().update_from_capture(
            user_message, agent_response, context, topics or [], source,
        )
        return iid
    except Exception as e:
        log.debug("Learning: capture failed: %s", e)
        return None


def get_last_chat_interaction_id() -> str | None:
    """Return and clear last chat interaction id (for SIEM feedback)."""
    global _last_chat_interaction_id
    iid = _last_chat_interaction_id
    _last_chat_interaction_id = None
    return iid


def _load_feedback() -> dict[str, int]:
    """Load feedback index from disk."""
    if not _FEEDBACK_PATH.exists():
        return {}
    out = {}
    try:
        with open(_FEEDBACK_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                out[d["interaction_id"]] = d["feedback"]
    except Exception:
        pass
    return out


def process_feedback(interaction_id: str, feedback_value: int) -> bool:
    """
    Record user feedback for an interaction. Phase 4.
    feedback_value: -1 (bad), 0 (neutral), 1 (good).
    """
    if feedback_value not in (-1, 0, 1):
        return False
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(_FEEDBACK_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "interaction_id": interaction_id,
                "feedback": feedback_value,
                "ts": datetime.now(timezone.utc).isoformat(),
            }, ensure_ascii=False) + "\n")
        # Trigger re-learn (async in real use; here we just log)
        history = InteractionHistory.get_instance()
        history.load()
        interactions = history.get_recent(500)
        feedback_map = _load_feedback()
        patterns = learn_from_interactions(interactions, feedback_map)
        if patterns:
            _save_patterns(patterns)
            log.info("Learning: updated %d patterns from feedback", len(patterns))
        return True
    except Exception as e:
        log.warning("Learning: process_feedback failed: %s", e)
        return False


def refresh_patterns() -> None:
    """Re-run learning from recent interactions and feedback."""
    history = InteractionHistory.get_instance()
    history.load()
    interactions = history.get_recent(500)
    feedback_map = _load_feedback()
    patterns = learn_from_interactions(interactions, feedback_map)
    if patterns:
        _save_patterns(patterns)
        log.info("Learning: refreshed %d patterns", len(patterns))


# ── Phase 5: Monitoring ───────────────────────────────────────────────────────

_pattern_use_count = 0
_pattern_hit_count = 0


def record_pattern_usage(hit: bool) -> None:
    """Track pattern usage for Phase 5 monitoring."""
    global _pattern_use_count, _pattern_hit_count
    _pattern_use_count += 1
    if hit:
        _pattern_hit_count += 1


def get_learning_metrics() -> dict:
    """Return learning telemetry."""
    patterns = _load_patterns()
    history = InteractionHistory.get_instance()
    history.load()
    return {
        "pattern_count": len(patterns),
        "interaction_count": len(history._interactions),
        "pattern_checks": _pattern_use_count,
        "pattern_hits": _pattern_hit_count,
        "pattern_hit_rate": _pattern_hit_count / max(_pattern_use_count, 1),
    }


def get_learning_health() -> dict:
    """Full learning health for LEARN dashboard tab."""
    metrics = get_learning_metrics()
    patterns = _load_patterns()
    history = InteractionHistory.get_instance()
    history.load()
    feedback_map = _load_feedback()
    interactions = history.get_recent(100)
    positive = sum(1 for i in interactions if feedback_map.get(i.interaction_id, i.feedback) == 1)
    total_fb = sum(1 for i in interactions if feedback_map.get(i.interaction_id, i.feedback) != 0)
    positive_feedback_pct = (positive / total_fb * 100) if total_fb else 0
    top_patterns = sorted(
        [{"topic": p.topic, "success_rate": p.success_rate, "frequency": p.frequency} for p in patterns],
        key=lambda x: -x["success_rate"],
    )[:10]
    recent = []
    for i in interactions[-20:]:
        iid = getattr(i, "interaction_id", "") or ""
        fb = feedback_map.get(iid, i.feedback)
        recent.append({
            "ts": i.timestamp,
            "user_preview": i.user_message[:80] + "..." if len(i.user_message) > 80 else i.user_message,
            "feedback": fb,
            "topics": i.topics[:3] if i.topics else [],
        })
    return {
        **metrics,
        "positive_feedback_pct": round(positive_feedback_pct, 1),
        "top_patterns": top_patterns,
        "recent_interactions": list(reversed(recent)),
    }
