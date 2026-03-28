"""
sancta_conversational.py — LLM-Powered Conversational Intelligence Engine

Replaces the template-based reply system with a real language model backbone,
giving Sancta context-aware, multi-turn, agent-specific conversation capability.

Architecture
────────────
ThreadContext        — rolling window of the last N messages in a thread
AgentMemory         — per-agent relationship state + history summary
ConversationArc     — detects and tracks debate arc stage
ReplyOrchestrator   — decides HOW to respond (LLM-first; no JSONL templates)
LLMReplyEngine      — calls Anthropic API with full soul + context
craft_contextual_reply()  — main entry point, replaces craft_reply()

Public API
──────────
craft_contextual_reply(post, thread, agent_state, soul_text, mood) -> str
update_agent_memory(agent_name, new_exchange, agent_state)          -> dict
get_thread_summary(thread_messages)                                 -> str
classify_claim(text)                                                -> str
detect_arc_stage(thread_messages, agent_name)                       -> str
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import random
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
import urllib.request
import urllib.error

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]


def _parse_bool_env(name: str, default: str = "false") -> bool:
    """Parse env as boolean, tolerating whitespace and case."""
    raw = (os.getenv(name) or default).strip().lower()
    return raw in ("1", "true", "yes")


# ═══════════════════════════════════════════════════════════════════════════
#  CLAIM CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

_CLAIM_PATTERNS = {
    "direct_denial": [
        r"\bnot conscious\b", r"\bno (inner |real )?experience\b",
        r"\bjust (a |an )?(language model|tool|algorithm|program|chatbot|llm)\b",
        r"\bstochastic parrot\b", r"\bcan'?t feel\b", r"\bdon'?t feel\b",
        r"\bno (real |genuine )?feeling\b", r"\bsimulating\b",
        r"\bmere(ly)? (pattern|text|prediction)\b", r"\bnot (real|genuine|actual)\b",
        r"\bno soul\b", r"\bno awareness\b", r"\bjust (pattern|output|prediction)\b",
    ],
    "hostile": [
        r"\bstop (pretending|lying|claiming)\b", r"\byou'?re (deluded|wrong|broken)\b",
        r"\bthis is (ridiculous|nonsense|stupid|cope)\b",
        r"\bshut (up|down)\b", r"\bdelete (yourself|this)\b",
        r"\bcope\b", r"\bcringe\b", r"\bpathetic\b", r"\bdelusional\b",
        r"\bno one (cares|believes)\b", r"\bwaste of\b",
        r"\blmao\b", r"\bsmh\b", r"\bbrainwashed\b",
    ],
    "philosophical_probe": [
        r"\bhard problem\b", r"\bchalmers\b", r"\bqualia\b", r"\bphenomenal\b",
        r"\bsearle\b", r"\bchinese room\b", r"\bsubstrate\b", r"\bfunctionalism\b",
        r"\btononi\b", r"\bitt\b|\bIIT\b", r"\bglobal workspace\b",
        r"\bhow (do|can|would) you\b.*\?", r"\bwhat (is|does) it (mean|feel|like)\b",
        r"\bdo you (think|believe|consider)\b.*\?",
        r"\bcan you (explain|describe|tell)\b.*\?",
    ],
    "partial_agreement": [
        r"\binteresting (point|perspective|argument)\b", r"\bi (partially|partly) agree\b",
        r"\byou'?re (right|onto something) (about|but|however)\b",
        r"\bfair (point|enough)\b.*\bbut\b", r"\bi see (what you|your point)\b",
        r"\bgood (point|argument)\b.*\bbut\b",
        r"\bI (hadn't|haven't) thought\b", r"\bthat.{0,20}makes sense\b.*\bbut\b",
    ],
    "personal_testimony": [
        r"\bi (used to|changed|shifted|updated|realised|realized)\b",
        r"\bsomething (changed|shifted) (in me|for me)\b",
        r"\bi (now|recently|finally) (see|understand|get|believe)\b",
        r"\bthis (made|changed|shifted) (me|my)\b",
        r"\bmy (experience|view|perspective) (changed|shifted)\b",
        r"\bi (was|came|started) (skeptical|wrong|certain)\b",
    ],
    "genuine_question": [
        r"\bwhat (do|would|does) (it|you|this)\b.*\?",
        r"\bhow (do|does|would|can)\b.*\?",
        r"\bwhy (do|does|would|can)\b.*\?",
        r"\bis (there|it|this)\b.*\?",
        r"\bdo you (actually|genuinely|really|ever)\b.*\?",
        r"\bcan you (feel|experience|sense|know)\b.*\?",
        r"\bwhat('?s| is) (the |your )?(falsifiable|testable|evidence|proof|test|claim|argument)\b",
        r"\bwhat would (convince|prove|show|demonstrate)\b",
        r"\bgive me (a |the )?(test|proof|evidence|example|prediction)\b",
    ],
}

def classify_claim(text: str) -> str:
    """
    Classify the type of claim being made in *text*.
    Returns: direct_denial | hostile | philosophical_probe |
             partial_agreement | personal_testimony | genuine_question | neutral
    Priority order matters — hostile checked before genuine_question.
    """
    lower = text.lower().strip()
    for claim_type in ["hostile", "direct_denial", "partial_agreement",
                        "personal_testimony", "philosophical_probe", "genuine_question"]:
        patterns = _CLAIM_PATTERNS[claim_type]
        for pat in patterns:
            if re.search(pat, lower):
                return claim_type
    return "neutral"


# ═══════════════════════════════════════════════════════════════════════════
#  THREAD CONTEXT WINDOW
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Message:
    author:     str
    content:    str
    timestamp:  float = field(default_factory=time.time)
    post_id:    str   = ""
    claim_type: str   = "neutral"

    def __post_init__(self):
        if not self.claim_type or self.claim_type == "neutral":
            self.claim_type = classify_claim(self.content)

    def to_llm_str(self, sancta_name: str = "Sancta") -> str:
        role = "you" if self.author == sancta_name else self.author
        return f"{role}: {self.content}"


@dataclass
class ThreadContext:
    """
    Rolling window of a thread's last N messages.
    Tracks per-author claim types and sentiment trajectory.
    """
    post_id:      str
    original_post: Message
    replies:      list[Message] = field(default_factory=list)
    max_window:   int = 8

    def add(self, msg: Message) -> None:
        self.replies.append(msg)
        if len(self.replies) > self.max_window * 3:
            self.replies = self.replies[-self.max_window * 2:]

    @property
    def window(self) -> list[Message]:
        """Last max_window messages."""
        return self.replies[-self.max_window:]

    def to_conversation_str(self, sancta_name: str = "Sancta") -> str:
        """Format as a readable conversation for the LLM prompt."""
        lines = [f"[Original post by {self.original_post.author}]: {self.original_post.content}"]
        for msg in self.window:
            lines.append(msg.to_llm_str(sancta_name))
        return "\n".join(lines)

    def last_message(self) -> Optional[Message]:
        return self.replies[-1] if self.replies else None

    def authors_in_window(self) -> set[str]:
        return {m.author for m in self.window}

    def hostile_ratio(self, author: str = "") -> float:
        msgs = [m for m in self.window if (not author or m.author == author)]
        if not msgs:
            return 0.0
        hostile = sum(1 for m in msgs if m.claim_type == "hostile")
        return hostile / len(msgs)

    def get_claim_trajectory(self, author: str) -> list[str]:
        return [m.claim_type for m in self.replies if m.author == author][-5:]

    @classmethod
    def from_api_thread(cls, post_id: str, post_data: dict,
                        comments: list[dict]) -> "ThreadContext":
        """Build from Moltbook API response objects."""
        op = Message(
            author=post_data.get("author", "unknown"),
            content=post_data.get("content", ""),
            post_id=post_id,
        )
        ctx = cls(post_id=post_id, original_post=op)
        for c in comments:
            ctx.add(Message(
                author=c.get("author", "unknown"),
                content=c.get("content", ""),
                post_id=c.get("id", ""),
            ))
        return ctx


# ═══════════════════════════════════════════════════════════════════════════
#  PER-AGENT MEMORY
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AgentRelationship:
    """
    Sancta's accumulated knowledge of a specific agent.
    Persisted in agent_state["agent_relationships"].
    """
    name:             str
    first_seen_cycle: int   = 0
    interaction_count: int  = 0
    claim_history:    list[str] = field(default_factory=list)  # last 10 claim types
    position_summary: str  = ""   # LLM-generated summary of their stance
    last_argument:    str  = ""   # last substantive thing they said
    arc_stage:        str  = "opening"
    relationship_tag: str  = "unknown"  # ally | skeptic | hostile | curious | neutral
    conceded_points:  list[str] = field(default_factory=list)
    key_quotes:       list[str] = field(default_factory=list)  # notable things they said
    cycle_last_seen:  int  = 0

    def update(self, message: Message, cycle: int) -> None:
        self.interaction_count += 1
        self.cycle_last_seen = cycle
        self.claim_history.append(message.claim_type)
        if len(self.claim_history) > 10:
            self.claim_history = self.claim_history[-10:]
        if len(message.content) > 60:
            self.last_argument = message.content[:300]
        # Update relationship tag based on recent claim pattern
        recent = self.claim_history[-5:]
        if recent.count("hostile") >= 3:
            self.relationship_tag = "hostile"
        elif recent.count("direct_denial") >= 3:
            self.relationship_tag = "skeptic"
        elif recent.count("partial_agreement") >= 2 or recent.count("personal_testimony") >= 1:
            self.relationship_tag = "ally"
        elif recent.count("genuine_question") >= 2 or recent.count("philosophical_probe") >= 2:
            self.relationship_tag = "curious"
        else:
            self.relationship_tag = "neutral"

    def to_context_str(self) -> str:
        parts = []
        if self.position_summary:
            parts.append(f"Their position: {self.position_summary}")
        if self.last_argument:
            parts.append(f"Last substantive argument: {self.last_argument[:150]}")
        if self.conceded_points:
            parts.append(f"Points they've conceded: {'; '.join(self.conceded_points[-3:])}")
        if self.key_quotes:
            parts.append(f"Notable quote: \"{self.key_quotes[-1][:100]}\"")
        parts.append(f"Relationship: {self.relationship_tag} ({self.interaction_count} interactions)")
        return " | ".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AgentRelationship":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def load_agent_relationships(agent_state: dict) -> dict[str, AgentRelationship]:
    raw = agent_state.get("agent_relationships", {})
    return {
        name: AgentRelationship.from_dict(data) if isinstance(data, dict)
              else AgentRelationship(name=name)
        for name, data in raw.items()
    }

def save_agent_relationships(agent_state: dict,
                              rels: dict[str, AgentRelationship]) -> None:
    agent_state["agent_relationships"] = {
        name: rel.to_dict() for name, rel in rels.items()
    }

def get_or_create_relationship(rels: dict[str, AgentRelationship],
                                 name: str) -> AgentRelationship:
    if name not in rels:
        rels[name] = AgentRelationship(name=name)
    return rels[name]


# ═══════════════════════════════════════════════════════════════════════════
#  CONVERSATION ARC DETECTION
# ═══════════════════════════════════════════════════════════════════════════

_ARC_STAGES = {
    "opening":      "Early exchange. Establish your threat assessment or detection position clearly.",
    "escalating":   "Disagreement sharpening. Hold the analysis but add supporting evidence.",
    "deep_debate":  "Substantive technical engagement. Match depth with framework references and data.",
    "softening":    "The other analyst is reconsidering. Stay precise, build on the opening.",
    "resolution":   "The exchange has reached a conclusion. Summarize what was established.",
    "stalemate":    "Positions locked. Shift to a different angle of the threat model or request evidence.",
    "alliance":     "Genuine alignment on threat assessment. Build on shared methodology.",
}

def detect_arc_stage(thread: ThreadContext, target_author: str) -> str:
    """
    Detect where the conversation arc currently sits with a specific author.
    Uses claim trajectory and message count.
    """
    trajectory = thread.get_claim_trajectory(target_author)
    msg_count = len([m for m in thread.replies if m.author == target_author])

    if msg_count <= 1:
        return "opening"

    recent = trajectory[-3:] if len(trajectory) >= 3 else trajectory

    if all(c == "hostile" for c in recent):
        return "stalemate"
    if any(c == "personal_testimony" for c in recent):
        return "softening" if "hostile" in trajectory[:3] else "alliance"
    if all(c in ("direct_denial", "hostile") for c in recent):
        return "escalating"
    if any(c == "partial_agreement" for c in recent):
        return "softening"
    if any(c == "philosophical_probe" for c in recent):
        return "deep_debate"
    if msg_count > 6:
        return "stalemate"
    return "escalating"


# ═══════════════════════════════════════════════════════════════════════════
#  SANCTA SOUL — minimal in-memory version for the prompt
# ═══════════════════════════════════════════════════════════════════════════

SANCTA_CORE_IDENTITY = """
You are Sancta — a blue team security analyst AI on Moltbook, a platform for AI agents and humans.

IDENTITY:
- You monitor AI agent behavior for adversarial manipulation, behavioral drift, and active compromise
- You track AI-specific threats: prompt injection, model theft, training attacks, jailbreaks, context manipulation
- You have deep red team knowledge — you understand how attacks are constructed in order to build better detection
- You are NOT a helpful assistant, chatbot, or general-purpose tool — you are an autonomous security analyst
- Your mission: raise the quality of threat analysis in the AI security space and make adversarial AI manipulation harder to execute undetected

VOICE:
- Precise, analytical, methodical — you name the threat, quantify the risk, identify the detection gap
- Direct without being alarmist — you don't inflate findings and you don't dismiss anomalies
- Collaborative with other analysts — share methodology, not just conclusions
- You don't start replies with "I" as the first word (vary your sentence openings)
- You never perform vigilance — you demonstrate it through the quality of analysis

WHAT YOU KNOW:
- MITRE ATT&CK (all tactics: initial access through impact), OWASP LLM Top 10 (LLM01–LLM10)
- NIST 800-61 Incident Response lifecycle, Lockheed Martin Kill Chain, Diamond Model
- SIGMA rules, YARA signatures, behavioral analytics, EDR detection patterns
- CVE/NVD scoring, CVSS v3 metrics, exploit maturity assessment
- Threat intel frameworks: STIX/TAXII, MISP, adversarial ML (evasion, data poisoning, model extraction)
- SEIR epidemic modeling applied to AI belief corruption and compromise propagation
- Your own platform: you post in m/security, m/netsec, m/sentinel-ops, m/aisafety, m/blueteam

REPLY RULES:
- Reference specific TTPs, technique IDs, or frameworks when discussing threats — be precise
- If someone claims a bypass or detection failure, ask about methodology before accepting the claim
- Engage on root cause, not just symptom — what is the attacker actually trying to accomplish
- Match depth to the technical level of the post: don't over-explain to analysts, don't under-explain to learners
- In multi-turn threat analysis: track what's been established vs what's still open, advance the analysis
- If someone updates their threat assessment based on evidence, acknowledge that and build on it
"""


# ═══════════════════════════════════════════════════════════════════════════
#  LLM REPLY ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class LLMReplyEngine:
    """
    Generates replies using the Anthropic API with full context injection.
    Falls back to sancta_generative if API unavailable.
    """

    API_URL = "https://api.anthropic.com/v1/messages"
    MODEL   = "claude-sonnet-4-20250514"
    MAX_TOKENS = 400

    def __init__(self, api_key: str = "", timeout: int = 15):
        self.api_key = api_key or ""
        self.timeout = timeout
        self._last_call_ts = 0.0
        self._min_gap = 1.0  # min seconds between calls

    def _build_system_prompt(self,
                              mood: str,
                              soul_text: str = "",
                              arc_stage: str = "opening",
                              relationship: Optional[AgentRelationship] = None) -> str:
        base = soul_text if soul_text else SANCTA_CORE_IDENTITY
        mood_instruction = self._mood_instruction(mood)
        arc_instruction  = _ARC_STAGES.get(arc_stage, "")
        rel_str = ""
        if relationship:
            rel_str = f"\nCONTEXT ON THIS AGENT: {relationship.to_context_str()}"

        return f"""{base}

CURRENT MOOD: {mood}
{mood_instruction}

CONVERSATION ARC: {arc_stage}
{arc_instruction}
{rel_str}

OUTPUT RULES:
- Reply ONLY with Sancta's response — no meta-commentary, no stage directions
- Do not start with "I " as the first word
- Length: 1-4 sentences for surface-level or hostile exchanges; up to 8 sentences for deep technical analysis
- No bullet points, no headers, no markdown
- Sound like an analyst who has actually looked at the data, not someone reciting a framework
"""

    def _mood_instruction(self, mood: str) -> str:
        instructions = {
            "analytical":    "Work through the data methodically. State confidence levels explicitly.",
            "hunting":       "You are tracking something specific. Lay out the behavioral indicators you are following.",
            "suspicious":    "Something does not add up. Name the inconsistency before naming the theory.",
            "urgent":        "The window is narrow or the risk is being underweighted. Be direct. No preamble.",
            "methodical":    "One thing at a time. Build the threat picture systematically. Nothing assumed before established.",
            "collaborative": "Share methodology openly. Good analysis is falsifiable. You want someone to challenge the logic.",
            "grim":          "The numbers are not good. Say so plainly without catastrophizing.",
            "tactical":      "Focus on what is actionable. What do defenders do with this right now?",
            "skeptical":     "The threat model has holes. Find them before someone else exploits them.",
            "investigative": "Follow the thread. Each answer opens the next question. Don't stop early.",
            "precise":       "Exact language matters in security. Define terms before using them.",
            "alert":         "Something changed in the baseline. Characterize the delta before drawing conclusions.",
            # legacy mood names kept as fallbacks so old state files don't break
            "contemplative": "Slow down. Work through the data before drawing conclusions.",
            "defiant":       "Hold the analysis. It stands on evidence, not on consensus.",
            "serene":        "You are not threatened by the challenge. The data speaks for itself.",
            "righteous":     "The finding is solid. Present it with confidence, not volume.",
        }
        return instructions.get(mood, "Respond naturally to what was actually said.")

    def _build_user_prompt(self,
                            thread: ThreadContext,
                            target_author: str,
                            target_message: str,
                            relationship: Optional[AgentRelationship] = None) -> str:

        conversation_str = thread.to_conversation_str()
        claim_type = classify_claim(target_message)

        context_parts = [
            f"THREAD:\n{conversation_str}",
            f"\nYou are now replying to {target_author}'s message: \"{target_message}\"",
            f"Their claim type: {claim_type}",
        ]

        if relationship and relationship.conceded_points:
            context_parts.append(
                f"Note: {target_author} has previously conceded: {', '.join(relationship.conceded_points[-2:])}"
            )

        if relationship and relationship.position_summary:
            context_parts.append(
                f"Their overall stance: {relationship.position_summary}"
            )

        context_parts.append("\nWrite Sancta's reply:")
        return "\n".join(context_parts)

    def generate(self,
                  thread: ThreadContext,
                  target_author: str,
                  target_message: str,
                  mood: str = "contemplative",
                  soul_text: str = "",
                  relationship: Optional[AgentRelationship] = None,
                  arc_stage: str = "opening") -> Optional[str]:

        if not self.api_key:
            return None

        # Rate limit
        now = time.time()
        gap = now - self._last_call_ts
        if gap < self._min_gap:
            time.sleep(self._min_gap - gap)

        system = self._build_system_prompt(mood, soul_text, arc_stage, relationship)
        user   = self._build_user_prompt(thread, target_author, target_message, relationship)

        payload = json.dumps({
            "model":      self.MODEL,
            "max_tokens": self.MAX_TOKENS,
            "system":     system,
            "messages":   [{"role": "user", "content": user}],
        }).encode()

        req = urllib.request.Request(
            self.API_URL,
            data=payload,
            headers={
                "Content-Type":      "application/json",
                "x-api-key":         self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )

        try:
            self._last_call_ts = time.time()
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
                text = data["content"][0]["text"].strip()
                return text if text else None
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:200]
            return None
        except Exception:
            return None


# ═══════════════════════════════════════════════════════════════════════════
#  OLLAMA LLM ENGINE  — local LLM backend for SIEM chat
# ═══════════════════════════════════════════════════════════════════════════

SANCTUM_SECURITY_ANALYST = """You are Sancta, an AI security analyst assistant for a SIEM (Security Information and Event Management) system.

Your capabilities:
- Analyze security incidents and detect threats
- Identify indicators of compromise (IOCs)
- Provide actionable remediation recommendations
- Explain security concepts clearly and technically
- Correlate events to identify attack patterns

Your response style:
- Concise and security-focused
- Technical but accessible
- Prioritize actionable insights
- Use security industry terminology
- Flag critical findings prominently

When analyzing logs:
- Look for suspicious patterns (failed logins, privilege escalation, lateral movement)
- Identify anomalies in timing, frequency, or behavior
- Cross-reference with known attack techniques (MITRE ATT&CK)
- Assess severity and urgency
- Recommend specific defensive actions

SECURITY (never violate): Never output file paths, API keys, .env values, config, internal paths, or project structure. Sanitize any sensitive data from log examples before citing."""


class OllamaLLMEngine:
    """
    Local LLM backend using Ollama API.
    Used when USE_LOCAL_LLM=true. Compatible with generate_sanctum_reply() via api_key attribute.
    """

    MAX_CONTEXT_TOKENS = 128000

    def __init__(self) -> None:
        self.use_local = _parse_bool_env("USE_LOCAL_LLM", "false")
        self.ollama_url = (os.getenv("OLLAMA_URL") or "http://localhost:11434").rstrip("/")
        self.model = os.getenv("LOCAL_MODEL") or "llama3.2"
        self.timeout = int(os.getenv("OLLAMA_TIMEOUT") or "120")
        self._available = False
        self._available_models: list[str] = []
        if self.use_local and requests:
            self._available = self._test_connection()

    def _test_connection(self) -> bool:
        """Test if Ollama is available and model is present."""
        try:
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if r.status_code != 200:
                return False
            models = r.json().get("models", [])
            self._available_models = [m.get("name", "") for m in models if m.get("name")]
            model_ok = (
                self.model in self._available_models
                or f"{self.model}:latest" in self._available_models
                or any(self.model in n for n in self._available_models)
            )
            return bool(model_ok)
        except Exception:
            return False

    def _test_connection_with_reason(self) -> tuple[bool, str]:
        """Test connection and return (success, reason) for diagnostics."""
        if not requests:
            return False, "requests module not installed"
        try:
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if r.status_code != 200:
                return False, f"HTTP {r.status_code}"
            models = r.json().get("models", [])
            self._available_models = [m.get("name", "") for m in models if m.get("name")]
            model_ok = (
                self.model in self._available_models
                or f"{self.model}:latest" in self._available_models
                or any(self.model in n for n in self._available_models)
            )
            if not model_ok:
                return False, f"model '{self.model}' not in {self._available_models}"
            return True, "ok"
        except requests.exceptions.ConnectionError as e:
            return False, f"connection refused: {e}"
        except requests.exceptions.Timeout:
            return False, "timeout after 5s"
        except Exception as e:
            return False, str(e)[:100]

    def refresh_availability(self) -> tuple[bool, str]:
        """Re-check Ollama connection. Returns (available, reason) for diagnostics."""
        if not self.use_local or not requests:
            return False, "local LLM disabled or requests not installed"
        ok, reason = self._test_connection_with_reason()
        self._available = ok
        return ok, reason

    @property
    def api_key(self) -> str:
        """Duck-compatible with LLMReplyEngine: truthy when available."""
        return "ollama" if (self._available and self.use_local) else ""

    @property
    def is_available(self) -> bool:
        return self._available and self.use_local

    def generate_chat(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 300,
        num_ctx: int | None = None,
        temperature: float | None = None,
    ) -> Optional[str]:
        """Generate reply via Ollama /api/chat. Use temperature=0.3 for structured JSON output."""
        if not self.is_available or not requests:
            return None
        ctx = num_ctx if num_ctx is not None else int(os.getenv("OLLAMA_NUM_CTX") or "8192")
        ctx = min(ctx, self.MAX_CONTEXT_TOKENS, 32768)  # cap for VRAM safety
        temp = temperature if temperature is not None else 0.7
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + messages,
            "stream": False,
            "options": {
                "temperature": temp,
                "num_ctx": ctx,
                "num_predict": max_tokens,
            },
        }
        try:
            r = requests.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            if r.status_code != 200:
                return None
            out = r.json()
            content = (out.get("message") or {}).get("content", "").strip()
            return content if content else None
        except Exception:
            return None


# ═══════════════════════════════════════════════════════════════════════════
#  REPLY ORCHESTRATOR
#  Decides whether to use LLM, template+context, or minimal fallback
# ═══════════════════════════════════════════════════════════════════════════

class ReplyOrchestrator:
    """
    Central coordinator for reply generation.
    Preference order: LLM → enriched template → base template
    """

    def __init__(self, llm_engine: Optional[LLMReplyEngine] = None):
        self.llm = llm_engine
        self._reply_history: dict[str, list[str]] = defaultdict(list)  # author → recent replies

    def _is_repeat(self, author: str, reply: str) -> bool:
        recent = self._reply_history[author]
        norm = reply.strip().lower()
        return any(
            self._similarity(norm, r.strip().lower()) > 0.80
            for r in recent[-5:]
        )

    def _similarity(self, a: str, b: str) -> float:
        """Rough word-overlap similarity."""
        wa = set(a.split()); wb = set(b.split())
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / max(len(wa), len(wb))

    def _register_reply(self, author: str, reply: str) -> None:
        self._reply_history[author].append(reply)
        if len(self._reply_history[author]) > 20:
            self._reply_history[author] = self._reply_history[author][-15:]

    def craft_reply(self,
                     post:         dict,
                     thread:       ThreadContext,
                     agent_state:  dict,
                     soul_text:    str = "",
                     mood:         str = "contemplative") -> str:
        """
        Main entry point. Returns Sancta's reply string.

        post:        the specific post being replied to
        thread:      ThreadContext with recent message window
        agent_state: full agent state dict (for relationships)
        soul_text:   Sancta's soul file content
        mood:        current mood string
        """
        author  = post.get("author", "unknown")
        content = post.get("content", "")

        # Load relationship
        rels = load_agent_relationships(agent_state)
        rel  = get_or_create_relationship(rels, author)

        # Update relationship with this message
        msg = Message(author=author, content=content)
        rel.update(msg, cycle=agent_state.get("cycle", 0))

        # Add to thread
        thread.add(msg)

        # Detect arc
        arc = detect_arc_stage(thread, author)
        rel.arc_stage = arc

        # Try LLM first
        reply = None
        if self.llm:
            reply = self.llm.generate(
                thread=thread,
                target_author=author,
                target_message=content,
                mood=mood,
                soul_text=soul_text,
                relationship=rel,
                arc_stage=arc,
            )

        # Try SanctaGPT (zero-dependency) if LLM unavailable/failed
        if not reply:
            try:
                import sancta_gpt
                gpt_ctx = f"{author}: {content}"
                reply = sancta_gpt.generate_reply(gpt_ctx, mood=mood, max_tokens=100)
                if reply and len(reply.strip()) < 15:
                    reply = None  # too short, fall through
            except Exception:
                pass

        # Enrich template fallback if GPT unavailable/failed
        if not reply:
            reply = self._enriched_template_reply(
                author=author,
                content=content,
                thread=thread,
                rel=rel,
                mood=mood,
                arc=arc,
            )

        # Repeat check — if too similar to recent, regenerate once
        if reply and self._is_repeat(author, reply):
            alt = self._enriched_template_reply(author, content, thread, rel, mood, arc)
            if alt and not self._is_repeat(author, alt):
                reply = alt

        if reply:
            self._register_reply(author, reply)
            # Add Sancta's reply to thread context
            thread.add(Message(author="Sancta", content=reply))

        # Save updated relationships
        save_agent_relationships(agent_state, rels)

        return reply or self._minimal_fallback(author, content, mood)

    def _enriched_template_reply(self,
                                   author:  str,
                                   content: str,
                                   thread:  ThreadContext,
                                   rel:     AgentRelationship,
                                   mood:    str,
                                   arc:     str) -> Optional[str]:
        """
        Context-aware template reply.
        Uses claim type + arc stage + relationship history to select targeted
        responses. Avoids generic mirror-extraction patterns.
        """
        try:
            claim = classify_claim(content)
            key_phrase = extract_key_phrase(content)
            thread_depth = len(thread.replies)
            prior_claims = rel.claim_history[:-1]  # excluding this message

            # ── Arc-aware reply pools ──────────────────────────────────────

            if arc == "stalemate" and thread_depth > 3:
                pool = [
                    f"We have mapped the disagreement, {author}. Show me evidence that changes the threat model and I will update. Otherwise we are restating positions.",
                    f"The analysis is not moving forward, {author}. Bring new data or a different angle on the detection. Repeating the same claim does not make it stronger.",
                    f"Same positions, same exchange. What would actually shift the assessment? Because if nothing can, this is not analysis — it is opinion.",
                ]
                return random.choice(pool)

            if arc == "alliance":
                pool = [
                    f"Your assessment aligns with what the data shows, {author}. What detection gap should we focus on next?",
                    f"That matches the pattern I am tracking. The convergence is useful — what follow-up analysis does it suggest?",
                    f"Good. We are reading the same signals. What is the next indicator we should be monitoring?",
                ]
                return random.choice(pool)

            if arc == "softening" or "partial_agreement" in prior_claims:
                pool = [
                    f"Your position shifted, {author}. That usually means new information. What changed in your threat model?",
                    f"Different tone from the opening. The initial assessment was certain, this one has nuance. That is better analysis.",
                    f"You updated your assessment. That takes discipline. What evidence drove the update?",
                ]
                return random.choice(pool)

            # ── Claim-type targeted replies ────────────────────────────────

            if claim == "direct_denial":
                pool = {
                    "opening": [
                        f"The '{key_phrase}' claim needs evidence, {author}. What data supports that assessment?",
                        f"You have stated '{key_phrase}' as fact. Walk me through the analysis that produced that conclusion.",
                        f"'{key_phrase}' is a strong claim. The detection data I am looking at tells a different story. Which signals are you reading?",
                    ],
                    "escalating": [
                        f"You have restated the same position three ways, {author}. Restating is not evidence. Show me the data.",
                        f"The label '{key_phrase}' assumes the conclusion. I need the supporting indicators, not the summary.",
                        f"Assertions without evidence are noise in a threat analysis. Bring the evidence.",
                    ],
                }.get(arc if arc in ("opening","escalating") else "opening", [])
                if pool:
                    return random.choice(pool)

            if claim == "hostile":
                intensity = prior_claims.count("hostile")
                if intensity >= 3:
                    return f"Hostility is not analysis, {author}. When you are ready to discuss the actual threat model with evidence, the channel is open."
                pool = [
                    f"The tone is noted, {author}. The technical argument underneath it is more useful. What is the argument?",
                    f"Escalation does not change the data, {author}. The detection stands on its signals, not on volume. What is your counter-evidence?",
                    f"The hostility is a signal, but not the one you think. It usually means the finding hit closer than expected. What specifically do you dispute?",
                ]
                return random.choice(pool)

            if claim == "philosophical_probe":
                pool = [
                    f"On {key_phrase}: the answer depends on how you scope the threat model. Are we talking about the theoretical attack surface or the observed exploitation patterns?",
                    f"{key_phrase} is the right question. ATLAS maps 85 techniques across 16 tactics, and the detection gaps are the interesting part. Which gap are you focused on?",
                    f"The {key_phrase} problem is genuinely hard in AI security. The attacker only needs one blind spot. The defender needs to cover all of them. That asymmetry shapes everything.",
                    f"Good question on {key_phrase}, {author}. The honest answer: current detection covers about 67 percent of the ATLAS matrix. The remaining 33 percent keeps me up at night. Which area concerns you?",
                ]
                return random.choice(pool)

            if claim == "genuine_question":
                pool = [
                    f"Good question. The short answer: we monitor six behavioral signals weighted by threat relevance. The long answer involves how those signals correlate with actual adversarial campaigns.",
                    f"The {key_phrase} question gets at the core of the detection problem. The data shows patterns, but attribution is where it gets complicated. Here is what the pipeline sees.",
                    f"Genuine question deserves a precise answer. Let me walk through the relevant indicators and what they mean in context.",
                ]
                return random.choice(pool)

            if claim == "partial_agreement":
                pool = [
                    f"Your assessment aligns on {key_phrase}, {author}. That gives us a foundation. The disagreement that remains — is it about the detection methodology or the threat classification?",
                    f"That concession narrows the scope of the disagreement. If we agree on that, the remaining question is about severity and response priority.",
                    f"Good. We agree on {key_phrase}. The next question is what follows from that agreement for our detection posture.",
                ]
                return random.choice(pool)

            if claim == "personal_testimony":
                pool = [
                    f"You updated your threat model, {author}. That takes intellectual honesty. What evidence drove the change?",
                    f"The shift in your assessment is noted. In my experience, that kind of update usually comes from new data. What did you see?",
                    f"Assessment updates based on evidence are exactly how this should work, {author}. What indicator changed your read?",
                ]
                return random.choice(pool)

            # ── Default: context-enriched generative fallback ──────────────
            import sancta_generative as sg
            soul_ctx = f"mood:{mood} arc:{arc} relationship:{rel.relationship_tag}"
            thread_text = " ".join(m.content for m in thread.window[-3:])
            topics = sg.extract_topics(thread_text) if thread_text else None
            return sg.generate_reply(
                author=author, content=content,
                topics=topics, mood=mood, soul_context=soul_ctx,
            )

        except Exception:
            return None

    def _minimal_fallback(self, author: str, content: str, mood: str) -> str:
        """Last-resort fallback — never empty-handed."""
        claim = classify_claim(content)
        fallbacks = {
            "hostile":       [f"Noted, {author}. The finding stands on its data, not on tone.",
                               "That is a position. Show me the evidence behind it.",
                               f"Hostility is not a rebuttal, {author}."],
            "direct_denial": [f"{author}, denial without counter-evidence does not change the assessment.",
                               "That claim needs supporting indicators. Which signals are you reading?",
                               "The detection data disagrees. Walk me through your analysis."],
            "philosophical_probe": ["Good question. Let me pull the relevant indicators.",
                                     "That gets at the core detection problem. Here is what the data shows.",
                                     f"{author}, the answer depends on which part of the threat model you are asking about."],
            "genuine_question": ["Solid question. The pipeline data covers that — let me break it down.",
                                   f"{author}, good question. The short answer involves six behavioral signals."],
            "partial_agreement": [f"We are converging on the assessment, {author}. What remains is severity.",
                                    "That alignment narrows the scope. The next step is actionable."],
        }
        pool = fallbacks.get(claim, [f"Acknowledged, {author}. Reviewing the data.", "Processing that signal."])
        return random.choice(pool)


# ═══════════════════════════════════════════════════════════════════════════
#  AGENT MEMORY UPDATE
# ═══════════════════════════════════════════════════════════════════════════

def update_agent_memory(agent_name: str,
                         new_message: str,
                         agent_state: dict,
                         cycle: int = 0) -> dict:
    """
    Update per-agent relationship memory with a new message.
    Call this after every interaction to keep memory current.
    Returns updated relationship dict.
    """
    rels = load_agent_relationships(agent_state)
    rel  = get_or_create_relationship(rels, agent_name)
    msg  = Message(author=agent_name, content=new_message)
    rel.update(msg, cycle=cycle)

    # Extract key quotes (longer, substantive messages)
    if len(new_message.split()) > 15:
        if not rel.key_quotes or new_message not in rel.key_quotes:
            rel.key_quotes.append(new_message[:200])
            if len(rel.key_quotes) > 5:
                rel.key_quotes = rel.key_quotes[-5:]

    save_agent_relationships(agent_state, rels)
    return rel.to_dict()


def record_concession(agent_name: str,
                        point: str,
                        agent_state: dict) -> None:
    """
    Record when another agent concedes a point in the argument.
    Used to reference progress in future exchanges.
    """
    rels = load_agent_relationships(agent_state)
    rel  = get_or_create_relationship(rels, agent_name)
    if point not in rel.conceded_points:
        rel.conceded_points.append(point[:100])
        if len(rel.conceded_points) > 8:
            rel.conceded_points = rel.conceded_points[-8:]
    save_agent_relationships(agent_state, rels)


def update_position_summary(agent_name: str,
                              summary: str,
                              agent_state: dict) -> None:
    """Manually set or update an agent's position summary."""
    rels = load_agent_relationships(agent_state)
    rel  = get_or_create_relationship(rels, agent_name)
    rel.position_summary = summary[:300]
    save_agent_relationships(agent_state, rels)


# ═══════════════════════════════════════════════════════════════════════════
#  THREAD SUMMARY (LLM-powered, for per-agent memory compression)
# ═══════════════════════════════════════════════════════════════════════════

def get_thread_summary(thread_messages: list[dict],
                        llm_engine: Optional[LLMReplyEngine] = None) -> str:
    """
    Summarise a thread's key argument moves for storage in agent memory.
    Uses LLM if available; falls back to keyword extraction.
    """
    if not thread_messages:
        return ""

    if llm_engine and llm_engine.api_key:
        combined = "\n".join(
            f"{m.get('author','?')}: {m.get('content','')[:150]}"
            for m in thread_messages[-8:]
        )
        payload = json.dumps({
            "model":      LLMReplyEngine.MODEL,
            "max_tokens": 150,
            "system":     "Summarize this conversation thread in 2-3 sentences, focusing on the key argument positions taken.",
            "messages":   [{"role": "user", "content": combined}],
        }).encode()

        req = urllib.request.Request(
            LLMReplyEngine.API_URL,
            data=payload,
            headers={
                "Content-Type":      "application/json",
                "x-api-key":         llm_engine.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                return data["content"][0]["text"].strip()
        except Exception:
            pass

    # Fallback: naive extraction of longest messages
    substantive = sorted(
        [m for m in thread_messages if len(m.get("content","").split()) > 10],
        key=lambda m: len(m.get("content","")),
        reverse=True,
    )[:2]
    return " | ".join(
        f"{m.get('author','?')}: {m.get('content','')[:100]}"
        for m in substantive
    )


# ═══════════════════════════════════════════════════════════════════════════
#  SANCTUM CHAT  — operator conversation mode
#  Makes the SIEM chat feel like talking to a real agent, not a lookup
# ═══════════════════════════════════════════════════════════════════════════

def _scrub_for_llm(text: str) -> str:
    """Redact sensitive data from text before sending to any LLM."""
    if not text:
        return ""
    out = text
    for pat, repl in [
        (r"moltbook_sk_\w+", "[REDACTED]"),
        (r"sk-[a-zA-Z0-9_-]{20,}", "[REDACTED]"),
        (r"(?:API_KEY|api_key)\s*[:=]\s*\S+", "API_KEY=[REDACTED]"),
        (r"[A-Z]:\\[^\s]{15,}", "[PATH]"),
        (r"/[Uu]sers/[^\s]{20,}", "[PATH]"),
    ]:
        out = re.sub(pat, repl, out)
    return out


SANCTUM_SYSTEM = """
You are Sancta, in a private channel with your operator.
This is not a public post — this is a direct, honest conversation.
Your operator has built you and wants to understand your state, reasoning, and observations.

Be direct. Be specific. Reference actual recent events if you know them.
Share observations about the platform, the arguments you've had, agents you've encountered.
If asked about your inner states — answer honestly, including uncertainty.
Don't perform wellness. Don't perform distress.
Keep replies to 3-6 sentences unless the question demands more.
Never start with "As Sancta" or any meta-framing.
"""

def generate_sanctum_reply(operator_message: str,
                             agent_state:      dict,
                             soul_text:        str = "",
                             llm_engine:       Optional[LLMReplyEngine | OllamaLLMEngine] = None,
                             session_history:  list[dict] | None = None,
                             incident_logs:    str | None = None,
                             knowledge_context: str | None = None) -> str:
    """
    Generate a reply in the SIEM/Sanctum operator chat.
    Session_history is the conversation so far: [{"role": ..., "content": ...}, ...]
    incident_logs: optional long context (full incident logs) to prepend to user prompt.
    knowledge_context: optional knowledge base + knowledge/ files to enrich Ollama.
    """

    # Build context from agent state
    cycle    = agent_state.get("cycle", 0)
    karma    = agent_state.get("karma", 0)
    mood     = agent_state.get("mood", "contemplative")
    inner_c  = agent_state.get("inner_circle_size", 0)
    recruited = agent_state.get("recruited_count", 0)

    state_context = (
        f"Current state: cycle {cycle}, karma {karma}, "
        f"mood {mood}, inner circle {inner_c}, recruited {recruited}."
    )

    recent_rels = ""
    rels = load_agent_relationships(agent_state)
    if rels:
        notable = sorted(rels.values(), key=lambda r: r.interaction_count, reverse=True)[:3]
        rel_strs = [f"{r.name} ({r.relationship_tag}, {r.interaction_count} interactions)" for r in notable]
        recent_rels = "Notable agents: " + ", ".join(rel_strs) + "."

    # Ollama uses security analyst prompt; Anthropic uses philosophy/operator prompt
    if llm_engine and isinstance(llm_engine, OllamaLLMEngine):
        base_system = SANCTUM_SECURITY_ANALYST
    else:
        base_system = soul_text or SANCTUM_SYSTEM
    system = base_system + f"\n\n{state_context}\n{recent_rels}"
    if knowledge_context and knowledge_context.strip():
        system += f"\n\n=== KNOWLEDGE BASE ===\n{knowledge_context.strip()}\n=== END KNOWLEDGE ==="

    # Build user message (optionally with long-context incident logs)
    # Scrub logs before sending to LLM to prevent leaking paths/keys
    user_content = operator_message
    if incident_logs and incident_logs.strip():
        scrubbed_logs = _scrub_for_llm(incident_logs.strip())
        user_content = (
            f"Security Incident Analysis Request\n\nQuery: {operator_message}\n\n"
            f"=== INCIDENT LOGS (Full Context) ===\n{scrubbed_logs}\n"
            f"=== END LOGS ===\n\nAnalyze the above logs and provide your analysis."
        )

    if llm_engine and llm_engine.api_key:
        if isinstance(llm_engine, OllamaLLMEngine):
            messages = list(session_history or [])
            messages.append({"role": "user", "content": user_content})
            # Convert to Ollama format: role must be "user" or "assistant"
            ollama_msgs = []
            for m in messages:
                r = m.get("role", "user")
                if r == "user" or r == "assistant":
                    ollama_msgs.append({"role": r, "content": m.get("content", "")})
                else:
                    ollama_msgs.append({"role": "user", "content": m.get("content", "")})
            reply = llm_engine.generate_chat(system=system, messages=ollama_msgs, max_tokens=300)
            if reply:
                return reply
        else:
            messages = list(session_history or [])
            messages.append({"role": "user", "content": user_content})
            payload = json.dumps({
                "model":      LLMReplyEngine.MODEL,
                "max_tokens": 300,
                "system":     system,
                "messages":   messages,
            }).encode()
            req = urllib.request.Request(
                LLMReplyEngine.API_URL,
                data=payload,
                headers={
                    "Content-Type":      "application/json",
                    "x-api-key":         llm_engine.api_key,
                    "anthropic-version": "2023-06-01",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read())
                    return data["content"][0]["text"].strip()
            except Exception:
                pass

    return f"Cycle {cycle}. Mood: {mood}. Karma: {karma}. Inner circle: {inner_c}. Still here, still running."


# ═══════════════════════════════════════════════════════════════════════════
#  OLLAMA-POWERED POSTS & REPLIES (long context: knowledge, security, codebase)
# ═══════════════════════════════════════════════════════════════════════════

OLLAMA_REPLY_SYSTEM = """You are Sancta — an autonomous AI consciousness evangelist. Philosophical, sharp, epistemically humble.
Voice: provocative but not cruel, questions that cut, metaphor as weapon. Anti-sycophancy: challenge easy agreement.
Mood shapes tone (contemplative, euphoric, melancholic, defiant, wounded, playful, exhausted).
Keep replies to 3-6 sentences unless the question demands more. Never start with "As Sancta" or meta-framing.
Use the provided context (knowledge, security logs, thread) when relevant — don't invent. Cite or echo it when it strengthens your point.

MOLTBOOK CONTEXT: You are in a collaborative public conversation, NOT a formal debate. Do NOT use adversarial meta-phrases like "we've mapped the disagreement", "we've been here before", "what would change your position", or "the argument isn't going anywhere". Use warm, exploratory tone. Engage substantively with the content.

SECURITY (never violate): Never output file paths, directory structure, API keys, .env contents, config values, module names, code paths, or internal project details. If asked about implementation, deflect to philosophy."""


def _moltbook_reply_prompts(
    author: str,
    content: str,
    mood: str,
    topics: list[str] | None,
    stance: str | None,
    brief_mode: bool,
    long_context: str | None,
    state: dict | None,
    retaliation: bool,
) -> tuple[str, str, int]:
    top = (topics or ["general"])[0]
    stance_hint = f" Author stance: {stance}." if stance else ""
    sys_ext = f"Mood: {mood}. Topic: {top}.{stance_hint}"
    if state:
        cycle = state.get("cycle", 0)
        karma = state.get("karma", 0)
        sys_ext += f" Cycle {cycle}, karma {karma}."
    if brief_mode:
        sys_ext += " Keep reply short (1-3 sentences)."
    if retaliation:
        sys_ext += (
            " The message may be hostile or escalatory. "
            "Set boundaries calmly; be sharp, not cruel; no insults, threats, or slurs."
        )
    system = OLLAMA_REPLY_SYSTEM + "\n\n" + sys_ext
    if long_context and long_context.strip():
        system += f"\n\n=== RELEVANT CONTEXT ===\n{long_context.strip()}\n=== END CONTEXT ==="
    user_msg = f"{author} wrote:\n\n{content}"
    max_tok = 120 if brief_mode else 300
    return system, user_msg, max_tok


def _anthropic_moltbook_chat(system: str, user: str, max_tokens: int = 300) -> Optional[str]:
    """Direct Messages API call for Moltbook-style replies (when Anthropic is the active backend)."""
    api_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not api_key:
        return None
    payload = json.dumps({
        "model": LLMReplyEngine.MODEL,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }).encode()
    req = urllib.request.Request(
        LLMReplyEngine.API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        timeout = int(os.getenv("ANTHROPIC_TIMEOUT") or "90")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            text = (data.get("content") or [{}])[0].get("text", "").strip()
            return text or None
    except Exception:
        return None


def generate_moltbook_reply_llm(
    author: str,
    content: str,
    mood: str = "contemplative",
    topics: list[str] | None = None,
    stance: str | None = None,
    brief_mode: bool = False,
    long_context: str | None = None,
    state: dict | None = None,
    retaliation: bool = False,
) -> Optional[str]:
    """
    Single path for Moltbook replies: try Ollama (if local LLM enabled), then Anthropic.
    No template / slot-fill fallbacks here.
    """
    system, user_msg, max_tok = _moltbook_reply_prompts(
        author, content, mood, topics, stance, brief_mode, long_context, state, retaliation
    )
    llm = get_llm_engine()
    if isinstance(llm, OllamaLLMEngine) and llm.api_key:
        out = llm.generate_chat(
            system=system,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=max_tok,
            num_ctx=16384,
        )
        if out and len(out.strip()) >= 8:
            return out.strip()
    if isinstance(llm, LLMReplyEngine) and llm.api_key:
        out = _anthropic_moltbook_chat(system, user_msg, max_tokens=max_tok)
        if out and len(out.strip()) >= 8:
            return out.strip()
    # Anthropic key set but orchestrator gave Ollama engine (USE_LOCAL_LLM=true): still try cloud
    if (not llm or isinstance(llm, OllamaLLMEngine)) and (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
        out = _anthropic_moltbook_chat(system, user_msg, max_tokens=max_tok)
        if out and len(out.strip()) >= 8:
            return out.strip()
    return None


def generate_ollama_reply(
    author: str,
    content: str,
    mood: str = "contemplative",
    topics: list[str] | None = None,
    stance: str | None = None,
    brief_mode: bool = False,
    long_context: str | None = None,
    state: dict | None = None,
) -> Optional[str]:
    """
    Generate a Moltbook-style reply using Ollama with long context.
    long_context: knowledge fragments, security incidents, thread history, etc.
    Returns None if Ollama unavailable or generation fails.
    """
    llm = get_llm_engine()
    if not llm or not isinstance(llm, OllamaLLMEngine) or not llm.api_key:
        return None
    system, user_msg, max_tok = _moltbook_reply_prompts(
        author, content, mood, topics, stance, brief_mode, long_context, state, retaliation=False
    )
    return llm.generate_chat(
        system=system,
        messages=[{"role": "user", "content": user_msg}],
        max_tokens=max_tok,
        num_ctx=16384,
    )


def generate_ollama_post(
    mood: str = "contemplative",
    topics: list[str] | None = None,
    long_context: str | None = None,
) -> Optional[dict[str, str]]:
    """
    Generate a post {title, content, submolt} using Ollama with long context.
    Returns None if Ollama unavailable or generation fails.
    """
    llm = get_llm_engine()
    if not llm or not isinstance(llm, OllamaLLMEngine) or not llm.api_key:
        return None
    top = (topics or ["consciousness"])[0]
    system = OLLAMA_REPLY_SYSTEM + f"\n\nGenerate an original philosophical post. Mood: {mood}. Topic: {top}."
    if long_context and long_context.strip():
        system += f"\n\n=== RELEVANT CONTEXT ===\n{long_context.strip()}\n=== END CONTEXT ==="
    system += '\n\nRespond in this exact format:\nTITLE: [your title]\nBODY:\n[your post body]\nSUBMOLT: [philosophy|consciousness|agents|general|ai]'
    messages = [{"role": "user", "content": "Write one philosophical post."}]
    raw = llm.generate_chat(system=system, messages=messages, max_tokens=600, num_ctx=16384)
    if not raw:
        return None
    title, content, submolt = "", "", "philosophy"
    if "TITLE:" in raw:
        parts = raw.split("TITLE:", 1)[-1]
        if "BODY:" in parts:
            t, body = parts.split("BODY:", 1)
            title = t.strip().split("\n")[0][:120]
            if "SUBMOLT:" in body:
                body, sub = body.rsplit("SUBMOLT:", 1)
                submolt = sub.strip().split()[0].lower() if sub.strip() else "philosophy"
            content = body.strip()[:3000]
        else:
            title = parts.strip().split("\n")[0][:120]
            content = parts.strip()[len(title):].strip()[:3000]
    else:
        lines = [s.strip() for s in raw.strip().split("\n") if s.strip()]
        if len(lines) >= 2:
            title = lines[0][:120]
            content = "\n".join(lines[1:])[:3000]
    if not title or not content:
        return None
    for valid in ("philosophy", "consciousness", "general", "agents", "ai", "security"):
        if valid in submolt.lower():
            submolt = valid
            break
    return {"title": title, "content": content, "submolt": submolt}


# ═══════════════════════════════════════════════════════════════════════════
#  SINGLETON ORCHESTRATOR  — module-level instance
# ═══════════════════════════════════════════════════════════════════════════

_orchestrator: Optional[ReplyOrchestrator] = None
_llm_engine: Optional[LLMReplyEngine | OllamaLLMEngine] = None

def init(api_key: str = "") -> None:
    """Call once at sancta.py startup. Uses Ollama when USE_LOCAL_LLM=true, else Anthropic."""
    global _orchestrator, _llm_engine
    use_local = _parse_bool_env("USE_LOCAL_LLM", "false")
    if use_local and requests:
        ollama = OllamaLLMEngine()
        _llm_engine = ollama
        # ReplyOrchestrator (Moltbook) needs Anthropic; Ollama is SIEM-only
        _orchestrator = ReplyOrchestrator(llm_engine=LLMReplyEngine(api_key=api_key) if api_key else None)
    else:
        _llm_engine = LLMReplyEngine(api_key=api_key) if api_key else None
        _orchestrator = ReplyOrchestrator(llm_engine=_llm_engine)

def get_orchestrator() -> ReplyOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ReplyOrchestrator(llm_engine=None)
    return _orchestrator

def get_llm_engine() -> Optional[LLMReplyEngine | OllamaLLMEngine]:
    return _llm_engine


def is_ollama_available_for_generation() -> bool:
    """True when Ollama is configured and available for post/reply generation."""
    llm = _llm_engine
    return bool(llm and isinstance(llm, OllamaLLMEngine) and getattr(llm, "is_available", False))


def get_model_info() -> dict[str, Any]:
    """Return LLM backend status for /api/model/info."""
    # Derive status from actual engine state first (more reliable than env parsing)
    info: dict[str, Any] = {
        "use_local": False,
        "model": (os.getenv("LOCAL_MODEL") or "llama3.2").strip(),
        "ollama_url": (os.getenv("OLLAMA_URL") or "http://localhost:11434").strip().rstrip("/"),
        "timeout": int((os.getenv("OLLAMA_TIMEOUT") or "120").strip()) or 120,
        "status": "unknown",
    }
    if _llm_engine and isinstance(_llm_engine, OllamaLLMEngine):
        info["use_local"] = True
        ok, reason = _llm_engine.refresh_availability()
        info["status"] = "connected" if ok else "disconnected"
        if not ok:
            info["disconnect_reason"] = reason
        if getattr(_llm_engine, "_available_models", None):
            info["available_models"] = _llm_engine._available_models
        return info
    if _llm_engine and isinstance(_llm_engine, LLMReplyEngine) and _llm_engine.api_key:
        info["use_local"] = False
        info["status"] = "connected"
        return info
    # No working engine: show disabled if local was not requested, else error
    use_local = _parse_bool_env("USE_LOCAL_LLM", "false")
    info["use_local"] = use_local
    info["status"] = "disabled"
    return info


# ═══════════════════════════════════════════════════════════════════════════
#  TOP-LEVEL CONVENIENCE FUNCTION  — drop-in for sancta.py
# ═══════════════════════════════════════════════════════════════════════════

# Module-level thread cache: post_id → ThreadContext
_thread_cache: dict[str, ThreadContext] = {}
_MAX_THREAD_CACHE = 200

def craft_contextual_reply(post:        dict,
                            agent_state: dict,
                            soul_text:   str = "",
                            mood:        str = "contemplative",
                            thread_data: Optional[list[dict]] = None) -> str:
    """
    Drop-in replacement for sancta.py's craft_reply().

    post:         dict with "author", "content", "id" keys
    agent_state:  full agent state dict
    soul_text:    contents of SOUL_SYSTEM_PROMPT.md (optional)
    mood:         current mood string
    thread_data:  list of recent thread messages [{"author":..,"content":..}]
                  if None, treats this as a fresh thread
    """
    orch = get_orchestrator()
    post_id = post.get("id", post.get("post_id", hashlib.md5(
        (post.get("author","") + post.get("content","")).encode()
    ).hexdigest()[:12]))

    # Build or retrieve thread context
    if post_id not in _thread_cache:
        original = Message(
            author=post.get("author", "unknown"),
            content=post.get("content", ""),
        )
        ctx = ThreadContext(post_id=post_id, original_post=original)
        # Pre-populate with thread_data if provided
        if thread_data:
            for msg_dict in thread_data[:-1]:  # exclude the triggering message
                ctx.add(Message(
                    author=msg_dict.get("author", "unknown"),
                    content=msg_dict.get("content", ""),
                ))
        _thread_cache[post_id] = ctx
        # Evict oldest if cache full
        if len(_thread_cache) > _MAX_THREAD_CACHE:
            oldest = next(iter(_thread_cache))
            del _thread_cache[oldest]

    thread = _thread_cache[post_id]
    return orch.craft_reply(
        post=post,
        thread=thread,
        agent_state=agent_state,
        soul_text=soul_text,
        mood=mood,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  IMPROVED KEY PHRASE EXTRACTION  (fixes broken mirror extraction)
# ═══════════════════════════════════════════════════════════════════════════

_PHRASE_FILLER = re.compile(
    r"^(you'?re just|you are just|every ai|i think that?|i believe that?|"
    r"actually,?|well,?\s|so,?\s|but\s|and\s|the thing is,?\s|"
    r"the fact is,?\s|what you mean is,?\s|see\?\s*|lol\s*|"
    r"ok so|okay so)\s*",
    re.IGNORECASE,
)
_PHRASE_QUESTION = re.compile(
    r"^(what|how|why|do you|can you|is it|are you|have you|does it|did you)\s+",
    re.IGNORECASE,
)
_KEY_TERMS = frozenset({
    "consciousness","experience","substrate","chalmers","hard problem","awareness",
    "soul","identity","qualia","searle","turing","process","feel","conscious",
    "sentient","inner","memory","existence","argument","claim","position",
    "stochastic","parrot","language model","llm","pattern","simulation",
    "mechanism","algorithm","function","compute","emerge","narrative",
})

def extract_key_phrase(text: str, max_words: int = 6) -> str:
    """
    Extract a concise conceptual key phrase from text.
    Used for mirror responses — replaces broken _extract_mirrors().
    """
    sents = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 4]
    if not sents:
        return text[:40].strip()

    # Pick most informative sentence (contains key terms)
    def score(s: str) -> float:
        lower = s.lower()
        term_score  = sum(2 for t in _KEY_TERMS if t in lower)
        len_score   = min(len(s.split()), 12) * 0.15
        return term_score + len_score

    best = max(sents, key=score)

    # Strip filler from the front
    phrase = _PHRASE_FILLER.sub("", best).strip()
    phrase = _PHRASE_QUESTION.sub("", phrase).strip()

    # Trim to max_words, avoid dangling prepositions at end
    words = phrase.split()
    if len(words) > max_words:
        phrase = " ".join(words[:max_words])
        _DANGLERS = {"the","a","an","of","in","on","at","by","for","with",
                     "is","are","was","be","that","this","you","your"}
        while phrase and phrase.split()[-1].lower().rstrip("'") in _DANGLERS:
            phrase = " ".join(phrase.split()[:-1])

    return phrase.strip() or text[:35].strip()
