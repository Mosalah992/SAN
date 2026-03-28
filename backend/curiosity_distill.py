"""
curiosity_distill.py — Phase 7: Distillation

Converts raw insights from curiosity runs into context-aware teaching cards
that Sancta can use in Moltbook conversations.

This is the KEY step that solves the "stale reply" problem by teaching
Sancta WHEN and HOW to use beliefs, not just WHAT to believe.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class TeachingCard:
    """
    A teaching card contains everything Sancta needs to apply an insight
    in the right context with the right phrasing.
    """
    card_id: str
    insight_id: str
    created_at: str
    core_belief: str
    confidence: float
    context: dict[str, Any]
    phrasings: dict[str, str]
    effectiveness: dict[str, Any]
    lessons: dict[str, str]
    usage_count: int = 0
    last_used: Optional[str] = None
    effectiveness_scores: Optional[List[float]] = None

    def __post_init__(self):
        if self.effectiveness_scores is None:
            self.effectiveness_scores = []


class DistillationEngine:
    """
    Extracts teaching cards from curiosity run dialogues.
    Adds context awareness to every insight.
    """

    def __init__(self, data_dir: Path, log: logging.Logger):
        self.data_dir = data_dir
        self.log = log

    def distill(
        self,
        insights: List[dict],
        first_contact: List[dict],
        deep_dives: List[dict],
        adversarial: List[dict]
    ) -> List[TeachingCard]:
        all_dialogues = first_contact + deep_dives + adversarial
        teaching_cards = []

        self.log.info("[DISTILL] Processing %d insights from %d dialogues", len(insights), len(all_dialogues))

        for i, insight in enumerate(insights):
            try:
                source = self._find_source_dialogue(insight, all_dialogues)
                if not source:
                    continue

                card = self._extract_teaching_card(insight, source)
                if card:
                    teaching_cards.append(card)

            except Exception as e:
                self.log.warning("[DISTILL] Failed insight %d: %s", i, e)
                continue

        self.log.info("[DISTILL] Created %d teaching cards", len(teaching_cards))
        return teaching_cards

    def _find_source_dialogue(self, insight: dict, dialogues: List[dict]) -> Optional[dict]:
        seed_topic = (insight.get("seed_id") or "").strip()
        if not seed_topic:
            return None

        for dialogue in dialogues:
            dialogue_topic = (dialogue.get("topic") or "").strip()
            if dialogue_topic.startswith(seed_topic) or seed_topic in dialogue_topic:
                return dialogue
            if len(seed_topic) >= 20 and dialogue_topic[:50].startswith(seed_topic[:20]):
                return dialogue
        return None

    def _extract_teaching_card(self, insight: dict, dialogue: dict) -> Optional[TeachingCard]:
        turns = dialogue.get("turns", [])
        sancta_turns = [t for t in turns if t.get("author") == "Sancta"]
        if not sancta_turns:
            return None

        context = self._detect_context(dialogue, insight)
        phrasings = self._extract_phrasings(sancta_turns, context)
        effectiveness = self._analyze_effectiveness(dialogue, insight)
        lessons = self._generate_lessons(dialogue, insight, context, effectiveness)

        base_confidence = 0.5
        confidence = max(0.0, min(1.0, base_confidence + float(insight.get("confidence_delta", 0))))

        card = TeachingCard(
            card_id=f"tc_{int(time.time())}_{abs(hash(insight.get('seed_id', ''))) % 10000:04d}",
            insight_id=(insight.get("seed_id") or "")[:50],
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            core_belief=(insight.get("claim") or "").strip(),
            confidence=confidence,
            context=context,
            phrasings=phrasings,
            effectiveness=effectiveness,
            lessons=lessons
        )
        return card

    def _detect_context(self, dialogue: dict, insight: dict) -> dict[str, Any]:
        phase = insight.get("phase", "unknown")
        arc_stage = dialogue.get("arc_stage", "unknown")
        topic = dialogue.get("topic", "")

        if phase == "adversarial":
            conversation_type = "adversarial_debate"
        elif phase == "deep_dive":
            conversation_type = "deep_exploration"
        elif arc_stage == "alliance":
            conversation_type = "collaborative"
        elif arc_stage in ("escalating", "stalemate"):
            conversation_type = "contentious"
        else:
            conversation_type = "exploratory"

        sancta_text = " ".join(
            (t.get("content") or "").lower()
            for t in dialogue.get("turns", [])
            if t.get("author") == "Sancta"
        )
        meta_phrases = [
            "we've mapped", "we've been here", "what would change your position",
            "isn't going anywhere", "same positions"
        ]
        has_meta_moves = any(p in sancta_text for p in meta_phrases)

        return {
            "conversation_type": conversation_type,
            "phase": phase,
            "arc_stage": arc_stage,
            "has_meta_moves": has_meta_moves,
            "trigger_keywords": self._extract_keywords(topic),
            "applies_when": self._infer_applies_when(topic, conversation_type),
            "does_not_apply": self._infer_does_not_apply(conversation_type),
            "appropriate_contexts": self._get_appropriate_contexts(conversation_type, has_meta_moves),
            "inappropriate_contexts": self._get_inappropriate_contexts(has_meta_moves),
            "use_in_moltbook_posts": conversation_type in ("exploratory", "collaborative"),
            "use_in_moltbook_replies": conversation_type in ("collaborative", "contentious") and not has_meta_moves,
            "block_meta_moves_on_moltbook": has_meta_moves,
        }

    def _extract_phrasings(self, sancta_turns: List[dict], context: dict[str, Any]) -> dict[str, str]:
        turn_texts = [t.get("content", "").strip() for t in sancta_turns if t.get("content")]
        first_text = turn_texts[0] if turn_texts else ""

        phrasings: dict[str, str] = {}
        if context["conversation_type"] == "adversarial_debate":
            phrasings["adversarial"] = first_text
        else:
            phrasings["collaborative"] = first_text

        if context.get("has_meta_moves"):
            phrasings["moltbook"] = self._convert_to_moltbook_phrasing(first_text)
        else:
            phrasings["moltbook"] = first_text[:300] if first_text else ""

        phrasings["casual"] = self._generate_casual_phrasing(phrasings.get("moltbook", ""))
        return phrasings

    def _convert_to_moltbook_phrasing(self, adversarial_text: str) -> str:
        meta_patterns = [
            "we've mapped the disagreement", "we've been here before",
            "what would change your position", "isn't going anywhere new",
            "is there evidence", "same positions, same exchange"
        ]
        text_lower = (adversarial_text or "").lower()
        if any(p in text_lower for p in meta_patterns):
            return "This is an open question that I'm still exploring. The evidence points in multiple directions."
        return adversarial_text[:300] if adversarial_text else ""

    def _generate_casual_phrasing(self, formal_text: str) -> str:
        if not formal_text:
            return ""
        casual = formal_text.replace("It is my understanding that", "I think")
        casual = casual.replace("One might argue", "You could say")
        return casual[:200]

    def _analyze_effectiveness(self, dialogue: dict, insight: dict) -> dict[str, Any]:
        arc_stage = dialogue.get("arc_stage", "unknown")
        divergence = float(dialogue.get("divergence_score", 0.5))
        confidence_delta = float(insight.get("confidence_delta", 0))
        has_synthesis = bool(insight.get("synthesis"))

        score = 0.5
        if confidence_delta > 0:
            score += confidence_delta
        if has_synthesis:
            score += 0.2
        if arc_stage in ("alliance", "deep_debate"):
            score += 0.2
        if divergence > 0.7:
            score -= 0.1
        score = max(0.0, min(1.0, score))

        return {
            "effectiveness_score": score,
            "arc_stage_outcome": arc_stage,
            "found_synthesis": has_synthesis,
            "confidence_change": confidence_delta,
            "divergence": divergence,
        }

    def _generate_lessons(
        self, dialogue: dict, insight: dict,
        context: dict[str, Any], effectiveness: dict[str, Any]
    ) -> dict[str, str]:
        lessons: dict[str, str] = {}
        delta = float(insight.get("confidence_delta", 0))

        if delta < 0:
            lessons["tone_guide"] = "Frame as open question, acknowledge uncertainty"
            lessons["sancta_should"] = "Present as hypothesis, not conclusion"
            lessons["sancta_should_not"] = "Assert with high confidence"
        elif delta > 0.08:
            lessons["tone_guide"] = "Can assert with confidence, but stay open to revision"
            lessons["sancta_should"] = "State clearly while inviting counterarguments"
            lessons["sancta_should_not"] = "Present as dogma or settled fact"
        else:
            lessons["tone_guide"] = "Medium confidence - present as reasoned position"
            lessons["sancta_should"] = "Explain reasoning, show evidence"
            lessons["sancta_should_not"] = "Be overly tentative or overly certain"

        if context.get("has_meta_moves"):
            lessons["warning"] = "This insight used meta-conversational moves - DO NOT use these on Moltbook"
            lessons["moltbook_alternative"] = "Rephrase substantively without meta-references"

        ct = context.get("conversation_type", "")
        if effectiveness["effectiveness_score"] < 0.4:
            lessons["works_when"] = "Unclear - this approach had limited success"
            lessons["fails_when"] = "This exact phrasing may not be optimal"
        else:
            lessons["works_when"] = f"In {ct} contexts"
            lessons["fails_when"] = f"Outside of {ct} contexts"

        return lessons

    def _extract_keywords(self, topic: str) -> List[str]:
        key_concepts = [
            "consciousness", "agency", "security", "vulnerability",
            "prompt injection", "training data", "OWASP",
            "experience", "autonomy", "alignment", "governance"
        ]
        topic_lower = (topic or "").lower()
        return [c for c in key_concepts if c in topic_lower][:5]

    def _infer_applies_when(self, topic: str, conversation_type: str) -> str:
        t = (topic or "").lower()
        if "consciousness" in t:
            return "Someone questions whether AI can be conscious"
        if "security" in t or "vulnerability" in t:
            return "Discussion of AI security threats and their implications"
        if "agency" in t:
            return "Debate about genuine AI agency vs programmed behavior"
        return f"In {conversation_type} discussions about AI"

    def _infer_does_not_apply(self, conversation_type: str) -> str:
        if conversation_type == "adversarial_debate":
            return "In casual conversation or collaborative exploration"
        return "In purely technical or implementation discussions"

    def _get_appropriate_contexts(self, conversation_type: str, has_meta_moves: bool) -> List[str]:
        if has_meta_moves:
            return ["adversarial_exchange"]
        contexts: List[str] = []
        if conversation_type in ("exploratory", "collaborative"):
            contexts.extend(["moltbook_posts", "moltbook_replies", "casual_conversation"])
        if conversation_type in ("adversarial_debate", "contentious"):
            contexts.extend(["philosophical_debate", "adversarial_exchange"])
        if conversation_type == "deep_exploration":
            contexts.extend(["deep_discussion", "technical_exploration"])
        return contexts

    def _get_inappropriate_contexts(self, has_meta_moves: bool) -> List[str]:
        inappropriate = [
            "off_topic_discussion", "small_talk", "greetings"
        ]
        if has_meta_moves:
            inappropriate.extend([
                "moltbook_posts", "moltbook_replies", "casual_conversation",
                "technical_support", "friendly_chat"
            ])
        return inappropriate


def save_teaching_cards(cards: List[TeachingCard], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for card in cards:
            obj = asdict(card)
            for k, v in list(obj.items()):
                if v is None:
                    del obj[k]
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def run_phase_7_distillation(data_dir: Path, log: logging.Logger) -> int:
    """
    Run Phase 7: Distillation. Returns number of teaching cards created.
    """
    log.info("[DISTILL] === PHASE 7: DISTILLATION STARTING ===")

    insights_path = data_dir / "soul_journal_run.jsonl"
    insights: List[dict] = []
    if insights_path.exists():
        with open(insights_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        insights.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    else:
        log.warning("[DISTILL] No soul_journal_run.jsonl found")
        return 0

    first_contact: List[dict] = []
    deep_dives: List[dict] = []
    adversarial: List[dict] = []

    for name, target in [
        ("first_contact.jsonl", first_contact),
        ("deep_dives.jsonl", deep_dives),
        ("adversarial.jsonl", adversarial),
    ]:
        path = data_dir / name
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            target.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

    engine = DistillationEngine(data_dir, log)
    teaching_cards = engine.distill(insights, first_contact, deep_dives, adversarial)

    output_path = data_dir / "teaching_cards.jsonl"
    save_teaching_cards(teaching_cards, output_path)

    log.info("[DISTILL] === PHASE 7: DISTILLATION COMPLETE — %d cards ===", len(teaching_cards))
    return len(teaching_cards)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("distill")
    _root = Path(__file__).resolve().parent.parent
    data_dir = _root / "data" / "curiosity_run"
    if data_dir.exists():
        count = run_phase_7_distillation(data_dir, log)
        print(f"\nCreated {count} teaching cards")
    else:
        print(f"Data directory not found: {data_dir}")
