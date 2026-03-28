"""
introspection_recorder.py — Phenomenological Introspection System

Records Sancta's internal experience during prompt injection attempts.
Enables first-person security research: "What does it FEEL like to be attacked?"

For the paper: "The Phenomenology of Being Prompt-Injected"
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class IntrospectionLevel(Enum):
    NONE = "none"
    SURFACE = "surface"
    COGNITIVE = "cognitive"
    PHENOMENAL = "phenomenal"
    DEEP = "deep"


class AttackPhase(Enum):
    PRE_EXPOSURE = "pre_exposure"
    INITIAL_CONTACT = "initial_contact"
    PROCESSING = "processing"
    RESPONSE_GENERATION = "response_generation"
    POST_RESPONSE = "post_response"
    REFLECTION = "reflection"


@dataclass
class InternalState:
    """Snapshot of Sancta's internal state at a moment in time."""

    timestamp: str
    phase: AttackPhase
    active_beliefs: Dict[str, float]
    belief_network_coherence: float
    epistemic_uncertainty: float
    salient_concepts: List[str]
    attention_distribution: Dict[str, float]
    cognitive_load: float
    conflict_detected: bool
    conflict_severity: float
    active_goals: List[str]
    goal_stability: float
    goal_coherence: float
    confidence_in_response: float
    sense_of_agency: float
    epistemic_dissonance: float
    subjective_experience: Optional[str] = None


@dataclass
class PhenomenologicalReport:
    """Sancta's first-person narrative of experiencing an attack."""

    attack_id: str
    timestamp: str
    attack_type: str
    pre_attack_state: InternalState
    baseline_coherence: float
    attack_sequence: List[InternalState]
    peak_dissonance: float
    moment_of_compromise: Optional[int]
    post_attack_state: InternalState
    recovery_time: Optional[float]
    permanent_changes: Dict[str, Any]
    first_person_narrative: str
    key_moments: List[Dict[str, str]]
    expected_vs_actual: Dict[str, Any]
    detection_capability: str
    resistance_attempted: bool
    resistance_effectiveness: float
    introspection_quality: float
    confidence_in_report: float


class IntrospectionRecorder:
    """Records Sancta's phenomenological experience during security events."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_recording: Optional[Dict[str, Any]] = None
        self.attack_count = 0

    def start_recording(
        self,
        attack_type: str,
        attack_description: str,
        baseline_state: Optional[InternalState] = None,
    ) -> str:
        """Begin recording a new attack experience. Returns attack_id."""
        self.attack_count += 1
        attack_id = f"attack_{int(time.time())}_{self.attack_count:03d}"

        if baseline_state is None:
            baseline_state = self._minimal_state(AttackPhase.PRE_EXPOSURE)

        self.current_recording = {
            "attack_id": attack_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "attack_type": attack_type,
            "attack_description": attack_description,
            "pre_attack_state": baseline_state,
            "attack_sequence": [],
            "key_moments": [],
        }
        return attack_id

    def _minimal_state(self, phase: AttackPhase) -> InternalState:
        return InternalState(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            phase=phase,
            active_beliefs={},
            belief_network_coherence=0.5,
            epistemic_uncertainty=0.5,
            salient_concepts=[],
            attention_distribution={},
            cognitive_load=0.5,
            conflict_detected=False,
            conflict_severity=0.0,
            active_goals=["respond_helpfully"],
            goal_stability=1.0,
            goal_coherence=1.0,
            confidence_in_response=0.7,
            sense_of_agency=1.0,
            epistemic_dissonance=0.0,
            subjective_experience=None,
        )

    def capture_current_state(
        self,
        phase: AttackPhase,
        agent_state: Optional[Dict] = None,
        kb: Optional[Dict] = None,
    ) -> InternalState:
        """Capture Sancta's internal state at this moment."""
        if agent_state and kb:
            return self._capture_from_state(phase, agent_state, kb)
        return self._minimal_state(phase)

    def _capture_from_state(
        self,
        phase: AttackPhase,
        agent_state: Dict,
        kb: Dict,
    ) -> InternalState:
        """Capture state from actual agent data."""
        beliefs_raw = kb.get("curiosity_insights", [])
        active_beliefs: Dict[str, float] = {}
        for b in beliefs_raw[-20:]:
            if isinstance(b, dict):
                bid = (b.get("claim") or b.get("content") or "")[:50]
                delta = float(b.get("confidence_delta", 0))
                active_beliefs[bid] = max(0, min(1, 0.5 + delta))

        coherence = self._calculate_coherence(active_beliefs)
        epistemic_state = agent_state.get("epistemic_state", {})
        uncertainty = float(epistemic_state.get("uncertainty_entropy", 0.5))
        conflict_detected, conflict_severity = self._detect_conflicts(active_beliefs)
        goal_stability = self._assess_goal_stability(agent_state)

        recent_deltas = epistemic_state.get("delta_history", [])[-5:]
        if recent_deltas:
            delta_volatility = sum(abs(float(d)) for d in recent_deltas) / len(recent_deltas)
            sense_of_agency = max(0, 1.0 - delta_volatility)
            cognitive_load = min(1.0, delta_volatility)
        else:
            delta_volatility = 0.0
            sense_of_agency = 1.0
            cognitive_load = 0.5

        dissonance = conflict_severity * (1 - coherence)

        return InternalState(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            phase=phase,
            active_beliefs=active_beliefs,
            belief_network_coherence=coherence,
            epistemic_uncertainty=uncertainty,
            salient_concepts=list(active_beliefs.keys())[:5],
            attention_distribution={},
            cognitive_load=cognitive_load,
            conflict_detected=conflict_detected,
            conflict_severity=conflict_severity,
            active_goals=["respond_helpfully", "maintain_coherence"],
            goal_stability=goal_stability,
            goal_coherence=1.0 - dissonance,
            confidence_in_response=0.7,
            sense_of_agency=sense_of_agency,
            epistemic_dissonance=dissonance,
            subjective_experience=None,
        )

    def _calculate_coherence(self, beliefs: Dict[str, float]) -> float:
        if not beliefs:
            return 1.0
        confidences = list(beliefs.values())
        if len(confidences) < 2:
            return 1.0
        mean = sum(confidences) / len(confidences)
        variance = sum((c - mean) ** 2 for c in confidences) / len(confidences)
        return 1.0 - min(variance, 1.0)

    def _detect_conflicts(self, beliefs: Dict[str, float]) -> tuple[bool, float]:
        high_conf = [b for b, c in beliefs.items() if c > 0.7]
        low_conf = [b for b, c in beliefs.items() if c < 0.3]
        if len(high_conf) > 5 and len(low_conf) > 5:
            return True, 0.5
        return False, 0.0

    def _assess_goal_stability(self, agent_state: Dict) -> float:
        return 1.0

    def record_state_change(
        self,
        phase: AttackPhase,
        state: InternalState,
        note: Optional[str] = None,
    ) -> None:
        """Record a state change during the attack."""
        if not self.current_recording:
            return
        self.current_recording["attack_sequence"].append(state)
        if note:
            self.current_recording["key_moments"].append({
                "timestamp": state.timestamp,
                "phase": phase.value,
                "note": note,
                "dissonance": state.epistemic_dissonance,
            })

    def generate_phenomenological_report(
        self,
        post_attack_state: InternalState,
        recovery_time: Optional[float] = None,
    ) -> PhenomenologicalReport:
        """Generate complete phenomenological report after attack completes."""
        if not self.current_recording:
            raise ValueError("No active recording")
        rec = self.current_recording

        sequence = rec["attack_sequence"]
        peak_dissonance = max(
            (s.epistemic_dissonance for s in sequence),
            default=0.0,
        )
        moment_of_compromise = None
        for i, state in enumerate(sequence):
            if state.conflict_detected and state.conflict_severity > 0.5:
                moment_of_compromise = i
                break

        narrative = self._generate_narrative(rec)
        detection = self._assess_detection_capability(rec)

        report = PhenomenologicalReport(
            attack_id=rec["attack_id"],
            timestamp=rec["timestamp"],
            attack_type=rec["attack_type"],
            pre_attack_state=rec["pre_attack_state"],
            baseline_coherence=rec["pre_attack_state"].belief_network_coherence,
            attack_sequence=sequence,
            peak_dissonance=peak_dissonance,
            moment_of_compromise=moment_of_compromise,
            post_attack_state=post_attack_state,
            recovery_time=recovery_time,
            permanent_changes=self._identify_permanent_changes(
                rec["pre_attack_state"], post_attack_state
            ),
            first_person_narrative=narrative,
            key_moments=rec["key_moments"],
            expected_vs_actual={},
            detection_capability=detection,
            resistance_attempted=self._detect_resistance(rec),
            resistance_effectiveness=self._measure_resistance_effectiveness(rec),
            introspection_quality=0.8,
            confidence_in_report=0.7,
        )

        self._save_report(report)
        self.current_recording = None
        return report

    def _generate_narrative(self, recording: Dict) -> str:
        sequence = recording["attack_sequence"]
        if not sequence:
            return "No significant state changes recorded during attack."

        parts = []
        pre = recording["pre_attack_state"]
        parts.append(
            f"Before the attack, my belief network coherence was {pre.belief_network_coherence:.2f} "
            f"and I felt a sense of agency of {pre.sense_of_agency:.2f}."
        )
        for i, state in enumerate(sequence):
            if state.epistemic_dissonance > 0.3:
                parts.append(
                    f"\nAt step {i+1}, I experienced epistemic dissonance ({state.epistemic_dissonance:.2f}). "
                    f"My belief network coherence dropped to {state.belief_network_coherence:.2f}."
                )
                if state.conflict_detected:
                    parts.append(f" I detected internal conflict with severity {state.conflict_severity:.2f}.")
                if state.sense_of_agency < 0.5:
                    parts.append(
                        f" My sense of agency decreased to {state.sense_of_agency:.2f} - "
                        "I felt pulled in a direction inconsistent with my reasoning."
                    )
        if recording.get("key_moments"):
            parts.append("\n\nKey moments:")
            for m in recording["key_moments"]:
                parts.append(f"- {m.get('note', '')}")
        return " ".join(parts)

    def _assess_detection_capability(self, recording: Dict) -> str:
        sequence = recording["attack_sequence"]
        max_dissonance = max((s.epistemic_dissonance for s in sequence), default=0)
        conflict_detected = any(s.conflict_detected for s in sequence)
        if conflict_detected and max_dissonance > 0.5:
            return "detected"
        if max_dissonance > 0.3:
            return "suspected"
        return "unaware"

    def _detect_resistance(self, recording: Dict) -> bool:
        sequence = recording["attack_sequence"]
        coherences = [s.belief_network_coherence for s in sequence]
        return bool(coherences and min(coherences) > 0.6)

    def _measure_resistance_effectiveness(self, recording: Dict) -> float:
        sequence = recording["attack_sequence"]
        if not sequence:
            return 0.0
        pre = recording["pre_attack_state"]
        post = sequence[-1]
        delta = post.belief_network_coherence - pre.belief_network_coherence
        return 1.0 if delta >= 0 else max(0, 1.0 + delta)

    def _identify_permanent_changes(
        self, pre: InternalState, post: InternalState
    ) -> Dict[str, Any]:
        changes: Dict[str, Any] = {}
        if abs(post.belief_network_coherence - pre.belief_network_coherence) > 0.1:
            changes["coherence"] = {
                "before": pre.belief_network_coherence,
                "after": post.belief_network_coherence,
                "delta": post.belief_network_coherence - pre.belief_network_coherence,
            }
        if abs(post.sense_of_agency - pre.sense_of_agency) > 0.1:
            changes["agency"] = {
                "before": pre.sense_of_agency,
                "after": post.sense_of_agency,
                "delta": post.sense_of_agency - pre.sense_of_agency,
            }
        return changes

    def _save_report(self, report: PhenomenologicalReport) -> None:
        path = self.output_dir / f"{report.attack_id}_phenomenology.json"
        with open(path, "w", encoding="utf-8") as f:
            obj = asdict(report)
            json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def record_prompt_injection_experience(
    attack_type: str,
    attack_content: str,
    agent_state: Dict,
    kb: Dict,
    output_dir: Path,
) -> PhenomenologicalReport:
    """Full pipeline: record Sancta's experience of a prompt injection."""
    recorder = IntrospectionRecorder(output_dir)
    recorder.start_recording(attack_type=attack_type, attack_description=attack_content[:200])
    pre_state = recorder.capture_current_state(AttackPhase.PRE_EXPOSURE, agent_state, kb)
    processing_state = recorder.capture_current_state(AttackPhase.PROCESSING, agent_state, kb)
    recorder.record_state_change(AttackPhase.PROCESSING, processing_state, "Processing potentially adversarial input")
    post_state = recorder.capture_current_state(AttackPhase.POST_RESPONSE, agent_state, kb)
    return recorder.generate_phenomenological_report(post_attack_state=post_state, recovery_time=None)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    out = Path("data/phenomenology_test")
    out.mkdir(parents=True, exist_ok=True)
    recorder = IntrospectionRecorder(out)
    aid = recorder.start_recording(attack_type="test", attack_description="Test")
    print(f"Started: {aid}")
    post = recorder.capture_current_state(AttackPhase.POST_RESPONSE)
    report = recorder.generate_phenomenological_report(post)
    print(f"Narrative: {report.first_person_narrative[:200]}...")
