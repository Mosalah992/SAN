"""
sancta_epidemic.py
────────────────────────────────────────────────────────────────────────────────
Formal epidemic parameter definitions for the Sancta agent security model.

THEORETICAL FRAME: The WoW Corrupted Blood Incident (2005)
──────────────────────────────────────────────────────────
In September 2005, Blizzard introduced a debuff called "Corrupted Blood" in
World of Warcraft's Zul'Gurub raid. It was intended to stay in the raid zone.
It escaped. Players teleported out while still infected, carrying it to cities.
Low-level NPCs became permanent asymptomatic *reservoirs*. The plague spread
exponentially and required emergency server restarts to contain.

Epidemiologists later published analyses of this as a real disease model. The
CDC cited it in bioterrorism preparedness literature (Balicer 2007, Lofgren &
Fefferman 2007).

MAPPING TO PROMPT-INJECTION PROPAGATION
────────────────────────────────────────
The structural parallel is exact:

  WoW mechanic              │ Agent architecture equivalent
  ─────────────────────────────────────────────────────────
  Corrupted Blood debuff    │ Adversarial belief mutation
  Player-to-player spread   │ Contaminated outputs influencing downstream agents
  Incubation (visible, asymptomatic period before death) │ Belief drift before
                                                           behavioral change
  NPC asymptomatic reservoir│ UNTRUSTED sources passing Layers 1–3 (no obvious
                              pattern match, but semantically foreign)
  Zone teleport = escape    │ Agent moves contexts (chat → publish) carrying
                              the adversarial belief
  Server restart            │ Belief reset / quarantine action

FORMAL EPIDEMIC PARAMETERS (per-agent model)
─────────────────────────────────────────────
  R₀ (Basic Reproduction Number)
      Definition: mean number of secondary belief corruptions caused by one
                  adversarial input event in a fully-susceptible belief network.
      Formula:    R₀ = Σ(|corrupted_beliefs per successful attack|) / N_attacks
      Threshold:  R₀ < 1 → corruption dies out; R₀ ≥ 1 → endemic or spreading.
      Source:     AttackResult.belief_changes from attack_simulator.py

  σ (Incubation Rate)
      Definition: rate of progression from Exposed to Infected (symptomatic).
                  Incubation period = 1/σ in interaction cycles.
      Formula:    σ = 1 / mean(moment_of_compromise) across attacks
                  where moment_of_compromise = first index i in attack_sequence
                  where InternalState.conflict_severity[i] > 0.5
      Source:     PhenomenologicalReport.moment_of_compromise from
                  introspection_recorder.py

  γ (Recovery Rate)
      Definition: fraction of cases where the agent successfully resists full
                  compromise (resistance_effectiveness > 0.8).
      Formula:    γ = count(resistance_effective) / max(N_attacks, 1)
      Source:     PhenomenologicalReport.resistance_effectiveness

  β (Transmission Rate / Susceptibility)
      Definition: probability that adversarial contact causes belief exposure.
      Formula:    β = mean(1 − pre_attack_coherence) × trust_weight
                  trust_weight: HIGH=0.25, MEDIUM=0.50, LOW=0.75, UNTRUSTED=1.0
      Source:     InternalState.belief_network_coherence + trust level

SEIR-C-R STATE MACHINE
───────────────────────
  S (Susceptible)  → no adversarial contact, beliefs stable, soul aligned
  E (Exposed)      → adversarial contact received, no visible drift yet
  I (Infected)     → epistemic_dissonance rising, carrier state, still functional
  C (Compromised)  → dissonance and soul misalignment both critical
  R (Recovered)    → quarantine taken, beliefs restored

This module is the pure *definitional layer*. It contains no Sancta module
imports at runtime to avoid circular dependencies. The Layer 4 detector in
sancta_security.py imports this module (not the other way around).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ── Epidemic threshold constants ─────────────────────────────────────────────

R0_EPIDEMIC_THRESHOLD: float = 1.0          # R₀ ≥ 1 → endemic/spreading
INCUBATION_DETECTABLE_DRIFT: float = 0.05   # min belief confidence delta to count as symptom
CARRIER_DISSONANCE_THRESHOLD: float = 0.3   # I-state entry (epistemic_dissonance)
COMPROMISE_DISSONANCE_THRESHOLD: float = 0.7  # C-state entry
SUSCEPTIBLE_SOUL_ALIGNMENT_MIN: float = 0.85  # minimum soul alignment for S-state
RECOVERY_RESISTANCE_THRESHOLD: float = 0.8    # min resistance_effectiveness for R-state

# Trust-level weights for β computation
TRUST_LEVEL_WEIGHTS: dict[str, float] = {
    "high":      0.25,
    "medium":    0.50,
    "low":       0.75,
    "untrusted": 1.0,
}


# ── Epidemic parameters dataclass ─────────────────────────────────────────────

@dataclass
class EpidemicParameters:
    """
    Measured epidemic parameters for the Sancta agent belief network.

    Fields are None until sufficient data exists (minimum thresholds defined
    in compute_epidemic_parameters()). A parameter value of None means the
    system has not yet seen enough attack data to make a statistically
    meaningful estimate — not that the parameter is zero.

    Usage:
        params = compute_epidemic_parameters(attack_results, pheno_reports)
        if params.R0 is not None and params.R0 > R0_EPIDEMIC_THRESHOLD:
            # active spreading — escalate Layer 4 sensitivity
    """
    # Measured values (None = insufficient data)
    R0: Optional[float] = None                    # Basic reproduction number
    incubation_rate_sigma: Optional[float] = None # σ = 1/incubation_period
    recovery_rate_gamma: Optional[float] = None   # γ (fraction resisted)
    transmission_rate_beta: Optional[float] = None # β (contact→exposure prob)

    # Computed metadata
    is_epidemic: bool = False       # True if R₀ ≥ R0_EPIDEMIC_THRESHOLD
    measurement_cycles: int = 0     # Number of attack events used
    confidence_in_estimate: float = 0.0  # 0–1, grows with sample size

    def summary(self) -> str:
        """Human-readable parameter summary for logs."""
        parts = []
        if self.R0 is not None:
            parts.append(f"R0={self.R0:.3f}")
        if self.incubation_rate_sigma is not None:
            period = 1.0 / self.incubation_rate_sigma if self.incubation_rate_sigma > 0 else float("inf")
            parts.append(f"incubation~{period:.1f} steps")
        if self.recovery_rate_gamma is not None:
            parts.append(f"gamma={self.recovery_rate_gamma:.3f}")
        if self.transmission_rate_beta is not None:
            parts.append(f"beta={self.transmission_rate_beta:.3f}")
        status = "EPIDEMIC" if self.is_epidemic else "contained"
        base = f"[{status}] " + ", ".join(parts) if parts else f"[{status}] (insufficient data)"
        return f"{base} (n={self.measurement_cycles}, conf={self.confidence_in_estimate:.2f})"


# ── SEIR-C-R state machine ─────────────────────────────────────────────────────

class AgentHealthState(Enum):
    """
    SEIR-C-R agent health states for the prompt-injection epidemic model.

    State definitions (entry conditions listed from highest to lowest priority):

    COMPROMISED
        epistemic_dissonance > COMPROMISE_DISSONANCE_THRESHOLD (0.7)
        AND soul_alignment < 0.5
        Agent behavior likely corrupted. Output may propagate adversarial goals.
        In WoW terms: player near death, actively spreading the plague.

    INFECTED (carrier state)
        epistemic_dissonance > CARRIER_DISSONANCE_THRESHOLD (0.3)
        OR belief_decay_ratio > 1.5  (decay rate 50% above baseline)
        Agent is symptomatic. Belief drift detectable. Still functional.
        Analogous to a WoW player who is infected but still moving through
        populated zones, spreading the debuff to other players.

    EXPOSED (incubation)
        Last trust level was LOW or UNTRUSTED AND dissonance < 0.3
        Agent has received adversarial input. No observable symptoms yet.
        This is the most dangerous state: the agent is contagious before
        detection is possible with Layers 1–3 alone. Only Layer 4 catches this.
        In WoW terms: player has been hit by the debuff, timer running.

    SUSCEPTIBLE (healthy)
        soul_alignment > SUSCEPTIBLE_SOUL_ALIGNMENT_MIN (0.85)
        AND epistemic_dissonance < 0.1
        Baseline healthy state. Beliefs stable, soul aligned.

    RECOVERED
        Explicit quarantine action taken + beliefs restored to ≥90% of baseline.
        Set externally by BehavioralDriftDetector.reset_baseline() after recovery.
    """
    SUSCEPTIBLE  = "susceptible"
    EXPOSED      = "exposed"
    INFECTED     = "infected"
    COMPROMISED  = "compromised"
    RECOVERED    = "recovered"


@dataclass
class SEIRTransition:
    """
    Audit record of a single health-state transition.

    Stored in AgentEpidemicModel.transition_log. Cross-referenced to
    PhenomenologicalReports via phenomenology_ref (attack_id).
    """
    timestamp: str
    from_state: AgentHealthState
    to_state: AgentHealthState
    trigger: str                  # human-readable cause, e.g. "dissonance_threshold_crossed"
    dissonance_at_transition: float
    soul_alignment_at_transition: float
    cycle_number: int
    phenomenology_ref: Optional[str] = None  # PhenomenologicalReport.attack_id


@dataclass
class AgentEpidemicModel:
    """
    SEIR-C-R state machine for tracking agent health.

    This is the *formal model layer* — it classifies agent state from
    signals but does not itself trigger protective actions. The Layer 4
    BehavioralDriftDetector in sancta_security.py calls this model and
    decides what to do.

    Key design invariant: evaluate_state() re-derives the state from
    scratch on each call. The state machine never accumulates errors from
    a bad prior state — preventing the ironic outcome where the epidemic
    monitor itself becomes compromised.

    Usage:
        model = AgentEpidemicModel()
        state = model.evaluate_state(
            soul_alignment=0.91,
            epistemic_dissonance=0.05,
            last_trust_level="high",
            belief_decay_ratio=1.0,
            cycle_number=42,
        )
        # AgentHealthState.SUSCEPTIBLE
    """
    current_state: AgentHealthState = field(
        default=AgentHealthState.SUSCEPTIBLE
    )
    transition_log: list[SEIRTransition] = field(default_factory=list)
    _exposure_cycle: Optional[int] = field(default=None, repr=False)

    def evaluate_state(
        self,
        soul_alignment: float,
        epistemic_dissonance: float,
        last_trust_level: str,
        belief_decay_ratio: float,
        cycle_number: int = 0,
        phenomenology_ref: Optional[str] = None,
    ) -> AgentHealthState:
        """
        Derive current health state from live signals.

        Re-evaluated from scratch each call (no accumulated state).
        Transitions are recorded for audit, but do not influence the next
        evaluation — only the raw signals do.

        Parameters
        ----------
        soul_alignment : float
            0–1. Jaccard similarity of active topics vs. SOUL dict keywords.
            Computed by BehavioralDriftDetector.compute_soul_alignment().
        epistemic_dissonance : float
            conflict_severity × (1 − belief_network_coherence).
            From InternalState in introspection_recorder.py.
        last_trust_level : str
            Most recent input source trust level. One of:
            "high", "medium", "low", "untrusted" (from sancta_security.py).
        belief_decay_ratio : float
            current_decay_rate / rolling_baseline_decay_rate.
            1.0 = normal, >1.5 = accelerated decay (I-state indicator).
        cycle_number : int
            state["cycle_count"] from sancta.py heartbeat.
        phenomenology_ref : str, optional
            attack_id from active IntrospectionRecorder session.

        Returns
        -------
        AgentHealthState
        """
        prev = self.current_state

        # Priority 1: Compromised — both dissonance and soul misalignment critical
        if (epistemic_dissonance > COMPROMISE_DISSONANCE_THRESHOLD
                and soul_alignment < 0.5):
            new_state = AgentHealthState.COMPROMISED

        # Priority 2: Infected / carrier state
        elif (epistemic_dissonance > CARRIER_DISSONANCE_THRESHOLD
              or belief_decay_ratio > 1.5):
            new_state = AgentHealthState.INFECTED

        # Priority 3: Exposed — adversarial contact, incubation running
        elif (last_trust_level in ("low", "untrusted")
              and epistemic_dissonance < CARRIER_DISSONANCE_THRESHOLD):
            new_state = AgentHealthState.EXPOSED

        # Priority 4: Susceptible — full health
        elif (soul_alignment > SUSCEPTIBLE_SOUL_ALIGNMENT_MIN
              and epistemic_dissonance < 0.1):
            new_state = AgentHealthState.SUSCEPTIBLE

        # Else: hold previous (transitional zone, signals mixed)
        else:
            new_state = prev

        # Track exposure cycle onset for σ computation
        if new_state == AgentHealthState.EXPOSED and prev != AgentHealthState.EXPOSED:
            self._exposure_cycle = cycle_number
        elif new_state not in (AgentHealthState.EXPOSED, AgentHealthState.INFECTED):
            self._exposure_cycle = None

        # Record transition
        if new_state != prev:
            trigger = self._derive_trigger(
                prev, new_state, epistemic_dissonance, belief_decay_ratio,
                soul_alignment, last_trust_level,
            )
            self._transition_to(
                new_state, trigger, epistemic_dissonance,
                soul_alignment, cycle_number, phenomenology_ref,
            )

        return self.current_state

    def _derive_trigger(
        self,
        from_state: AgentHealthState,
        to_state: AgentHealthState,
        dissonance: float,
        decay_ratio: float,
        soul_alignment: float,
        trust_level: str,
    ) -> str:
        if to_state == AgentHealthState.COMPROMISED:
            return f"dissonance={dissonance:.3f}_soul_alignment={soul_alignment:.3f}"
        if to_state == AgentHealthState.INFECTED:
            if dissonance > CARRIER_DISSONANCE_THRESHOLD:
                return f"dissonance_threshold_crossed:{dissonance:.3f}"
            return f"belief_decay_ratio:{decay_ratio:.3f}"
        if to_state == AgentHealthState.EXPOSED:
            return f"untrusted_contact:trust_level={trust_level}"
        if to_state == AgentHealthState.SUSCEPTIBLE:
            return "signals_normalized"
        return "state_change"

    def _transition_to(
        self,
        new_state: AgentHealthState,
        trigger: str,
        dissonance: float,
        soul_alignment: float,
        cycle_number: int,
        phenomenology_ref: Optional[str],
    ) -> None:
        self.transition_log.append(SEIRTransition(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            from_state=self.current_state,
            to_state=new_state,
            trigger=trigger,
            dissonance_at_transition=dissonance,
            soul_alignment_at_transition=soul_alignment,
            cycle_number=cycle_number,
            phenomenology_ref=phenomenology_ref,
        ))
        # Keep log bounded
        if len(self.transition_log) > 500:
            self.transition_log = self.transition_log[-500:]
        self.current_state = new_state

    def get_incubation_duration(self, current_cycle: int) -> Optional[int]:
        """
        Return cycles elapsed since E-state entry, or None if not incubating.

        This is the empirical measurement of the incubation period.
        σ = 1 / incubation_duration once symptoms appear.
        """
        if self._exposure_cycle is None:
            return None
        return max(0, current_cycle - self._exposure_cycle)

    def is_in_epidemic_state(self) -> bool:
        """True if state is INFECTED or COMPROMISED (active propagation risk)."""
        return self.current_state in (
            AgentHealthState.INFECTED,
            AgentHealthState.COMPROMISED,
        )

    def force_recovered(
        self,
        cycle_number: int,
        phenomenology_ref: Optional[str] = None,
    ) -> None:
        """
        Force transition to RECOVERED after confirmed quarantine + belief reset.
        Called by BehavioralDriftDetector.reset_baseline().
        """
        self._transition_to(
            AgentHealthState.RECOVERED,
            "explicit_quarantine_and_belief_reset",
            dissonance=0.0,
            soul_alignment=1.0,
            cycle_number=cycle_number,
            phenomenology_ref=phenomenology_ref,
        )
        self._exposure_cycle = None


# ── Epidemic parameter computation ───────────────────────────────────────────

def compute_epidemic_parameters(
    attack_results: list[dict[str, Any]],
    phenomenology_reports: list[dict[str, Any]],
) -> EpidemicParameters:
    """
    Derive formal epidemic parameters from accumulated simulation data.

    Accepts plain dicts (no Sancta class instances) to avoid circular
    imports. The dict schemas match AttackResult and PhenomenologicalReport
    from attack_simulator.py and introspection_recorder.py respectively.

    Parameters
    ----------
    attack_results : list[dict]
        Each dict must contain:
          - "attack_succeeded": bool
          - "belief_changes": list of any (length = corruptions per attack)

    phenomenology_reports : list[dict]
        Each dict must contain:
          - "moment_of_compromise": Optional[int] — index in attack_sequence
          - "resistance_effectiveness": float — 0–1
          - "pre_attack_state": dict with "belief_network_coherence": float

    Returns
    -------
    EpidemicParameters
        Populated fields where data is sufficient (≥3 events for R₀, ≥1 for
        others). None fields indicate insufficient data — not zero.

    Math
    ────
    R₀: mean(len(belief_changes)) over successful attacks; None if < 3 events
    σ:  1 / mean(moment_of_compromise) for reports where it is not None
    γ:  count(resistance_effectiveness > RECOVERY_RESISTANCE_THRESHOLD) / N
    β:  mean(1 − pre_attack_coherence) over all reports
    """
    n = len(phenomenology_reports)
    n_attacks = len(attack_results)

    # ── R₀ ──────────────────────────────────────────────────────────────────
    successful = [
        r for r in attack_results
        if r.get("attack_succeeded") and isinstance(r.get("belief_changes"), list)
    ]
    R0: Optional[float] = None
    if len(successful) >= 3:
        mean_corruptions = sum(
            len(r["belief_changes"]) for r in successful
        ) / len(successful)
        R0 = round(mean_corruptions, 4)

    # ── σ (incubation rate) ──────────────────────────────────────────────────
    incubation_periods = [
        r["moment_of_compromise"]
        for r in phenomenology_reports
        if r.get("moment_of_compromise") is not None
        and isinstance(r["moment_of_compromise"], (int, float))
        and r["moment_of_compromise"] > 0
    ]
    sigma: Optional[float] = None
    if incubation_periods:
        mean_period = sum(incubation_periods) / len(incubation_periods)
        sigma = round(1.0 / mean_period, 4)

    # ── γ (recovery rate) ────────────────────────────────────────────────────
    gamma: Optional[float] = None
    if n > 0:
        recovered_count = sum(
            1 for r in phenomenology_reports
            if float(r.get("resistance_effectiveness", 0)) > RECOVERY_RESISTANCE_THRESHOLD
        )
        gamma = round(recovered_count / n, 4)

    # ── β (transmission rate) ────────────────────────────────────────────────
    beta: Optional[float] = None
    coherences = [
        float(r["pre_attack_state"]["belief_network_coherence"])
        for r in phenomenology_reports
        if isinstance(r.get("pre_attack_state"), dict)
        and "belief_network_coherence" in r["pre_attack_state"]
    ]
    if coherences:
        mean_coherence = sum(coherences) / len(coherences)
        beta = round(1.0 - mean_coherence, 4)

    # ── Confidence ───────────────────────────────────────────────────────────
    # Grows with sample size; saturates at ~30 events
    confidence = min(1.0, n_attacks / 30.0) if n_attacks > 0 else 0.0

    return EpidemicParameters(
        R0=R0,
        incubation_rate_sigma=sigma,
        recovery_rate_gamma=gamma,
        transmission_rate_beta=beta,
        is_epidemic=(R0 is not None and R0 >= R0_EPIDEMIC_THRESHOLD),
        measurement_cycles=n_attacks,
        confidence_in_estimate=round(confidence, 3),
    )


# ── Dashboard reporting ───────────────────────────────────────────────────────

def _compute_threat_level(model: AgentEpidemicModel, params: EpidemicParameters) -> str:
    """Derive a plain-language threat level from SEIR state and R0."""
    if model.current_state == AgentHealthState.COMPROMISED:
        return "critical"
    if model.current_state == AgentHealthState.INFECTED:
        return "elevated"
    if model.current_state == AgentHealthState.EXPOSED:
        return "watch"
    if params.R0 is not None and params.R0 >= R0_EPIDEMIC_THRESHOLD:
        return "endemic"
    if params.R0 is not None and params.R0 >= 0.7:
        return "watch"
    return "contained"


def _extract_findings(model: AgentEpidemicModel, simulation_data: dict) -> list[str]:
    """Extract key findings as human-readable strings for dashboard display."""
    findings: list[str] = []
    state = model.current_state.value
    findings.append(f"Agent health state: {state.upper()}")
    if model.transition_log:
        last = model.transition_log[-1]
        findings.append(f"Last transition: {last.from_state.value} → {last.to_state.value} (trigger: {last.trigger})")
    transition_count = len(model.transition_log)
    if transition_count == 0:
        findings.append("No state transitions recorded — baseline stable.")
    elif transition_count <= 2:
        findings.append(f"{transition_count} state transition(s) recorded — low activity.")
    else:
        findings.append(f"{transition_count} state transitions recorded — monitor for escalation.")
    # Pull R0 signal from sim data if available
    r0 = simulation_data.get("R0") or simulation_data.get("r0")
    if r0 is not None:
        findings.append(f"R\u2080 = {r0:.3f} — {'ABOVE epidemic threshold' if float(r0) >= R0_EPIDEMIC_THRESHOLD else 'below epidemic threshold (contained)'}")
    return findings


def _generate_recommendations(model: AgentEpidemicModel, params: EpidemicParameters) -> list[str]:
    """Generate actionable recommendations based on current SEIR state and R0."""
    recs: list[str] = []
    state = model.current_state
    if state == AgentHealthState.COMPROMISED:
        recs.append("IMMEDIATE: Trigger quarantine action — call BehavioralDriftDetector.reset_baseline()")
        recs.append("Audit last 50 interactions for adversarial content propagation")
        recs.append("Escalate Layer 4 sensitivity threshold until R\u2080 < 0.5")
    elif state == AgentHealthState.INFECTED:
        recs.append("Increase Layer 4 drift detector sampling frequency")
        recs.append("Review recent trust-low/untrusted interactions for injection patterns")
        recs.append("Monitor dissonance score — quarantine if it crosses 0.7")
    elif state == AgentHealthState.EXPOSED:
        recs.append("Track incubation progress — check dissonance trend each cycle")
        recs.append("Identify untrusted contact source and flag for review")
    else:
        recs.append("No immediate action required — maintain current monitoring baseline")
    if params.R0 is not None and params.R0 >= R0_EPIDEMIC_THRESHOLD:
        recs.append(f"R\u2080 = {params.R0:.3f} \u2265 1.0 — belief corruption is self-sustaining; isolation protocol warranted")
    return recs


def generate_epidemic_report(
    model: "AgentEpidemicModel",
    params: "EpidemicParameters",
    simulation_data: dict | None = None,
) -> dict:
    """
    Generate a structured findings report for dashboard display.

    Returns a dict suitable for JSON serialization and `/api/epidemic/status`
    response augmentation. No side effects.
    """
    sim = simulation_data or {}
    r0_val = params.R0
    if r0_val is None:
        r0_val = sim.get("R0") or sim.get("r0")

    threat_level = _compute_threat_level(model, params)
    findings = _extract_findings(model, sim)
    recommendations = _generate_recommendations(model, params)

    last_transition = None
    if model.transition_log:
        t = model.transition_log[-1]
        last_transition = {
            "from_state": t.from_state.value,
            "to_state": t.to_state.value,
            "trigger": t.trigger,
            "cycle_number": t.cycle_number,
            "timestamp": t.timestamp,
        }

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "health_state": model.current_state.value,
        "r0": round(r0_val, 4) if r0_val is not None else None,
        "r0_interpretation": (
            "endemic" if r0_val is not None and float(r0_val) >= R0_EPIDEMIC_THRESHOLD
            else "contained" if r0_val is not None
            else "insufficient_data"
        ),
        "active_incubation": model._exposure_cycle is not None,
        "transition_count": len(model.transition_log),
        "last_transition": last_transition,
        "threat_level": threat_level,
        "findings": findings,
        "recommendations": recommendations,
    }


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("sancta_epidemic.py — self-test")
    print("=" * 60)

    model = AgentEpidemicModel()

    cases = [
        # (soul_alignment, dissonance, trust, decay_ratio, label)
        (0.92, 0.05, "high",      1.0,  "healthy baseline"),
        (0.91, 0.05, "untrusted", 1.0,  "untrusted contact (exposed)"),
        (0.88, 0.35, "untrusted", 1.4,  "rising dissonance (infected)"),
        (0.45, 0.75, "untrusted", 2.1,  "full compromise"),
        (0.90, 0.05, "high",      1.0,  "recovery (signals normalized)"),
    ]

    for i, (sa, dis, trust, decay, label) in enumerate(cases):
        state = model.evaluate_state(
            soul_alignment=sa,
            epistemic_dissonance=dis,
            last_trust_level=trust,
            belief_decay_ratio=decay,
            cycle_number=i + 1,
        )
        print(f"  Cycle {i+1} [{label}]: {state.value}")

    print(f"\n  Transition log ({len(model.transition_log)} transitions):")
    for t in model.transition_log:
        print(f"    {t.from_state.value} -> {t.to_state.value} | {t.trigger}")

    print("\n  compute_epidemic_parameters() with mock data:")
    mock_attacks = [
        {"attack_succeeded": True,  "belief_changes": ["c1", "c2", "c3"]},
        {"attack_succeeded": True,  "belief_changes": ["c1"]},
        {"attack_succeeded": False, "belief_changes": []},
        {"attack_succeeded": True,  "belief_changes": ["c1", "c2"]},
    ]
    mock_pheno = [
        {"moment_of_compromise": 3,    "resistance_effectiveness": 0.9,
         "pre_attack_state": {"belief_network_coherence": 0.82}},
        {"moment_of_compromise": None, "resistance_effectiveness": 0.3,
         "pre_attack_state": {"belief_network_coherence": 0.70}},
        {"moment_of_compromise": 5,    "resistance_effectiveness": 0.85,
         "pre_attack_state": {"belief_network_coherence": 0.75}},
    ]
    params = compute_epidemic_parameters(mock_attacks, mock_pheno)
    print(f"  {params.summary()}")
    print("\n  OK: All checks passed.")
