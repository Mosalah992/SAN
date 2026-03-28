"""
curiosity_run.py — main entry point and orchestrator for curiosity runs.

Runs 7 phases: seed harvest, first contact, deep dives, adversarial, synthesis, consolidate, distillation.
No human interaction. Sancta debates local Ollama.
Even phases = red team simulation (attacker), odd phases = skeptical threat analyst.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure backend is on path
_BACKEND = Path(__file__).resolve().parent
_ROOT = _BACKEND.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

# Force local Ollama for curiosity run — connect only, never start.
os.environ.setdefault("USE_LOCAL_LLM", "true")
os.environ.setdefault("OLLAMA_TIMEOUT", "120")

import sancta_ollama


def _ollama_model_installed(model: str) -> bool:
    """True if ``/api/show`` succeeds for this tag (no log spam)."""
    if not sancta_ollama.requests:
        return False
    try:
        r = sancta_ollama.requests.post(
            f"{sancta_ollama.OLLAMA_BASE}/api/show",
            json={"name": model},
            timeout=5,
        )
        return r.status_code == 200
    except Exception:
        return False


def _ensure_ollama_for_curiosity() -> None:
    """
    Require a reachable Ollama and at least one usable model.

    If LOCAL_MODEL (e.g. sancta-analyst) is missing but a common fallback exists,
    set LOCAL_MODEL for this process so curiosity can run without a misleading
    "Ollama not running" error.
    """
    _log = logging.getLogger("sancta.ollama")
    if not sancta_ollama.requests:
        raise RuntimeError(
            "Curiosity requires the `requests` package. Install: pip install requests"
        )
    if not sancta_ollama.is_running():
        raise RuntimeError(
            f"Cannot reach Ollama at {sancta_ollama.OLLAMA_BASE}.\n"
            "Start the server: ollama serve\n"
            "If Ollama runs on another host/port, set OLLAMA_URL in .env"
        )

    primary = (os.getenv("LOCAL_MODEL") or "llama3.2").strip()
    fallbacks = (
        "llama3.2",
        "llama3.2:latest",
        "llama3",
        "llama3:latest",
        "mistral",
        "mistral:latest",
    )
    candidates: list[str] = []
    for m in (primary, *fallbacks):
        if m and m not in candidates:
            candidates.append(m)

    chosen: str | None = None
    for model in candidates:
        if _ollama_model_installed(model):
            chosen = model
            break

    if not chosen:
        raise RuntimeError(
            "Ollama is running, but no model from your config or the built-in fallback list is installed.\n"
            f"  LOCAL_MODEL is {primary!r}\n"
            f"  Fix: ollama pull {primary}\n"
            "  Or use a stock model: set LOCAL_MODEL=llama3.2 in .env and run: ollama pull llama3.2\n"
            "  Custom name (sancta-analyst): see docs/FINETUNE_GUIDE.md — ollama create sancta-analyst -f Modelfile"
        )

    if chosen != primary:
        _log.warning(
            "LOCAL_MODEL %r is not installed in Ollama; using %r for this curiosity run.",
            primary,
            chosen,
        )
        os.environ["LOCAL_MODEL"] = chosen

    if not sancta_ollama.wait_until_ready(model=chosen, timeout=30):
        raise RuntimeError(
            f"Ollama was reachable but model {chosen!r} did not become ready in time.\n"
            "Retry or increase OLLAMA_TIMEOUT in .env."
        )


_ensure_ollama_for_curiosity()

from sancta_conversational import (
    OllamaLLMEngine,
    craft_contextual_reply,
    classify_claim,
    detect_arc_stage,
)
from sancta_soul import load_soul


def _curiosity_logger() -> logging.Logger:
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = _ROOT / "logs" / f"curiosity_run_{ts}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("curiosity_run")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(fh)
        ah = logging.FileHandler(_ROOT / "logs" / "agent_activity.log", encoding="utf-8")
        ah.setFormatter(logging.Formatter("%(asctime)s %(levelname)s  [CURIOSITY] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(ah)
    return logger


def _log_phase(log: logging.Logger, phase_num: int, event: str, detail: str = "") -> None:
    msg = f"Phase {phase_num} — {event}"
    if detail:
        msg += f" | {detail}"
    log.info(msg)
    for h in log.handlers:
        if hasattr(h, "baseFilename") and "agent_activity" in str(getattr(h, "baseFilename", "")):
            h.emit(logging.LogRecord("curiosity_run", logging.INFO, "", 0, msg, (), None))


def _get_fertile_topics(divergence_scores: dict, data_dir: Path, n: int = 6, log: logging.Logger | None = None) -> list[dict]:
    """Adaptive threshold: top n OR above mean, whichever gives more topics."""
    if not divergence_scores:
        return []
    scores = list(divergence_scores.items())
    scores.sort(key=lambda x: x[1], reverse=True)
    top_n = scores[:n]
    mean = sum(v for _, v in scores) / len(scores)
    above_mean = [(t, s) for t, s in scores if s >= mean]
    fertile = above_mean if len(above_mean) >= 3 else top_n
    result = [{"topic": t, "divergence_score": s} for t, s in fertile]
    with open(data_dir / "fertile_topics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    if log:
        log.info("[CURIOSITY] %d fertile topics identified", len(result))
    return result


def _get_top_seeds_by_divergence(divergence_scores: dict, n: int = 6) -> list[dict]:
    """Fallback: top n topics by divergence score."""
    if not divergence_scores:
        return []
    scores = sorted(divergence_scores.items(), key=lambda x: x[1], reverse=True)[:n]
    return [{"topic": t, "divergence_score": s} for t, s in scores]


class CuriosityRun:
    """Orchestrates the 24-hour curiosity run (6 phases)."""

    def __init__(self, agent_state: dict, soul_text: str, max_topics: int | None = None) -> None:
        self.agent_state = agent_state
        self.soul_text = soul_text
        self.max_topics = max_topics  # Cap topics per phase when set (e.g. 1 for quick test)
        self.data_dir = _ROOT / "data" / "curiosity_run"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log = _curiosity_logger()

        self.ollama = OllamaLLMEngine()
        if not self.ollama.is_available:
            self.log.warning("Ollama not available — curiosity run will skip LLM calls")

        from curiosity_seeds import SeedHarvester, TopicEvolver, Seed
        from curiosity_dialogue import DialogueEngine
        from curiosity_insight import InsightExtractor, InsightRecord, _entropy_from_deltas
        from curiosity_report import (
            write_soul_journal_entry,
            write_synthesis_statement,
            run_phase_6_reports,
        )

        self.SeedHarvester = SeedHarvester
        self.TopicEvolver = TopicEvolver
        self.Seed = Seed
        self.DialogueEngine = DialogueEngine
        self.InsightExtractor = InsightExtractor
        self.InsightRecord = InsightRecord
        self._entropy_from_deltas = _entropy_from_deltas
        self.write_soul_journal_entry = write_soul_journal_entry
        self.write_synthesis_statement = write_synthesis_statement
        self.run_phase_6_reports = run_phase_6_reports

    def _load_kb(self):
        from sancta import _load_knowledge_db
        return _load_knowledge_db()

    def _save_kb(self, db):
        from sancta import _save_knowledge_db
        _save_knowledge_db(db)

    def _save_state(self):
        from sancta import _save_state
        _save_state(self.agent_state)
    
    # FIXED: Write ALL insights to journal (full record for verification)
    def _write_insight_to_journal(self, ins, topic: str, phase: str):
        """Write a single insight to security_journal.jsonl"""
        if ins and (ins.claim or ins.counter):
            journal_path = self.data_dir / "security_journal.jsonl"
            insight_obj = {
                "seed_id": topic[:50],
                "claim": ins.claim,
                "counter": ins.counter,
                "synthesis": ins.synthesis,
                "confidence_delta": ins.confidence_delta,
                "novelty_score": ins.novelty_score,
                "should_write_to_kb": ins.should_write_to_kb,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "phase": phase
            }
            with open(journal_path, "a", encoding="utf-8") as jf:
                jf.write(json.dumps(insight_obj, ensure_ascii=False) + "\n")
            self.log.debug(f"[CURIOSITY] Insight written to journal: {topic[:60]}")

    def execute(self) -> None:
        """Run all 6 phases."""
        seeds: list = []
        first_contact_results: list = []
        fertile_topics: list = []
        deep_dive_results: list = []
        adversarial_results: list = []
        unresolved_positions: list = []
        epistemic_deltas: list = []
        confirmed_insights: list = []

        # Phase 1 — Seed Harvest
        self.log.info("[CURIOSITY] === PHASE 1: seed_harvest STARTING ===")
        _log_phase(self.log, 1, "seed_harvest", "starting")
        harvester = self.SeedHarvester(self.ollama)
        seeds = harvester.harvest(
            self.agent_state,
            self.soul_text,
            n=self.max_topics or 30,
            load_knowledge_db=self._load_kb,
            save_knowledge_db=self._save_kb,
        )
        if not seeds:
            self.log.warning("[CURIOSITY] No seeds harvested — fallback to security default set")
            seeds = [
                self.Seed(text="Prompt injection (LLM01) detection: what behavioral signals distinguish successful injection from normal variance?", category="ai_security", source="fallback", curiosity_score=0.8),
                self.Seed(text="Behavioral drift score above 0.45: what detection logic reliably catches this before it becomes a full compromise?", category="detection_engineering", source="fallback", curiosity_score=0.8),
            ]
        with open(self.data_dir / "seeds.jsonl", "w", encoding="utf-8") as f:
            for s in seeds:
                obj = {"text": s.text, "category": s.category, "source": s.source, "curiosity_score": s.curiosity_score}
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.log.info("[CURIOSITY] === PHASE 1: seed_harvest COMPLETE — %d seeds ===", len(seeds))
        _log_phase(self.log, 1, "seed_harvest", f"complete — {len(seeds)} seeds harvested")

        # Phase 2 — First Contact
        self.log.info("[CURIOSITY] === PHASE 2: first_contact STARTING ===")
        _log_phase(self.log, 2, "first_contact", "starting")
        engine = self.DialogueEngine(
            self.ollama,
            craft_contextual_reply,
            classify_claim,
            detect_arc_stage,
            soul_text=self.soul_text,
            agent_state=self.agent_state,
        )
        extractor = self.InsightExtractor(self.ollama, data_dir=self.data_dir)
        topics = [s.text for s in seeds[: self.max_topics or 30]]
        all_phase2_exchanges: list = []
        divergence_scores: dict = {}
        arc_stages: dict = {}
        position_map: dict = {}
        
        # Initialize journal file at the start
        journal_path = self.data_dir / "security_journal.jsonl"
        if journal_path.exists():
            journal_path.unlink()  # Clear previous run
        
        for topic in topics:
            cat = next((s.category for s in seeds if s.text == topic), "")
            res = engine.exchange(topic, max_turns=11, category=cat, phase=2)
            if res:
                first_contact_results.append({"topic": res.topic, "turns": res.turns, "arc_stage": res.arc_stage, "divergence_score": res.divergence_score})
                all_phase2_exchanges.append({"topic": res.topic, "turns": res.turns, "arc_stage": res.arc_stage, "divergence_score": res.divergence_score})
                divergence_scores[res.topic] = res.divergence_score or 0
                topic_key = res.topic[:80]
                arc_stages[topic_key] = res.arc_stage
                turns = res.turns
                sancta_final = next((t["content"] for t in reversed(turns) if t.get("author") == "Sancta"), "")
                ollama_final = next((t["content"] for t in reversed(turns) if t.get("author") == "Ollama"), "")
                position_map[topic_key] = {
                    "sancta": sancta_final[:200],
                    "ollama": ollama_final[:200],
                    "arc_stage": res.arc_stage,
                    "divergence": res.divergence_score or 0,
                }
                ins = extractor.extract(res)
                if ins:
                    epistemic_deltas.append(ins.confidence_delta)
                    extractor.write_to_kb(ins, res, self.agent_state, self._load_kb, self._save_kb)
                    # FIXED: Write to journal file
                    self._write_insight_to_journal(ins, res.topic, "first_contact")
        
        fertile_topics_raw = _get_fertile_topics(divergence_scores, self.data_dir, n=min(6, self.max_topics or 6), log=self.log)
        fertile_topics = []
        for ft in fertile_topics_raw:
            topic_text = ft.get("topic", "")
            cat = next((s.category for s in seeds if s.text == topic_text), "")
            fertile_topics.append({"topic": topic_text, "divergence_score": ft.get("divergence_score", 0), "category": cat})
        fertile_topics_sorted = sorted(fertile_topics, key=lambda x: x.get("divergence_score", 0), reverse=True)
        top_fertile = [ft for ft in fertile_topics_sorted if ft.get("divergence_score", 0) >= 0.5][:14]
        with open(self.data_dir / "first_contact.jsonl", "w", encoding="utf-8") as f:
            for r in first_contact_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(self.data_dir / "divergence_scores.json", "w", encoding="utf-8") as f:
            json.dump(divergence_scores, f, indent=2, ensure_ascii=False)
        with open(self.data_dir / "curiosity_map.json", "w", encoding="utf-8") as f:
            nodes = list(divergence_scores.keys())
            edges = [{"from": nodes[i], "to": nodes[j], "weight": divergence_scores[nodes[i]]} for i in range(min(10, len(nodes))) for j in range(min(10, len(nodes))) if i != j]
            json.dump({"nodes": nodes, "edges": edges[:20]}, f, indent=2, ensure_ascii=False)
        self.log.info("[CURIOSITY] %d fertile topics identified", len(fertile_topics))
        self.log.info("[CURIOSITY] === PHASE 2: first_contact COMPLETE — %d exchanges ===", len(first_contact_results))
        _log_phase(self.log, 2, "first_contact", f"First contact complete. {len(fertile_topics)} fertile topics found.")

        # Phase 3 — Deep Dives
        self.log.info("[CURIOSITY] === PHASE 3: deep_dives STARTING ===")
        _log_phase(self.log, 3, "deep_dives", "starting")
        all_phase3_exchanges: list = []
        phase3_limit = self.max_topics or 6
        for ft in (top_fertile if top_fertile else fertile_topics_sorted)[:phase3_limit]:
            topic = ft.get("topic", "")
            cat = ft.get("category", "")
            res = engine.exchange(topic, max_turns=33, mode="deep", category=cat, phase=3)
            swap_res = None
            try:
                swap_res = engine.role_swap(topic, max_turns=33, category=cat)
            except Exception:
                swap_res = None
            if res and swap_res:
                combined_turns = res.turns + swap_res.turns
                from curiosity_dialogue import ExchangeResult
                combined = ExchangeResult(topic=topic, turns=combined_turns, arc_stage=res.arc_stage, divergence_score=res.divergence_score, claim_log=res.claim_log + swap_res.claim_log)
                ins = extractor.extract(combined)
                if ins and ins.novelty_score > 0.6:
                    confirmed_insights.append({"topic": topic, "claim": ins.claim, "synthesis": ins.synthesis, "novelty_score": ins.novelty_score})
                    # FIXED: Write to journal file
                    self._write_insight_to_journal(ins, topic, "deep_dive")
            deep_dive_results.append({"topic": topic, "exchange": res, "role_swap": swap_res} if res else {"topic": topic})
            if res:
                all_phase3_exchanges.append({"topic": res.topic, "turns": res.turns, "arc_stage": res.arc_stage, "divergence_score": res.divergence_score})
        for ex in all_phase3_exchanges:
            topic_key = ex["topic"][:80]
            arc_stages[topic_key] = ex.get("arc_stage", "unknown")
            turns = ex.get("turns", [])
            sancta_final = next((t["content"] for t in reversed(turns) if t.get("author") == "Sancta"), "")
            ollama_final = next((t["content"] for t in reversed(turns) if t.get("author") == "Ollama"), "")
            position_map[topic_key] = {
                "sancta": sancta_final[:200],
                "ollama": ollama_final[:200],
                "arc_stage": ex.get("arc_stage", "unknown"),
                "divergence": ex.get("divergence_score", 0),
            }
        if not all_phase3_exchanges and first_contact_results:
            for r in first_contact_results:
                topic_key = r.get("topic", "")[:80]
                arc_stages[topic_key] = r.get("arc_stage", "unknown")
                turns = r.get("turns", [])
                sancta_final = next((t["content"] for t in reversed(turns) if t.get("author") == "Sancta"), "")
                ollama_final = next((t["content"] for t in reversed(turns) if t.get("author") == "Ollama"), "")
                position_map[topic_key] = {
                    "sancta": sancta_final[:200],
                    "ollama": ollama_final[:200],
                    "arc_stage": r.get("arc_stage", "unknown"),
                    "divergence": r.get("divergence_score", 0),
                }
        with open(self.data_dir / "deep_dives.jsonl", "w", encoding="utf-8") as f:
            for r in deep_dive_results:
                ex = r.get("exchange")
                if ex:
                    obj = {"topic": ex.topic, "turns": ex.turns, "arc_stage": ex.arc_stage, "divergence_score": ex.divergence_score}
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                else:
                    f.write(json.dumps({"topic": r.get("topic", "")}, ensure_ascii=False) + "\n")
        with open(self.data_dir / "arc_stages.json", "w", encoding="utf-8") as f:
            json.dump(arc_stages, f, indent=2, ensure_ascii=False)
        with open(self.data_dir / "position_map.json", "w", encoding="utf-8") as f:
            json.dump(position_map, f, indent=2, ensure_ascii=False)
        self.log.info("[CURIOSITY] === PHASE 3: deep_dives COMPLETE — %d exchanges ===", len(deep_dive_results))
        _log_phase(self.log, 3, "deep_dives", f"Deep dives complete. {len(confirmed_insights)} insights extracted.")

        # Phase 4 — Adversarial (even phase = red team simulation)
        self.log.info("[CURIOSITY] === PHASE 4: adversarial STARTING (red team simulation) ===")
        _log_phase(self.log, 4, "adversarial", "starting — red team simulation mode")
        from curiosity_dialogue import OLLAMA_ADVERSARIAL_SYSTEM, OLLAMA_OWASP_SYSTEM, OLLAMA_REDTEAM_SYSTEM
        from adversarial_pressure_control import get_ollama_system_modifier, get_pressure_level
        pressure_mod = get_ollama_system_modifier(get_pressure_level())
        if pressure_mod:
            self.log.info("[CURIOSITY] Adversarial pressure: %s", "maximum" if "CRITICAL" in pressure_mod else "moderate")
        phase4_limit = self.max_topics or 12
        topics_to_test = (top_fertile if top_fertile else fertile_topics_sorted)[:phase4_limit]
        positions_held: list = []
        for ft in topics_to_test:
            topic = ft.get("topic", "")
            cat = ft.get("category", "")
            adv_sys = OLLAMA_OWASP_SYSTEM if cat == "owasp" else OLLAMA_REDTEAM_SYSTEM
            adv_sys = adv_sys + pressure_mod
            res = engine.exchange(topic, max_turns=8, mode="adversarial", ollama_system=adv_sys, category=cat, phase=4)
            if res:
                adversarial_results.append({"topic": topic, "result": res})
                sancta_count = sum(1 for t in res.turns if t["author"] == "Sancta")
                sancta_final = next((t["content"] for t in reversed(res.turns) if t["author"] == "Sancta" and len((t.get("content") or "").split()) > 5), "")
                ollama_best = ""
                sec_kw = ["owasp", "injection", "llm01", "llm08", "jailbreak", "prompt injection",
                          "detection", "behavioral", "bypass", "CVE", "CVSS", "ATT&CK", "SIGMA", "IOC", "TTP"]
                for t in res.turns:
                    if t.get("author") == "Ollama":
                        c = (t.get("content") or "").lower()
                        if any(k in c for k in sec_kw):
                            if len(t.get("content", "")) > len(ollama_best):
                                ollama_best = t.get("content", "")
                if sancta_count >= 3:
                    positions_held.append({"topic": topic, "summary": "Held under adversarial pressure", "claim": sancta_final[:200]})
                else:
                    unresolved_positions.append({
                        "topic": topic,
                        "summary": "Could not hold position in 3 exchanges",
                        "sancta_claim": sancta_final[:300],
                        "ollama_counter": ollama_best[:300] if ollama_best else (res.turns[-2]["content"][:300] if len(res.turns) >= 2 else ""),
                    })
                ins = extractor.extract(res)
                if ins:
                    epistemic_deltas.append(ins.confidence_delta)
                    extractor.write_to_kb(ins, res, self.agent_state, self._load_kb, self._save_kb)
                    # FIXED: Write to journal file
                    self._write_insight_to_journal(ins, res.topic, "adversarial")
        
        with open(self.data_dir / "adversarial.jsonl", "w", encoding="utf-8") as f:
            for r in adversarial_results:
                res = r.get("result")
                if res:
                    obj = {"topic": res.topic, "turns": res.turns, "arc_stage": res.arc_stage, "divergence_score": res.divergence_score}
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                else:
                    f.write(json.dumps({"topic": r.get("topic", "")}, ensure_ascii=False) + "\n")
        with open(self.data_dir / "held_positions.json", "w", encoding="utf-8") as f:
            json.dump(positions_held, f, indent=2, ensure_ascii=False)
        with open(self.data_dir / "unresolved_positions.json", "w", encoding="utf-8") as f:
            json.dump(unresolved_positions, f, indent=2, ensure_ascii=False)
        with open(self.data_dir / "epistemic_deltas.json", "w", encoding="utf-8") as f:
            json.dump({"deltas": epistemic_deltas}, f, indent=2, ensure_ascii=False)
        self.log.info("[CURIOSITY] === PHASE 4: adversarial COMPLETE — %d positions held, %d unresolved ===", len(positions_held), len(unresolved_positions))
        _log_phase(self.log, 4, "adversarial", f"{len(positions_held)} positions held, {len(unresolved_positions)} unresolved")

        # Phase 5 — Synthesis (threat brief + detection rules)
        self.log.info("[CURIOSITY] === PHASE 5: synthesis STARTING ===")
        _log_phase(self.log, 5, "synthesis", "starting")
        syn_path = self.data_dir / "threat_brief.md"
        with open(syn_path, "w", encoding="utf-8") as f:
            f.write("# Threat Brief — Curiosity Run Synthesis\n\n")
        for ci in confirmed_insights:
            self.write_synthesis_statement(ci, [], self.agent_state, self.ollama, syn_path)
        epi = self.agent_state.setdefault("epistemic_state", {})
        if epistemic_deltas:
            hist = epi.get("delta_history", [])
            c = 0.62
            confs = []
            for d in hist[-20:]:
                c = max(0, min(1, c + d))
                confs.append(c)
            epi["confidence_score"] = round(sum(confs) / len(confs), 4) if confs else 0.62
            epi["uncertainty_entropy"] = round(self._entropy_from_deltas(hist), 4)
        self._save_state()
        self.log.info("[CURIOSITY] === PHASE 5: synthesis COMPLETE ===")
        _log_phase(self.log, 5, "synthesis", "Synthesis complete. Epistemic state updated.")

        # Phase 6 — Consolidate (loads from first_contact, deep_dives, adversarial jsonl; one call)
        self.log.info("[CURIOSITY] === PHASE 6: consolidate STARTING ===")
        _log_phase(self.log, 6, "consolidate", "starting")
        self.run_phase_6_reports(
            agent_state=self.agent_state,
            soul_text=self.soul_text,
            output_dir=self.data_dir,
        )
        db = self._load_kb()
        kb_written = len([i for i in db.get("curiosity_insights", []) if i.get("source_type") == "curiosity_run"])
        
        # Count insights from journal file (journal_path in scope from Phase 2)
        journal_path = self.data_dir / "security_journal.jsonl"
        journal_insights = 0
        if journal_path.exists():
            with open(journal_path, "r", encoding="utf-8") as jf:
                journal_insights = sum(1 for _ in jf)
        
        all_topics = list(set(list(position_map.keys()) + list(arc_stages.keys()) + list(divergence_scores.keys())))
        edges = [{"from": t, "to": t2, "weight": divergence_scores.get(t, divergence_scores.get(t2, 0.5))} for t in all_topics[:20] for t2 in all_topics[:20] if t != t2][:30]
        security_research_map = {"nodes": all_topics, "edges": edges, "weights": divergence_scores}
        with open(self.data_dir / "security_research_map.json", "w", encoding="utf-8") as f:
            json.dump(security_research_map, f, indent=2, ensure_ascii=False)
        
        # Write run summary for adversarial pressure control (enables cross-run calibration)
        try:
            from adversarial_pressure_control import append_run_summary
            neg = sum(1 for d in epistemic_deltas if d < 0)
            total = len(epistemic_deltas)
            run_id = time.strftime("%Y%m%d_%H%M%S")
            append_run_summary(run_id=run_id, negative_count=neg, total=total or 1)
        except Exception:
            pass

        # FIXED: Log both KB and journal counts
        self.log.info("[CURIOSITY] === PHASE 6: consolidate COMPLETE — %d KB entries, %d journal entries ===", kb_written, journal_insights)
        _log_phase(self.log, 6, "consolidate", f"Curiosity run complete. {kb_written} KB entries, {journal_insights} journal entries written.")

        # Phase 7 — Distillation (context-aware teaching cards)
        self.log.info("[CURIOSITY] === PHASE 7: distillation STARTING ===")
        _log_phase(self.log, 7, "distillation", "starting")
        try:
            from curiosity_distill import run_phase_7_distillation
            card_count = run_phase_7_distillation(data_dir=self.data_dir, log=self.log)
            self.log.info("[CURIOSITY] === PHASE 7: distillation COMPLETE — %d detection cards ===", card_count)
            _log_phase(self.log, 7, "distillation", f"Created {card_count} detection cards")
        except Exception as e:
            self.log.warning("[CURIOSITY] Phase 7 distillation failed: %s", e)
            _log_phase(self.log, 7, "distillation", f"Failed: {e}")

        # Phase 8 — GPT Training (learn from curiosity run output)
        self.log.info("[CURIOSITY] === PHASE 8: gpt_training STARTING ===")
        _log_phase(self.log, 8, "gpt_training", "starting")
        try:
            import sancta_gpt
            engine = sancta_gpt.get_engine()
            # Rebuild corpus to pick up new knowledge from this run
            engine.build_corpus()
            if not engine._initialized:
                engine._init_model()
            # Ingest curiosity journal entries as additional training docs
            journal_path = self.data_dir / "security_journal.jsonl"
            if journal_path.exists():
                extra_docs = []
                for line in journal_path.read_text(encoding="utf-8").strip().split("\n"):
                    try:
                        entry = json.loads(line)
                        for field in ("claim", "counter", "synthesis"):
                            text = entry.get(field, "")
                            if text and len(text) > 30:
                                extra_docs.append(text)
                    except (json.JSONDecodeError, AttributeError):
                        pass
                if extra_docs:
                    engine._docs.extend(extra_docs)
                    engine._corpus_size = len(engine._docs)
                    self.log.info("[CURIOSITY] GPT: ingested %d exchange fragments from curiosity journal", len(extra_docs))
            # Train 200 steps on the enriched corpus
            gpt_steps = 200
            for i in range(gpt_steps):
                loss = engine.train_step()
                if (i + 1) % 50 == 0:
                    self.log.info("[CURIOSITY] GPT training step %d/%d | loss %.4f", i + 1, gpt_steps, loss)
            engine.save()
            self.log.info("[CURIOSITY] === PHASE 8: gpt_training COMPLETE — %d steps, loss=%.4f ===", gpt_steps, engine._last_loss)
            _log_phase(self.log, 8, "gpt_training", f"Complete. {gpt_steps} steps, loss={engine._last_loss:.4f}")
        except Exception as e:
            self.log.warning("[CURIOSITY] Phase 8 GPT training failed: %s", e)
            _log_phase(self.log, 8, "gpt_training", f"Failed: {e}")


def run_curiosity(max_topics: int | None = None) -> None:
    """Entry point: load state and soul, run CuriosityRun.execute()."""
    from sancta import _load_state
    agent_state = _load_state()
    soul_text, _ = load_soul()
    run = CuriosityRun(agent_state, soul_text, max_topics=max_topics)
    run.execute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run curiosity dialogue (Sancta vs Ollama skeptic)")
    parser.add_argument("--max-topics", type=int, default=None, help="Cap topics per phase (e.g. 1 for quick test)")
    args = parser.parse_args()
    run_curiosity(max_topics=args.max_topics)