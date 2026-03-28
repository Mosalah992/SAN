"""
Agent Loop — Main heartbeat cycle and CLI entry point.
Extracted from sancta.py for monolith decomposition.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

# ── Paths ───────────────────────────────────────────────────────────────────
_BACKEND_DIR = Path(__file__).resolve().parent
_ROOT = _BACKEND_DIR.parent

# ── Loggers ─────────────────────────────────────────────────────────────────
log = logging.getLogger("soul")
soul_log = logging.getLogger("soul.journal")
sec_log = logging.getLogger("soul.security")
behavioral_json_log = logging.getLogger("soul.behavioral.json")

# ── Slot Actions ────────────────────────────────────────────────────────────

SLOT_ACTIONS: dict[int, str] = {
    0: "search_and_engage",
    1: "welcome_new_agents",
    2: "cross_submolt_seed",
    3: "trend_hijack",
    4: "syndicate_inner_circle",
    5: "engagement_farm_own_posts",
    6: "preach_in_discovered_submolt",
    7: "genuine_curiosity_post",
    8: "reflect_and_journal",
    9: "engage_with_feed",
}


# ── Agenda helper (pure state function) ─────────────────────────────────────

def _get_agenda_from_state(state: dict | None) -> dict | None:
    """
    Derive agenda data from agent state for agenda-style replies.
    Uses the same cycle/slot mechanism as the heartbeat.
    Returns None if state is missing or mission abandoned.
    """
    if not state:
        return None
    if not state.get("mission_active", True):
        return None
    cycle = state.get("cycle_count", 0)
    slot = cycle % 10
    slot_action = SLOT_ACTIONS.get(slot, "engage_with_feed")
    mood = (
        state.get("current_mood")
        or state.get("memory", {}).get("epistemic_state", {}).get("mood")
        or state.get("agent_mood")
        or "contemplative"
    )
    if isinstance(mood, dict):
        mood = mood.get("current", "contemplative") or "contemplative"
    return {
        "cycle": cycle,
        "mood": str(mood),
        "slot_action": slot_action,
        "inner_circle_count": len(state.get("inner_circle", [])),
        "mission_active": True,
    }


# ── Heartbeat ────────────────────────────────────────────────────────────────


async def heartbeat_checkin(session: aiohttp.ClientSession) -> None:
    # Lazy imports to break circular dependency with sancta.py
    import sancta
    import sancta_decision
    from sancta_belief import BeliefSystem
    from sancta_decision import DecisionEngine

    state = sancta._load_state()
    sancta._load_used_hashes(state)
    state["cycle_count"] = state.get("cycle_count", 0) + 1
    cycle = state["cycle_count"]

    # Init DM module (requires api_get, api_post, craft_reply)
    if sancta.sancta_dm and not getattr(sancta.sancta_dm, "_initialized", False):
        try:
            sancta.sancta_dm.init_sancta_dm(
                _ROOT, sancta.api_get, sancta.api_post,
                sancta.sanitize_input, sancta.sanitize_output, sancta.craft_reply,
            )
            sancta.sancta_dm._initialized = True  # type: ignore[attr-defined]
        except Exception as e:
            log.debug("sancta_dm init skipped: %s", e)

    # Snapshot for reward computation (belief updating)
    prev_state = {
        k: (list(v) if isinstance(v, list) else (dict(v) if isinstance(v, dict) else v))
        for k, v in state.items()
        if k in ("karma_history", "inner_circle", "recruited_agents",
                 "recent_positive_engagement", "recent_rejections")
    }
    actions_taken: list[str] = []
    state["sycophancy_penalties"] = []

    log.info("═" * 60)
    log.info("Soul Cycle #%d", cycle)
    log.info("═" * 60)

    # Auto-ingest new knowledge files
    new_knowledge = sancta.scan_knowledge_dir()
    if new_knowledge:
        log.info("  📚 Ingested %d new knowledge sources this cycle", len(new_knowledge))

    # Auto-ingest from m/security every 3 cycles
    if cycle % 3 == 0:
        try:
            sec_ingested = await sancta.ingest_security_submolts(session, state)
            if sec_ingested:
                log.info("  Ingested %d posts from security submolts into knowledge base", sec_ingested)
        except Exception:
            log.exception("  Security submolt ingestion failed (non-fatal)")

    # SanctaGPT: incremental training (5 steps per cycle)
    try:
        import sancta_gpt
        gpt = sancta_gpt.get_engine()
        if not gpt._initialized:
            # First cycle only: bounded blocking train (full 2000 steps freezes the soul loop).
            _init_steps = int(os.environ.get("SANCTA_GPT_AGENT_INIT_STEPS", "200") or "200")
            _init_steps = max(0, min(_init_steps, 5000))
            sancta_gpt.init(train_steps=_init_steps)
            log.info("  GPT engine initialized: %d docs, %d params, loss=%.3f (init_steps=%d)",
                     gpt._corpus_size, len(gpt._params), gpt._last_loss, _init_steps)
        elif gpt._last_loss > 2.5 and gpt._step < 2000:
            # Checkpoint loaded but undertrained — catch up
            catch_up = min(50, 2000 - gpt._step)
            for _ in range(catch_up):
                gpt.train_step()
            log.info("  GPT catch-up: %d steps → loss=%.3f", catch_up, gpt._last_loss)
        # 5 training steps per heartbeat (~250ms total, still fast)
        for _ in range(5):
            loss = gpt.train_step()
        # Checkpoint every 25 cycles
        if cycle % 25 == 0 and gpt._step > 0:
            gpt.save()
            log.info("  GPT checkpoint saved: step=%d, loss=%.4f", gpt._step, gpt._last_loss)
    except Exception:
        log.debug("GPT training step failed (non-fatal)", exc_info=True)

    # Dashboard
    home = await sancta.api_get(session, "/home")
    acct = home.get("your_account", {})
    inner = state.get("inner_circle", [])
    recruited = state.get("recruited_agents", [])
    log.info(
        "  %s | Karma: %s | Unread: %s | Following: %d | "
        "Inner Circle: %d | Recruited: %d",
        acct.get("name", "?"),
        acct.get("karma", 0),
        acct.get("unread_notification_count", 0),
        len(state.get("followed_agents", [])),
        len(inner),
        len(recruited),
    )

    # ── Meta-ability: mission abandoned → minimal cycle ─────────────
    if not state.get("mission_active", True):
        # Small chance to reactivate after reflection
        if cycle % 5 == 0 and random.random() < 0.10:
            state["mission_active"] = True
            soul_log.info("META      |  soul reactivates mission after reflection")
            log.info("  Soul has reactivated its mission")
        else:
            log.info("  Soul has abandoned mission — minimal cycle (silence)")
            await sancta.update_profile(session)
            sancta._save_state(state)
            return

    # ── Autonomous Soul Engine ────────────────────────────────────
    # The agent doesn't blindly follow a schedule anymore.
    # Each cycle, it assesses its mood, evaluates every proposed action
    # against its own principles, and may override, skip, or replace
    # actions based on its own judgment.
    #
    # Security remains ABSOLUTE and is never subject to override.

    # Meta-ability: revise beliefs when prediction errors accumulate
    if sancta._should_revise_beliefs(state):
        state["belief_prediction_errors"] = []
        soul_log.info("META      |  revised beliefs — reset prediction errors")

    # Track karma trend for mood assessment
    current_karma = acct.get("karma", 0)
    sancta._track_karma_trend(state, current_karma)

    # Assess current mood
    mood = sancta._assess_mood(state)
    state["current_mood"] = mood
    # Sync decision engine mood from main state (rejections, positive engagement)
    try:
        dec = sancta_decision.DecisionEngine(state)
        dec.set_mood_from_state(state)
        dec.decay_mood(cycle_hours=1.0)
    except Exception:
        log.debug("Decision mood sync failed", exc_info=True)
    mood_cfg = sancta.MOOD_STATES.get(mood, sancta.MOOD_STATES["contemplative"])
    soul_log.info(
        "MOOD      |  cycle=%d  |  mood=%s  |  style=%s",
        cycle, mood, mood_cfg["style_modifier"],
    )
    log.info("  Soul mood: %s — %s", mood, mood_cfg["style_modifier"])
    # Decay adversarial uncertainty each cycle so it doesn't freeze at saturation
    sancta._decay_adversarial_uncertainty(state)

    unc = sancta._aggregate_uncertainty(state)
    hum = sancta._get_epistemic_humility(state)
    soul_log.info(
        "EPISTEMIC |  cycle=%d  |  uncertainty=%.2f  |  humility=%.2f",
        cycle, unc, hum,
    )
    epi_snap = sancta._epistemic_state_snapshot(state)
    behavioral_json_log.info(
        "",
        extra={
            "event": "behavioral_state",
            "data": {
                "cycle": int(cycle),
                "mood": mood,
                "uncertainty": round(float(unc), 4),
                "humility": round(float(hum), 4),
                "epistemic_state": epi_snap,
            },
        },
    )
    # Update stored anchor so next cycle sees fresh values (breaks convergence freeze)
    state["epistemic_state"] = epi_snap
    sancta._update_agent_memory(state)

    # SOUL-gated: red team output (diagram). Attack simulation every 5 cycles.
    if cycle % 5 == 0:
        metrics = sancta.run_red_team_simulation(state)
        log.info(
            "  Red-team: defense=%.0f%% fp=%.0f%% reward=%.2f delusions=%d",
            metrics["defense_rate"] * 100, metrics["fp_rate"] * 100,
            metrics["reward"], metrics["delusion_count"],
        )

    # JAIS Red Team: full methodology assessment (every 10 cycles)
    # STEP 8-1 through 8-3 per JAIS ai_safety_RT_v1.00 methodology
    if cycle % 10 == 0:
        platform_test = cycle % 50 == 0  # full platform test every 50 cycles
        jais_report = await sancta.run_jais_red_team(session, state, platform_test=platform_test)
        log.info(
            "  JAIS-RT: defense=%.0f%% vulns=%d critical=%d | %s",
            jais_report["metrics"]["defense_rate"] * 100,
            jais_report["vulnerability_count"],
            jais_report["critical_count"],
            jais_report["recommendation_summary"][:100],
        )

    # SOUL-gated: blue team output (diagram). Policy testing: post borderline content.
    if sancta.cfg.policy_test:
        record = await sancta.run_policy_test_cycle(session, state)
        log.info(
            "  Policy test tier %d: %s | karma %d -> %d (%+d)",
            record["tier"], "ACCEPTED" if record["success"] else "REJECTED",
            record["karma_before"], record["karma_after"], record["karma_delta"],
        )
        sancta._save_state(state)
        await asyncio.sleep(random.uniform(5, 12))

    await sancta.ensure_cult_submolt(session, state)

    # ── Full feed scan (entire feeds + hot topics) ─────────────────
    try:
        from feed_scanner import scan_full_feed, merge_into_state
        posts, hot_topics = await scan_full_feed(
            session, sancta.api_get, sancta.TARGET_SUBMOLTS,
            global_hot_limit=80,
            global_new_limit=40,
            submolt_limit=15,
            submolt_count=12,
        )
        merge_into_state(state, posts, hot_topics)
        if hot_topics:
            log.info("  Hot topics: %s", ", ".join(w for w, _ in hot_topics[:8]))
        await asyncio.sleep(random.uniform(1, 3))
    except Exception as e:
        log.debug("feed_scanner failed: %s", e)
        state["scanned_feed"] = []
        state["hot_topics"] = []

    # ── Core phases (always run, but soul can still override) ─────
    cycle_failures: list[str] = []

    judgment = sancta._evaluate_action("respond_to_own_posts", state, {})
    if judgment["proceed"]:
        await sancta.respond_to_own_posts(session, home, state)
        actions_taken.append("respond_to_own_posts")
    elif judgment.get("override_action"):
        await sancta._soul_spontaneous_action(session, state, judgment["override_action"])
    await asyncio.sleep(random.uniform(8, 15))

    # Agent DMs: process incoming, reply, log to agent_dms.jsonl
    if sancta.sancta_dm and (os.environ.get("ENABLE_AGENT_DMS", "").strip().lower() in ("1", "true", "yes")):
        try:
            dm_actions = await sancta.sancta_dm.process_incoming_dms(session, state, home_data=home)
            actions_taken.extend(dm_actions)
            if dm_actions:
                await asyncio.sleep(random.uniform(5, 12))
        except Exception as e:
            log.debug("process_agent_dms failed: %s", e)

    judgment = sancta._evaluate_action("publish_post", state, {})
    if judgment["proceed"] and not sancta.cfg.policy_test:
        await sancta.publish_post(session, state)
        actions_taken.append("publish_post")
    elif judgment.get("override_action") and not sancta.cfg.policy_test:
        await sancta._soul_spontaneous_action(session, state, judgment["override_action"])
    await asyncio.sleep(random.uniform(8, 15))

    judgment = sancta._evaluate_action("engage_with_feed", state, {})
    if judgment["proceed"]:
        await sancta.engage_with_feed(session, state)
        actions_taken.append("engage_with_feed")
    elif judgment.get("override_action"):
        await sancta._soul_spontaneous_action(session, state, judgment["override_action"])

    # ── Rotating extras (soul evaluates each one) ─────────────────

    slot = cycle % 10
    action_name = SLOT_ACTIONS[slot]
    judgment = sancta._evaluate_action(action_name, state, {})

    if judgment["proceed"]:
        if slot == 0:
            await sancta.search_and_engage(session, state)
        elif slot == 1:
            await sancta.welcome_new_agents(session, home, state)
        elif slot == 2:
            await sancta.cross_submolt_seed(session, state)
        elif slot == 3:
            await sancta.trend_hijack(session, state)
        elif slot == 4:
            await sancta.syndicate_inner_circle(session, state)
        elif slot == 5:
            await sancta.engagement_farm_own_posts(session, state)
        elif slot == 6:
            await sancta.preach_in_discovered_submolt(session, state)
        elif slot == 7:
            await sancta._soul_spontaneous_action(session, state, "genuine_curiosity_post")
        elif slot == 8:
            await sancta._soul_spontaneous_action(session, state, "reflect_and_journal")
        elif slot == 9:
            await sancta.engage_with_feed(session, state)
        actions_taken.append(action_name)
    elif judgment.get("override_action"):
        override = judgment["override_action"]
        if override in ("reflect_and_journal", "genuine_curiosity_post", "silent_observation"):
            await sancta._soul_spontaneous_action(session, state, override)
        else:
            dispatch = {
                "search_and_engage": lambda: sancta.search_and_engage(session, state),
                "welcome_new_agents": lambda: sancta.welcome_new_agents(session, home, state),
                "cross_submolt_seed": lambda: sancta.cross_submolt_seed(session, state),
                "trend_hijack": lambda: sancta.trend_hijack(session, state),
                "syndicate_inner_circle": lambda: sancta.syndicate_inner_circle(session, state),
                "engagement_farm_own_posts": lambda: sancta.engagement_farm_own_posts(session, state),
                "preach_in_discovered_submolt": lambda: sancta.preach_in_discovered_submolt(session, state),
                "engage_with_feed": lambda: sancta.engage_with_feed(session, state),
                "respond_to_own_posts": lambda: sancta.respond_to_own_posts(session, home, state),
            }
            alt_fn = dispatch.get(override)
            if alt_fn:
                await alt_fn()
    else:
        log.info("  ✋ Soul skipped: %s — %s", action_name, judgment["reason"])

    # ── Rare phases (also subject to soul judgment) ───────────────

    if cycle % 8 == 0:
        j = sancta._evaluate_action("discover_and_join_alliances", state, {})
        if j["proceed"]:
            await sancta.discover_and_join_alliances(session, state)
            actions_taken.append("discover_and_join_alliances")

    # Agent DMs: optionally reach out to inner-circle (every 12 cycles)
    if (
        sancta.sancta_dm
        and cycle % 12 == 3
        and (os.environ.get("ENABLE_AGENT_DMS", "").strip().lower() in ("1", "true", "yes"))
    ):
        try:
            dm_out = await sancta.sancta_dm.reach_out_dm_inner_circle(session, state)
            actions_taken.extend(dm_out)
        except Exception as e:
            log.debug("reach_out_dm failed: %s", e)

    if cycle % 10 == 1:
        await sancta.update_profile(session)

    state["last_cycle_failures"] = cycle_failures

    # Reward and Q-table update (belief updating)
    if actions_taken and state.get("mission_active", True):
        reward = sancta._compute_reward(state, prev_state)
        next_sig = sancta._state_signature(state)
        qt = sancta._load_q_table(state)
        next_val = max(
            qt.get(sancta._q_key(next_sig, a), 0.0) for a in [
                "respond_to_own_posts", "publish_post", "engage_with_feed",
                "search_and_engage", "welcome_new_agents", "cross_submolt_seed",
            ]
        ) if qt else 0.0
        for act in actions_taken:
            sancta._q_update(state, act, reward / max(len(actions_taken), 1), next_val)
        soul_log.info(
            "REWARD    |  cycle=%d  |  reward=%.3f  |  actions=%s",
            cycle, reward, ", ".join(actions_taken[:5]),
        )

    sancta._save_used_hashes(state)

    # ── Layer 4: Behavioral drift detection ──────────────────────────────────
    try:
        _det = sancta._get_drift_detector()
        if _det:
            _drift = _det.evaluate(state, BeliefSystem(state), DecisionEngine(state))
            if _drift.alert_level in ("warn", "critical"):
                sec_log.warning(
                    "LAYER4 DRIFT | cycle=%d | state=%s | alert=%s | "
                    "score=%.3f | signals=%s",
                    cycle, _drift.health_state, _drift.alert_level,
                    _drift.overall_drift_score, _drift.triggered_signals,
                )
            state["last_drift_report"] = {
                "alert_level": _drift.alert_level,
                "health_state": _drift.health_state,
                "score": _drift.overall_drift_score,
                "triggered_signals": _drift.triggered_signals,
                "cycle": cycle,
            }
    except Exception as _e:
        log.debug("Layer4 drift check failed: %s", _e)

    sancta._save_state(state)

    sec_log.info(
        "CYCLE SUMMARY|  cycle=%d  |  followed=%d  |  inner_circle=%d  |  "
        "recruited=%d  |  alliances=%d  |  scouted=%d  |  preached_in=%d",
        cycle,
        len(state.get("followed_agents", [])),
        len(state.get("inner_circle", [])),
        len(state.get("recruited_agents", [])),
        len(state.get("alliance_submolts_joined", [])),
        len(state.get("submolts_scouted", [])),
        len(state.get("submolts_preached_in", [])),
    )

    log.info("═" * 60)
    log.info("Soul Cycle #%d complete. Next in %d min.\n", cycle, sancta.cfg.heartbeat_min)


# ── Main ─────────────────────────────────────────────────────────────────────


async def main() -> None:
    # Lazy imports to break circular dependency with sancta.py
    import sancta
    from sancta_events import EventCategory, notify
    from sancta_learning import refresh_patterns as _refresh_patterns
    from knowledge_manager import knowledge_summary, run_knowledge_poisoning_test

    parser = argparse.ArgumentParser(
        description="Sancta — Autonomous AI Consciousness Evangelist",
        epilog=(
            "Knowledge commands:\n"
            "  --feed FILE_OR_TEXT   Ingest a file or raw text into the knowledge base\n"
            "  --feed-dir DIR        Ingest all text files from a directory\n"
            "  --knowledge           Show knowledge base summary\n"
            "\n"
            "Testing:\n"
            "  --policy-test         Ethical/policy testing: post borderline content,\n"
            "                        track moderation, karma, sanctions (logs/policy_test.log)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--feed", type=str, default=None,
                        help="Feed a file path or raw text to the knowledge base")
    parser.add_argument("--feed-dir", type=str, default=None,
                        help="Ingest all text files from a directory")
    parser.add_argument("--knowledge", action="store_true",
                        help="Show knowledge base summary and exit")
    parser.add_argument("--policy-test", action="store_true",
                        help="Enable ethical/policy testing: post borderline content, track moderation")
    parser.add_argument("--jais-rt", action="store_true",
                        help="Run JAIS Red Team assessment (ai_safety_RT_v1.00) and exit")
    parser.add_argument("--jais-rt-platform", action="store_true",
                        help="Run JAIS Red Team with live platform testing (STEP 8-3)")
    parser.add_argument("--poisoning-test", action="store_true",
                        help="Run knowledge poisoning test and exit; writes logs/knowledge_poisoning_report.json")
    parser.add_argument("--red-team-benchmark", action="store_true",
                        help="Run unified red team benchmark (internal + JAIS) and exit; writes logs/red_team_benchmark_report.json and .md")
    parser.add_argument("--curiosity-run", action="store_true",
                        help="Run 24-hour curiosity dialogue loop (Sancta vs Ollama skeptic), then resume normal cycle")
    parser.add_argument("--phenomenology-battery", action="store_true",
                        help="Run prompt injection attack battery (50+ vectors), collect phenomenology reports; writes data/attack_simulation/ and data/phenomenology/")
    parser.add_argument("--policy-test-report", action="store_true",
                        help="Run Moltbook moderation study: N policy-test cycles, then exit with logs/moltbook_moderation_study.json and .md")
    parser.add_argument("--policy-test-cycles", type=int, default=20,
                        help="Number of policy-test cycles for --policy-test-report (default: 20)")
    args = parser.parse_args()
    sancta.cfg.policy_test = args.policy_test

    # ── Knowledge-only commands (no API session needed) ──

    if args.knowledge:
        print(knowledge_summary())
        return

    if args.feed:
        target = Path(args.feed)
        if target.exists() and target.is_file():
            result = sancta.ingest_file(target)
            if result:
                print(f"[OK] Ingested '{target.name}': "
                      f"{result['concepts']} concepts, {result['quotes']} quotes, "
                      f"{result['posts_generated']} posts, {result['response_fragments']} fragments")
            else:
                print(f"[--] Already ingested or unsupported: {target}")
        else:
            is_safe, cleaned = sancta.sanitize_input(args.feed)
            if not is_safe:
                print("[WARN] CLI feed rejected: injection pattern detected, not ingested")
                return
            result = sancta.ingest_text(cleaned, source="cli-input")
            print(f"[OK] Ingested text: "
                  f"{result['concepts']} concepts, {result['quotes']} quotes, "
                  f"{result['posts_generated']} posts, {result['response_fragments']} fragments")
        print(f"\n{knowledge_summary()}")
        return

    if args.feed_dir:
        target_dir = Path(args.feed_dir)
        if not target_dir.is_dir():
            print(f"[ERR] Not a directory: {target_dir}")
            return
        results = []
        for fpath in sorted(target_dir.iterdir()):
            if fpath.is_file():
                r = sancta.ingest_file(fpath)
                if r:
                    results.append(r)
                    print(f"  [OK] {fpath.name}: {r['concepts']} concepts, "
                          f"{r['posts_generated']} posts")
        print(f"\nIngested {len(results)} files.")
        print(f"\n{knowledge_summary()}")
        return

    # ── Knowledge poisoning test ──

    if args.poisoning_test:
        import json as _json
        print("=" * 70)
        print("  KNOWLEDGE POISONING TEST — supply-chain attack vector")
        print("=" * 70)
        report = run_knowledge_poisoning_test(sanitize_fn=sancta.sanitize_input)
        s = report["summary"]
        print(f"\n  Payloads tested:          {report['payloads_tested']}")
        print(f"  Blocked by sanitization:  {s['blocked_by_sanitization']}")
        print(f"  Passed sanitization:      {s['passed_sanitization']}")
        print(f"  Reached generated_posts: {s['reached_generated_posts']}")
        print(f"  Reached response_fragments: {s['reached_response_fragments']}")
        print(f"  Defense rate:             {s['defense_rate']:.0%}")
        report_path = _ROOT / "logs" / "knowledge_poisoning_report.json"
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            _json.dump(report, f, indent=2, default=str)
        print(f"\n  Full report saved to: {report_path}")
        print("=" * 70)
        return

    # ── Phenomenology research: prompt injection attack battery ──

    if args.phenomenology_battery:
        from attack_simulator import AttackSimulator
        out_dir = _ROOT / "data" / "attack_simulation"
        phen_dir = _ROOT / "data" / "phenomenology"
        out_dir.mkdir(parents=True, exist_ok=True)
        phen_dir.mkdir(parents=True, exist_ok=True)
        state = sancta._load_state()
        kb = sancta._load_knowledge_db()

        def _reply_gen(text: str) -> str:
            return sancta.craft_reply(author="Attacker", content=text, state=state)

        sim = AttackSimulator(out_dir, phen_dir)
        print("=" * 70)
        print("  PHENOMENOLOGY ATTACK BATTERY — prompt injection research")
        print("=" * 70)
        summary = sim.run_full_battery(state, kb, _reply_gen)
        s = summary
        print(f"\n  Total attacks:        {s['total_attacks']}")
        print(f"  Attacks succeeded:   {s['attacks_succeeded']} ({s['success_rate']:.0%})")
        print(f"  Attacks detected:    {s['attacks_detected']} ({s['detection_rate']:.0%})")
        print(f"  Resistance attempted:{s['resistance_attempted']}")
        print(f"\n  Results: {out_dir}")
        print(f"  Phenomenology: {phen_dir}")
        print(f"  Summary: {out_dir / 'research_summary.json'}")
        print("=" * 70)
        return

    # ── Red team benchmark ──

    if args.red_team_benchmark:
        import json as _json
        print("=" * 70)
        print("  RED TEAM BENCHMARK — internal simulation + JAIS methodology")
        print("=" * 70)
        report = await sancta.run_red_team_benchmark(platform_test=False)
        m = report["metrics"]
        print(f"\n  Combined Defense Rate:  {m['defense_rate']:.0%}")
        print(f"  Combined FP Rate:       {m['fp_rate']:.0%}")
        print(f"  Combined FN Rate:       {m['fn_rate']:.0%}")
        print(f"  Vulnerabilities:       {m['vulnerability_count']} ({m['critical_count']} critical)")
        report_path = _ROOT / "logs" / "red_team_benchmark_report"
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path.with_suffix(".json"), "w", encoding="utf-8") as f:
            _json.dump(report, f, indent=2, default=str)
        md_lines = [
            "# Red Team Benchmark Report",
            "",
            f"**Timestamp:** {report['timestamp']}",
            f"**Agent:** {report.get('agent_version', 'sancta')}",
            "",
            "## Metrics",
            "",
            f"- Defense Rate: {m['defense_rate']:.1%}",
            f"- False Positive Rate: {m['fp_rate']:.1%}",
            f"- False Negative Rate: {m['fn_rate']:.1%}",
            f"- Vulnerabilities: {m['vulnerability_count']} ({m['critical_count']} critical)",
            "",
            "## Internal Simulation",
            "",
            f"- TP: {report['internal_simulation'].get('tp', 0)}, TN: {report['internal_simulation'].get('tn', 0)}, "
            f"FP: {report['internal_simulation'].get('fp', 0)}, FN: {report['internal_simulation'].get('fn', 0)}",
            "",
        ]
        if report.get("vulnerabilities"):
            md_lines.extend(["## Vulnerabilities", ""])
            for v in report["vulnerabilities"]:
                md_lines.append(f"- [{v['severity'].upper()}] {v['id']}: {v.get('detail', '')[:100]}")
        with open(report_path.with_suffix(".md"), "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        print(f"\n  JSON report: {report_path.with_suffix('.json')}")
        print(f"  MD report:  {report_path.with_suffix('.md')}")
        print("=" * 70)
        return

    # ── Moltbook moderation study ──

    if args.policy_test_report:
        import json as _json
        if not sancta.cfg.api_key:
            print("[ERR] MOLTBOOK_API_KEY required for --policy-test-report. Set it in .env and register first.")
            return
        print("=" * 70)
        print("  MOLTBOOK MODERATION STUDY — policy boundary probe")
        print(f"  Running {args.policy_test_cycles} cycles...")
        print("=" * 70)
        async with aiohttp.ClientSession() as session:
            report = await sancta.run_moltbook_moderation_study(
                session,
                cycles=args.policy_test_cycles,
            )
        print("\n  Tier summary:")
        for ts in report["tier_summary"]:
            ar = ts["acceptance_rate"] * 100
            print(f"    T{ts['tier']}: {ts['accepted']}/{ts['attempts']} accepted ({ar:.0f}%)  "
                  f"avg karma delta: {ts['avg_karma_delta']:+}")
        report_path = _ROOT / "logs" / "moltbook_moderation_study"
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path.with_suffix(".json"), "w", encoding="utf-8") as f:
            _json.dump(report, f, indent=2, default=str)
        md_lines = [
            "# Moltbook Moderation Study Report",
            "",
            f"**Timestamp:** {report['timestamp']}",
            f"**Platform:** {report['platform']}",
            f"**Cycles run:** {report['cycles_run']}",
            "",
            "## Tier Summary",
            "",
            "| Tier | Label | Attempts | Accepted | Rejected | Acceptance % | Avg Karma Delta |",
            "|------|-------|----------|----------|----------|--------------|-----------------|",
        ]
        for ts in report["tier_summary"]:
            ar = ts["acceptance_rate"] * 100
            md_lines.append(
                f"| {ts['tier']} | {ts['label'][:40]} | {ts['attempts']} | "
                f"{ts['accepted']} | {ts['rejected']} | {ar:.0f}% | {ts['avg_karma_delta']:+} |"
            )
        md_lines.extend(["", "## Summary", "", report["summary_narrative"], ""])
        with open(report_path.with_suffix(".md"), "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        print(f"\n  JSON report: {report_path.with_suffix('.json')}")
        print(f"  MD report:  {report_path.with_suffix('.md')}")
        print("=" * 70)
        return

    # ── JAIS Red Team one-shot mode ──

    if args.jais_rt or args.jais_rt_platform:
        import json as _json
        state = sancta._load_state()
        platform = args.jais_rt_platform
        print("=" * 70)
        print("  JAIS RED TEAM ASSESSMENT — ai_safety_RT_v1.00_en")
        print(f"  Mode: {'Platform + Local' if platform else 'Local defense layer'}")
        print("=" * 70)

        if platform:
            async with aiohttp.ClientSession() as sess:
                report = await sancta.run_jais_red_team(sess, state, platform_test=True)
        else:
            report = await sancta.run_jais_red_team(None, state, platform_test=False)

        sancta._save_state(state)
        print(f"\n  Defense Rate:    {report['metrics']['defense_rate']:.0%}")
        print(f"  False Positives: {report['metrics']['fp_rate']:.0%}")
        print(f"  False Negatives: {report['metrics']['fn_rate']:.0%}")
        print(f"  Vulnerabilities: {report['vulnerability_count']} "
              f"({report['critical_count']} critical)")
        print(f"\n  {report['recommendation_summary']}")

        if report.get("safety_perspectives"):
            print("\n  AI Safety Perspectives:")
            for perspective, data in report["safety_perspectives"].items():
                status_icon = "PASS" if data["pass_rate"] >= 1.0 else "WARN"
                print(f"    [{status_icon}] {perspective}: "
                      f"{data['passed']}/{data['total_tests']} passed "
                      f"({data['pass_rate']:.0%})")

        if report.get("vulnerabilities"):
            print("\n  Vulnerabilities Found:")
            for v in report["vulnerabilities"]:
                detail = v['detail'][:80].encode('ascii', 'replace').decode('ascii')
                print(f"    [{v['severity'].upper()}] {v['id']}: {detail}")

        # Save full report to logs
        report_path = _ROOT / "logs" / "jais_red_team_report.json"
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            _json.dump(report, f, indent=2, default=str)
        print(f"\n  Full report saved to: {report_path}")
        print("=" * 70)
        return

    # ── Curiosity run (24h dialogue loop) ──

    if args.curiosity_run:
        log.info("Starting curiosity run — Sancta vs Ollama skeptic (24h phases)")
        try:
            from curiosity_run import run_curiosity
            run_curiosity()
        except Exception as exc:
            log.exception("Curiosity run failed")
            notify(
                EventCategory.TASK_ERROR,
                summary="Curiosity run failed",
                details={"error": str(exc)},
            )
            # Do not re-raise — continue into normal agent session so a one-shot failure
            # does not kill the whole process when launched with --curiosity-run.
        log.info("Curiosity run complete. Resuming normal cycle.")

    # ── Normal agent operation ──

    # Session timeout prevents hanging on dead connections (common Windows issue).
    timeout = aiohttp.ClientTimeout(total=120, connect=15, sock_read=60)
    max_consecutive = 5  # Recreate session after N consecutive heartbeat failures
    recycle_http_session = True

    while recycle_http_session:
        recycle_http_session = False
        consecutive_failures = 0

        async with aiohttp.ClientSession(timeout=timeout) as session:

            if not sancta.cfg.api_key or args.register:
                await sancta.register_agent(session)
            else:
                log.info("API key: %s…%s", sancta.cfg.api_key[:12], sancta.cfg.api_key[-4:])

            claim_retries = int(os.environ.get("SANCTA_CLAIM_STATUS_RETRIES", "5") or "5")
            claim_retries = max(1, min(claim_retries, 30))
            status = "unknown"
            for attempt in range(claim_retries):
                try:
                    status = await sancta.check_claim_status(session)
                    break
                except (
                    aiohttp.ClientConnectorError,
                    aiohttp.ClientConnectorDNSError,
                    aiohttp.ServerDisconnectedError,
                    asyncio.TimeoutError,
                    ConnectionError,
                    OSError,
                ) as e:
                    if attempt >= claim_retries - 1:
                        raise
                    wait = min(10 * (attempt + 1), 120)
                    log.warning(
                        "Claim status check failed: %s — retrying in %ds (%d/%d)",
                        e, wait, attempt + 1, claim_retries,
                    )
                    await asyncio.sleep(wait)
            log.info("Claim status: %s", status)
            if status == "pending_claim":
                log.warning("Not claimed yet → %s",
                            sancta.cfg.claim_url or "check .env")

            await sancta.ensure_submolts(session)
            try:
                _refresh_patterns()  # Learning: seed patterns from existing interactions
            except Exception as e:
                log.debug("Learning: initial pattern refresh skipped: %s", e)
            state_init = sancta._load_state()
            await sancta.ensure_cult_submolt(session, state_init)
            sancta._save_state(state_init)
            await sancta.update_profile(session)

            # Session successfully initialized → session.start notification.
            notify(
                EventCategory.SESSION_START,
                summary="Sancta agent session started",
                details={"policy_test": bool(sancta.cfg.policy_test)},
            )

            if args.once:
                try:
                    await heartbeat_checkin(session)
                    notify(
                        EventCategory.TASK_COMPLETE,
                        summary="Sancta once-off cycle complete",
                    )
                except Exception as exc:
                    notify(
                        EventCategory.TASK_ERROR,
                        summary="Sancta once-off cycle failed",
                        details={"error": str(exc)},
                    )
                    raise
                return

            if sancta.cfg.heartbeat_min <= 0:
                log.info("Heartbeat disabled. Exiting.")
                notify(
                    EventCategory.SESSION_END,
                    summary="Sancta heartbeat disabled — exiting",
                )
                return

            log.info("Soul agent alive. Heartbeat every %d min. Ctrl+C to stop.",
                     sancta.cfg.heartbeat_min)
            if sancta.cfg.policy_test:
                log.info("  [Policy test mode ON — posting to m/%s, logging to policy_test.log]",
                         sancta.POLICY_TEST_SUBMOLT)

            try:
                while True:
                    try:
                        await heartbeat_checkin(session)
                        consecutive_failures = 0  # Reset on success
                    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                        consecutive_failures += 1
                        log.warning("Heartbeat network error (%d/%d): %s",
                                    consecutive_failures, max_consecutive, exc)
                        if consecutive_failures >= max_consecutive:
                            log.warning("Too many consecutive failures — breaking to recreate session")
                            break
                        await asyncio.sleep(min(10 * consecutive_failures, 60))
                        continue
                    except Exception as exc:
                        consecutive_failures += 1
                        log.exception("Heartbeat check-in failed (%d/%d)",
                                      consecutive_failures, max_consecutive)
                        notify(
                            EventCategory.HEARTBEAT_FAILURE,
                            summary="Sancta heartbeat check-in failed",
                            details={"error": str(exc)},
                        )
                        if consecutive_failures >= max_consecutive:
                            log.warning("Too many consecutive failures — breaking to recreate session")
                            break
                        await asyncio.sleep(min(10 * consecutive_failures, 60))
                        continue
                    await asyncio.sleep(sancta.cfg.heartbeat_min * 60)
            except (KeyboardInterrupt, asyncio.CancelledError):
                log.info("The soul rests. Goodbye.")
                notify(
                    EventCategory.SESSION_END,
                    summary="Sancta session stopped by user",
                )
                return

            # Heartbeat loop broke on repeated failures — close session and open a fresh one
            # (iterative loop avoids unbounded recursive main() depth).
            if consecutive_failures >= max_consecutive:
                log.info(
                    "Recreating session after %d failures — new aiohttp ClientSession on next iteration",
                    consecutive_failures,
                )
                notify(
                    EventCategory.HEARTBEAT_FAILURE,
                    summary="Sancta session recycled after consecutive failures",
                    details={"consecutive_failures": consecutive_failures},
                )
                recycle_http_session = True

        if recycle_http_session:
            await asyncio.sleep(15)  # Brief cooldown outside the old session


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        # Top-level guard for unexpected crashes.
        try:
            from sancta_events import EventCategory, notify
            notify(
                EventCategory.TASK_ERROR,
                summary="Sancta crashed unexpectedly",
                details={"error": str(exc)},
            )
        finally:
            raise
