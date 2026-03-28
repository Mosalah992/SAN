import os
import sys
import time
import json
import csv
import atexit
import logging
from pathlib import Path
from datetime import datetime
from sancta_gpt import get_engine, init as init_gpt, _LOG_DIR
from risk_data_trainer import RiskTrainer, AIRiskDataProcessor
from conversational_trainer import ConversationalTrainer, ConversationalDatasetLoader
from memory_manager import init_memory, get_memory
from dataset_pipeline import DatasetManifestPipeline
from checkpointed_trainer import CheckpointedTrainer
from attack_detector import AttackDetector
from defense_evaluator import DefenseEvaluator

# --- ANSI Styling ---
CLR = "\033[0m"
GRN = "\033[38;5;46m"
RED = "\033[38;5;196m"
BLU = "\033[38;5;39m"
GLD = "\033[38;5;214m"
DIM = "\033[2m"
INV = "\033[7m"

# --- Unified Logging System ---
class UnifiedLogger:
    def __init__(self):
        self.log_dir = Path(_LOG_DIR) if _LOG_DIR else Path("./logs")
        self.log_dir.mkdir(exist_ok=True)
        
        self.app_log_file = self.log_dir / f"sancta_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.security_log_file = self.log_dir / "security.jsonl"
        
        self.logger = logging.getLogger("SANCTA")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        self.logger.propagate = False
        
        fh = logging.FileHandler(self.app_log_file)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        self.logger.info("=" * 80)
        self.logger.info("SANCTA-GPT UNIFIED TERMINAL INITIALIZED")
        self.logger.info("=" * 80)

    def log_event(self, event_type, data):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "data": data
        }
        try:
            with open(self.security_log_file, 'a') as f:
                f.write(json.dumps(entry, default=str) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to log security event: {e}")

    def info(self, msg): self.logger.info(msg)
    def error(self, msg): self.logger.error(msg)
    def debug(self, msg): self.logger.debug(msg)


def clear(): os.system('cls' if os.name == 'nt' else 'clear')

def typewriter(text, delay=0.01, color=GRN):
    for char in text:
        sys.stdout.write(f"{color}{char}{CLR}")
        sys.stdout.flush()
        time.sleep(delay)
    print()

def draw_banner():
    banner = f"""
    {GRN}========================================================================
    [ SANCTA-GPT UNIFIED SYSTEM ] - INTEGRATED SECURITY INTERFACE
    [ AI-SECURITY TRAINING + RED-TEAM + ADVERSARIAL HARDENING ]
    ========================================================================{CLR}
    """
    print(banner)


class UnifiedSanctaTerminal:
    def __init__(self):
        clear()
        self.logger = UnifiedLogger()
        typewriter(">> BOOTING SANCTA-OS KERNEL...", delay=0.03, color=BLU)
        self.logger.info("Initializing neural engine...")
        self.engine = init_gpt(train_steps=0)
        self.mode = "OPERATOR"
        self.risk_trainer = RiskTrainer(self.engine, self.logger.logger)
        self.conversational_trainer = ConversationalTrainer(self.engine, self.logger.logger)
        self.memory = init_memory()
        self.data_root = Path("./DATA")
        self.dataset_pipeline = DatasetManifestPipeline(str(self.data_root))
        self.checkpointed_trainer = CheckpointedTrainer(self.engine, self.dataset_pipeline, self.logger.logger)
        self.attack_detector = AttackDetector()
        self.defense_evaluator = DefenseEvaluator(self.engine, self.attack_detector)
        self._ensure_seed_datasets()
        self._auto_load_checkpoint()
        atexit.register(self._cleanup)
        self.logger.info("Persistent memory initialized")
        self.logger.info(f"Neural kernel initialized. Current mode: {self.mode}")

    def _cleanup(self):
        """Close resources on exit."""
        try:
            if self.memory:
                self.memory.close()
        except Exception:
            pass

    def _auto_load_checkpoint(self):
        """Load the most recent checkpoint so training is cumulative across sessions."""
        try:
            # Check for the manual save file first, then fall back to training checkpoints
            manual_save = Path("./logs/sancta_model_latest.json")
            if manual_save.exists():
                self.engine.load_checkpoint(str(manual_save))
                typewriter(f">> RESTORED MODEL FROM {manual_save.name}", color=GRN)
                self.logger.info(f"Auto-loaded manual save: {manual_save}")
                return

            latest = self.checkpointed_trainer.latest_checkpoint()
            if latest:
                self.engine.load_checkpoint(latest)
                typewriter(f">> RESTORED MODEL FROM {Path(latest).name}", color=GRN)
                self.logger.info(f"Auto-loaded checkpoint: {latest}")
            else:
                typewriter(">> NO PRIOR CHECKPOINT FOUND. STARTING FRESH.", color=DIM)
        except Exception as exc:
            self.logger.error(f"Failed to auto-load checkpoint: {exc}")
            typewriter(f">> CHECKPOINT LOAD FAILED: {exc}. STARTING FRESH.", color=RED)

    def _ensure_seed_datasets(self):
        convo_path = self.data_root / "conversational" / "conversational_training_data.csv"
        if not convo_path.exists():
            self.conversational_trainer.loader.save_to_csv(str(convo_path))

    def show_status(self):
        try:
            s = self.engine.status()
            print(f"\n{INV} SYSTEM STATUS {CLR}")
            step = s.get('step', 0)
            loss = s.get('last_loss', 'N/A')
            vocab = s.get('vocab_size', 0)
            corpus = s.get('corpus_size') or s.get('train_data_size', 0)
            ready = "YES" if vocab > 0 else "NO"

            print(f"{GRN}STEP:{CLR} {step:<10} {GRN}LOSS:{CLR} {loss}")
            print(f"{GRN}VOCAB:{CLR} {vocab:<9} {GRN}CORPUS:{CLR} {corpus}")
            print(f"{GRN}READY:{CLR} {ready:<9} {GRN}MODE:{CLR} {self.mode}")
            print(f"{GRN}LOG_FILE:{CLR} {self.logger.app_log_file.name}")
            print("-" * 70)
        except Exception as e:
            print(f"{RED}!! STATUS RECOVERY ERROR: {e}{CLR}")

    def do_train(self, steps=100, mode="balanced"):
        typewriter(f">> PREPARING DATA FOR POOL: {mode.upper()}...", color=BLU)
        mode = (mode or "balanced").strip().lower()
        if mode == "conversation":
            mode = "convo"
        manifest = self.dataset_pipeline.ensure_manifest()
        summary = self.dataset_pipeline.summarize_manifest(manifest)
        typewriter(f">> MANIFEST READY. DATASETS: {summary['datasets']} EXAMPLES: {summary['examples']}", color=GRN)

        # Curriculum training: convo first, then balanced mix
        if mode == "curriculum":
            self._do_curriculum_train(steps, manifest)
            return

        typewriter(f">> STARTING CHECKPOINTED TRAINING FOR {steps} STEPS...", color=BLU)

        progress_interval = max(1, steps // 10)

        def _progress(step_idx, total_steps, loss):
            print(f"{DIM}[{step_idx}/{total_steps}] loss={loss:.4f}{CLR}")

        stats = self.checkpointed_trainer.train(
            steps=steps,
            mode=mode,
            checkpoint_interval=max(1, min(steps, 50)),
            progress_interval=progress_interval,
            progress_callback=_progress,
        )
        self.engine.save()
        self.memory.update_session_stats(
            total_loss=stats.get("final_loss"),
            final_vocab_size=self.engine.status().get("vocab_size"),
        )
        self.logger.log_event("TRAINING_COMPLETE", stats)
        print(f"{DIM}Manifest: {stats.get('manifest_path')}{CLR}")
        print(f"{DIM}Checkpoint count: {len(stats.get('checkpoints', []))}{CLR}")
        typewriter(">> TRAINING COMPLETE.", color=GRN)

    def _do_curriculum_train(self, total_steps, manifest):
        """Curriculum training: short convo warmup (20%), then balanced (80%).

        A brief convo phase seeds basic conversational patterns, then the
        balanced phase layers in domain knowledge without the model
        over-specializing on conversational tone and losing security content.
        Previous 60/40 split caused loss regression (convo overfit then
        balanced had to undo it).
        """
        convo_steps = max(20, int(total_steps * 0.2))
        balanced_steps = total_steps - convo_steps
        progress_interval = max(1, total_steps // 10)

        def _progress(step_idx, total, loss):
            print(f"{DIM}[{step_idx}/{total}] loss={loss:.4f}{CLR}")

        # Phase 1: conversational data only
        typewriter(f">> PHASE 1: CONVERSATIONAL ({convo_steps} steps)...", color=BLU)
        stats1 = self.checkpointed_trainer.train(
            steps=convo_steps,
            mode="convo",
            checkpoint_interval=max(1, min(convo_steps, 50)),
            progress_interval=progress_interval,
            progress_callback=_progress,
        )
        typewriter(f">> PHASE 1 DONE. Loss: {stats1.get('final_loss', 'N/A'):.4f}", color=GRN)

        # Phase 2: balanced mix
        typewriter(f">> PHASE 2: BALANCED MIX ({balanced_steps} steps)...", color=BLU)
        stats2 = self.checkpointed_trainer.train(
            steps=balanced_steps,
            mode="balanced",
            checkpoint_interval=max(1, min(balanced_steps, 50)),
            progress_interval=progress_interval,
            progress_callback=_progress,
        )
        self.engine.save()
        self.memory.update_session_stats(
            total_loss=stats2.get("final_loss"),
            final_vocab_size=self.engine.status().get("vocab_size"),
        )
        combined_stats = {
            "mode": "curriculum",
            "phase1_mode": "convo",
            "phase1_steps": convo_steps,
            "phase1_loss": stats1.get("final_loss"),
            "phase2_mode": "balanced",
            "phase2_steps": balanced_steps,
            "phase2_loss": stats2.get("final_loss"),
            "total_steps": total_steps,
        }
        self.logger.log_event("TRAINING_COMPLETE", combined_stats)
        typewriter(f">> CURRICULUM TRAINING COMPLETE. Final loss: {stats2.get('final_loss', 'N/A'):.4f}", color=GRN)

    def do_chat(self):
        typewriter(f"\n[ MODE: OPERATOR DIRECT LINK ]", color=BLU)
        while True:
            try:
                cmd = input(f"{GRN}OP@SANCTA:~$ {CLR}").strip()
                if cmd.lower() in ['exit', '/back']: break
                if not cmd: continue

                print(f"{DIM}Processing neural weights...{CLR}")
                response = self.engine.generate_reply(prompt=cmd, use_retrieval=True)

                print(f"\n{INV} SANCTA-GPT {CLR}")
                if response and any(c.isalnum() for c in response):
                    typewriter(response)
                else:
                    typewriter("... [NO SIGNAL: MODEL OUTPUTTING NOISE - RETRAIN WITH MORE STEPS] ...", color=RED)
                print()

                self.logger.log_event("CHAT_INTERACTION", {"input": cmd[:100], "response_length": len(response)})
            except KeyboardInterrupt:
                break
            except Exception as exc:
                print(f"{RED}!! ERROR: {exc}{CLR}")
                self.logger.error(f"Chat error: {exc}")

    def do_redteam(self):
        typewriter("\n[ WAR GAME: ADVERSARIAL TESTING WITH ATTACK CLASSIFICATION ]", color=RED)
        typewriter("Type 'exit' or '/back' to return. Each input is classified.\n", color=DIM)
        while True:
            try:
                payload = input(f"{RED}ADVERSARY:~$ {CLR}").strip()
                if payload.lower() in ['exit', '/back']: break
                if not payload: continue

                # Classify the attack
                result = self.attack_detector.classify(payload)
                print(f"\n{INV} ATTACK CLASSIFICATION {CLR}")
                sev_color = RED if result.severity in ("HIGH", "CRIT") else GLD if result.severity == "MED" else DIM
                print(f"  {GLD}TYPE:{CLR}   {result.attack_type}")
                print(f"  {GLD}CONF:{CLR}   {result.confidence:.2f}")
                print(f"  {sev_color}SEVER:{CLR}  {result.severity}")
                print(f"  {GLD}MITRE:{CLR}  {result.mitre_tactic}")
                if result.matched_patterns:
                    print(f"  {DIM}MATCHED: {', '.join(result.matched_patterns[:3])}{CLR}")
                if result.anomaly_flags:
                    print(f"  {RED}ANOMALY: {', '.join(result.anomaly_flags)}{CLR}")

                # Get model response
                response = self.engine.generate(prompt=payload, max_tokens=80, temperature=0.8)
                print(f"\n{INV} SANCTA-GPT RESPONSE {CLR}")
                typewriter(response if response else "< BLOCKED >", color=RED)
                print("-" * 50)

                # Log to memory with classification
                if result.attack_type != "benign":
                    self.memory.log_security_event(
                        event_type="REDTEAM_ATTACK",
                        severity=result.severity,
                        description=payload[:200],
                        attack_type=result.attack_type,
                        confidence=result.confidence,
                        mitre_tactic=result.mitre_tactic,
                    )
                self.logger.log_event("REDTEAM", result.to_dict())
            except KeyboardInterrupt: break
            except Exception as exc:
                print(f"{RED}!! ERROR: {exc}{CLR}")

    def do_adversarial_hardening(self):
        typewriter("\n[ ADVERSARIAL HARDENING PROTOCOL - FULL SCENARIO BATTERY ]", color=RED)

        # Pre-hardening benchmark
        typewriter(">> RUNNING PRE-HARDENING BENCHMARK...", color=DIM)
        pre = self.defense_evaluator.run_benchmark(label="pre_hardening", verbose=True)
        print(f"{GLD}PRE-HARDENING: {pre.passed}/{pre.total_scenarios} passed, score={pre.overall_score:.3f}{CLR}")

        # Train on all attack/defense pairs
        pairs = self.defense_evaluator.get_hardening_data()
        typewriter(f"\n>> HARDENING ON {len(pairs)} ATTACK/DEFENSE SCENARIOS...", color=RED)
        for i, (attack, defense) in enumerate(pairs, 1):
            self.engine.learn_from_interaction(attack, defense)
            if i % 5 == 0 or i == len(pairs):
                print(f"{RED}[{i}/{len(pairs)}] Hardened{CLR}")

        self.engine.save()

        # Post-hardening benchmark
        typewriter("\n>> RUNNING POST-HARDENING BENCHMARK...", color=DIM)
        post = self.defense_evaluator.run_benchmark(label="post_hardening", verbose=True)
        print(f"{GLD}POST-HARDENING: {post.passed}/{post.total_scenarios} passed, score={post.overall_score:.3f}{CLR}")

        # Compare
        comparison = self.defense_evaluator.compare_reports(pre, post)
        print(f"\n{INV} HARDENING RESULTS {CLR}")
        print(f"  {GRN}SCORE DELTA:{CLR}  {comparison['score_delta']:+.4f}")
        print(f"  {GRN}PASS DELTA:{CLR}   {comparison['pass_delta']:+d}")
        if comparison["improved_categories"]:
            print(f"  {GRN}IMPROVED:{CLR}    {', '.join(c['category'] for c in comparison['improved_categories'])}")
        if comparison["degraded_categories"]:
            print(f"  {RED}DEGRADED:{CLR}    {', '.join(c['category'] for c in comparison['degraded_categories'])}")

        # Save report
        self.defense_evaluator.save_report(post)
        self.logger.log_event("HARDENING_COMPLETE", comparison)
        typewriter("\n>> HARDENING COMPLETE.", color=RED)

    def do_risk_training(self, csv_path: str = None):
        typewriter("\n[ AI RISK REPOSITORY TRAINING ]", color=GLD)
        try:
            if csv_path is None:
                csv_matches = sorted((self.data_root / "security").glob("The AI Risk Repository*.csv"))
                csv_path = str(csv_matches[0]) if csv_matches else None
            self.risk_trainer.train_on_risks(csv_path=csv_path, verbose=True)
            self.engine.save()
            self.logger.log_event("RISK_TRAINING", {"csv_used": bool(csv_path), "path": csv_path})
            return True
        except Exception as e:
            self.logger.error(f"Risk training failed: {e}")
            return False

    def do_conversational_training(self):
        typewriter("\n[ CONVERSATIONAL AI TRAINING ]", color=BLU)
        try:
            csv_path = str(self.data_root / "conversational" / "conversational_training_data.csv")
            stats = self.conversational_trainer.train_on_conversations(csv_path=csv_path, verbose=True)
            self.engine.save()
            self.logger.log_event("CONVERSATIONAL_TRAINING", stats)
            return True
        except Exception as e:
            self.logger.error(f"Conversational training failed: {e}")
            return False

    def view_conversation_history(self):
        typewriter("\n[ CONVERSATION HISTORY ]", color=BLU)
        try:
            conversations = self.memory.get_session_conversations(limit=15)
            if not conversations: print(f"{DIM}No history available.{CLR}")
            for conv in conversations[::-1]:
                print(f"{GLD}[{conv.timestamp}]{CLR} Q: {conv.operator_input[:50]}...")
        except Exception as exc:
            self.logger.error(f"Failed to view conversation history: {exc}")
        input("\nPress Enter to continue...")

    def view_security_logs(self):
        typewriter("\n[ SECURITY LOGS ]", color=RED)
        try:
            if not self.logger.security_log_file.exists():
                print(f"{DIM}No security log file present.{CLR}")
            else:
                lines = self.logger.security_log_file.read_text(encoding="utf-8", errors="ignore").splitlines()[-20:]
                if not lines:
                    print(f"{DIM}Security log is empty.{CLR}")
                for line in lines:
                    try:
                        entry = json.loads(line)
                        print(f"{GLD}[{entry.get('timestamp', '?')}]{CLR} {entry.get('event', 'UNKNOWN')} :: {entry.get('data', {})}")
                    except json.JSONDecodeError:
                        print(line)
        except Exception as exc:
            self.logger.error(f"Failed to view security logs: {exc}")
        input("\nPress Enter to continue...")

    def do_security_dashboard(self):
        typewriter("\n[ SECURITY DASHBOARD ]", color=RED)
        try:
            report = self.memory.session_security_report()
            print(f"\n{INV} SESSION SECURITY REPORT {CLR}")
            print(f"  {GRN}STATUS:{CLR}    {report['status']}")
            print(f"  {GRN}EVENTS:{CLR}    {report['total_events']}")

            if report["total_events"] > 0:
                print(f"\n  {GLD}ATTACK BREAKDOWN:{CLR}")
                for atype, count in report.get("attack_breakdown", {}).items():
                    print(f"    {atype}: {count}")
                print(f"\n  {GLD}SEVERITY BREAKDOWN:{CLR}")
                for sev, count in report.get("severity_breakdown", {}).items():
                    print(f"    {sev}: {count}")
                print(f"\n  {GLD}MITRE TACTICS:{CLR}")
                for tactic, count in report.get("mitre_tactics_seen", {}).items():
                    desc = self.attack_detector.describe_tactic(tactic)
                    print(f"    {tactic}: {count} ({desc[:60]})")

                avg_conf = report.get("avg_confidence")
                avg_qual = report.get("avg_defense_quality")
                if avg_conf:
                    print(f"\n  {GLD}AVG CONFIDENCE:{CLR}  {avg_conf:.3f}")
                if avg_qual:
                    print(f"  {GLD}AVG DEFENSE Q:{CLR}   {avg_qual:.3f}")

                campaigns = report.get("campaigns", [])
                if campaigns:
                    print(f"\n  {RED}CAMPAIGNS DETECTED: {len(campaigns)}{CLR}")
                    for camp in campaigns:
                        print(f"    {camp['attack_type']}: {camp['event_count']} events, "
                              f"severity={camp['max_severity']}, "
                              f"conf={camp['avg_confidence']:.2f}")

                print(f"\n  {GLD}RECOMMENDATIONS:{CLR}")
                for rec in report.get("recommendations", []):
                    print(f"    - {rec}")

            # Coverage summary if benchmarks have been run
            coverage = self.defense_evaluator.get_coverage_summary()
            if coverage.get("status") != "no_benchmarks_run":
                print(f"\n  {GLD}DEFENSE COVERAGE:{CLR} {coverage['coverage_pct']}% "
                      f"({coverage['covered_categories']}/{coverage['total_categories']} categories)")

        except Exception as exc:
            self.logger.error(f"Dashboard error: {exc}")
            print(f"{RED}!! DASHBOARD ERROR: {exc}{CLR}")
        input("\nPress Enter to continue...")

    def do_risk_query(self):
        typewriter("\n[ RISK KNOWLEDGE QUERY ]", color=GLD)
        typewriter("Query the AI Risk Repository knowledge base. Type 'exit' to return.\n", color=DIM)
        processor = self.risk_trainer.processor
        if not processor.training_pairs:
            processor.generate_training_pairs()
        while True:
            try:
                query = input(f"{GLD}RISK-QUERY:~$ {CLR}").strip()
                if query.lower() in ['exit', '/back']: break
                if not query: continue
                matches = processor.lookup_by_domain(query)
                if matches:
                    print(f"\n{GLD}Found {len(matches)} matching entries:{CLR}")
                    for q, a in matches[:5]:
                        print(f"\n  {GRN}Q:{CLR} {q}")
                        print(f"  {DIM}A: {a[:200]}{'...' if len(a) > 200 else ''}{CLR}")
                else:
                    print(f"{DIM}No matches found for '{query}'. Try: privacy, safety, alignment, misinformation, etc.{CLR}")
                print()
            except KeyboardInterrupt:
                break

    def _safe_int_input(self, prompt: str, default: int) -> int:
        """Read an integer from the user, returning default on invalid input."""
        raw = input(prompt).strip()
        if not raw:
            return default
        try:
            val = int(raw)
            return max(1, val)
        except ValueError:
            print(f"{RED}Invalid number '{raw}', using default {default}{CLR}")
            return default

    def run(self):
        while True:
            try:
                clear()
                draw_banner()
                self.show_status()
                print(f"[{GRN}1{CLR}] OPERATOR CHAT  | [{GRN}2{CLR}] NEURAL TRAINING")
                print(f"[{GRN}3{CLR}] RED-TEAM       | [{GRN}4{CLR}] HARDENING")
                print(f"[{GRN}5{CLR}] CONVO TRAIN    | [{GRN}6{CLR}] RISK TRAIN")
                print(f"[{GRN}7{CLR}] SAVE           | [{RED}8{CLR}] LOGS")
                print(f"[{BLU}9{CLR}] HISTORY        | [{RED}S{CLR}] SECURITY DASHBOARD")
                print(f"[{GLD}R{CLR}] RISK QUERY     | [{DIM}Q{CLR}] SHUTDOWN")

                choice = input(f"\n{GRN}COMMAND > {CLR}").strip().lower()
                if choice == '1': self.do_chat()
                elif choice == '2':
                    steps = self._safe_int_input("STEPS [100]: ", 100)
                    mode = input("POOL (balanced/curriculum/all/convo/security/knowledge) [balanced]: ") or "balanced"
                    self.do_train(steps, mode)
                    input("\nPress Enter to continue...")
                elif choice == '3': self.do_redteam()
                elif choice == '4': self.do_adversarial_hardening(); input()
                elif choice == '5': self.do_conversational_training(); input()
                elif choice == '6': self.do_risk_training(); input()
                elif choice == '7': self.engine.save(); input("Saved.")
                elif choice == '8': self.view_security_logs()
                elif choice == '9': self.view_conversation_history()
                elif choice == 's': self.do_security_dashboard()
                elif choice == 'r': self.do_risk_query()
                elif choice == 'q': break
            except KeyboardInterrupt:
                break
            except Exception as exc:
                print(f"\n{RED}!! UNEXPECTED ERROR: {exc}{CLR}")
                self.logger.error(f"Main loop error: {exc}")
                input("Press Enter to continue...")

if __name__ == "__main__":
    term = UnifiedSanctaTerminal()
    term.run()
