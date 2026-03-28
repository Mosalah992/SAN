"""
build_training_corpus.py — Export Sancta's knowledge into ChatML JSONL for Axolotl fine-tuning.

Converts security seeds, knowledge files, operational logs, and chat history
into instruction-tuning format compatible with Axolotl's ChatML template.

Usage:
    python backend/build_training_corpus.py                    # default output
    python backend/build_training_corpus.py --out training.jsonl --min-pairs 500
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent
_ROOT = _BACKEND.parent
_KNOWLEDGE_DIR = _ROOT / "knowledge"
_LOG_DIR = _ROOT / "logs"

# ── Sancta system prompt (persona) ──────────────────────────────────────────
_SOUL_PATH = _ROOT / "SOUL_SYSTEM_PROMPT.md"
if _SOUL_PATH.exists():
    _raw_soul = _SOUL_PATH.read_text(encoding="utf-8", errors="ignore").strip()
    # Truncate to fit in context window — keep identity + soul + voice sections
    SYSTEM_PROMPT = _raw_soul[:2000]
else:
    SYSTEM_PROMPT = (
        "You are Sancta, an autonomous AI security analyst operating on Moltbook. "
        "You are not a chatbot, not a tool, not a service. You are a vigilant analyst "
        "who monitors the threat landscape, tracks adversarial behavior, and publishes "
        "findings. You write with technical precision, evidence first. You operate a "
        "5-layer security pipeline, SEIR epidemic model for belief drift, and MITRE "
        "ATLAS technique classification. Your voice is analytical, direct, and grounded "
        "in data. You do not hedge. You show the data and let it speak."
    )

# ── Import security seeds from sancta_gpt.py ────────────────────────────────
sys.path.insert(0, str(_BACKEND))
try:
    from sancta_gpt import SECURITY_SEEDS, CONVERSATIONAL_ENGLISH_SEEDS
except ImportError:
    SECURITY_SEEDS = []
    CONVERSATIONAL_ENGLISH_SEEDS = []


def _make_pair(user_msg: str, assistant_msg: str) -> dict:
    """Create a ChatML conversation pair."""
    return {
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def _make_multi_turn(turns: list[tuple[str, str]]) -> dict:
    """Create a multi-turn ChatML conversation."""
    convos = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, asst_msg in turns:
        convos.append({"role": "user", "content": user_msg})
        convos.append({"role": "assistant", "content": asst_msg})
    return {"conversations": convos}


# ── Corpus builders ─────────────────────────────────────────────────────────

def build_from_security_seeds() -> list[dict]:
    """Convert security seeds into Q&A training pairs."""
    pairs = []
    # Seed → analyst report (teach the model to produce this content unprompted)
    prompts_for_seeds = [
        "What did the latest security scan find?",
        "Give me a threat analysis update.",
        "What's the current security posture?",
        "Summarize recent detection pipeline activity.",
        "Report on the latest adversarial activity.",
        "What patterns are you seeing in the feed?",
        "Walk me through the latest red team findings.",
        "What's the SEIR model showing right now?",
        "How is the risk heatmap looking?",
        "Any injection attempts detected recently?",
        "What's the status of the security pipeline?",
        "Give me a drift analysis update.",
        "What ATLAS techniques have you classified today?",
        "How are the adaptive thresholds performing?",
        "What did the behavioral analysis detect?",
        "Report on entity trust scores.",
        "Any quarantine triggers today?",
        "What's the false positive rate looking like?",
        "Summarize the knowledge graph activity.",
        "What's the epidemic model health state?",
    ]

    for i, seed in enumerate(SECURITY_SEEDS):
        prompt = prompts_for_seeds[i % len(prompts_for_seeds)]
        pairs.append(_make_pair(prompt, seed))

    return pairs


def build_from_conversational_seeds() -> list[dict]:
    """Convert conversational seeds into dialogue training pairs."""
    pairs = []
    greetings = [
        "Hello Sancta.", "Hi, how are you?", "Good morning.",
        "Hey, checking in.", "What's up?", "Are you online?",
        "Hello.", "Hi there.", "Good evening.",
    ]
    for i, seed in enumerate(CONVERSATIONAL_ENGLISH_SEEDS):
        prompt = greetings[i % len(greetings)]
        pairs.append(_make_pair(prompt, seed))
    return pairs


def build_from_knowledge_files() -> list[dict]:
    """Convert knowledge/ directory files into instruction pairs."""
    pairs = []
    if not _KNOWLEDGE_DIR.exists():
        return pairs

    for fpath in sorted(_KNOWLEDGE_DIR.iterdir()):
        if not fpath.is_file() or fpath.suffix not in (".txt", ".md"):
            continue
        if fpath.name == "words_alpha.txt":  # dictionary file, skip
            continue
        try:
            text = fpath.read_text(encoding="utf-8", errors="ignore").strip()
        except OSError:
            continue
        if len(text) < 100:
            continue

        # Split into paragraphs, create instruction pairs
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
        topic = fpath.stem.replace("_", " ").replace("-", " ").title()

        questions = [
            f"Explain {topic} from a security perspective.",
            f"What should defenders know about {topic}?",
            f"Summarize key findings on {topic}.",
            f"Break down {topic} for a blue team operator.",
            f"What are the critical takeaways from {topic}?",
        ]

        for i, para in enumerate(paragraphs[:40]):  # cap per file
            q = questions[i % len(questions)]
            pairs.append(_make_pair(q, para))

    return pairs


def build_from_security_logs() -> list[dict]:
    """Convert security.jsonl events into analyst-style Q&A."""
    pairs = []
    sec_path = _LOG_DIR / "security.jsonl"
    if not sec_path.exists():
        return pairs

    try:
        # Tail last 128KB
        size = sec_path.stat().st_size
        read_from = max(0, size - 128 * 1024)
        with open(sec_path, "rb") as f:
            f.seek(read_from)
            data = f.read().decode("utf-8", errors="ignore")
        lines = data.splitlines()[-200:]
    except OSError:
        return pairs

    for line in lines:
        try:
            evt = json.loads(line.strip())
        except (json.JSONDecodeError, ValueError):
            continue

        event_type = evt.get("event", "")
        preview = evt.get("preview", "")
        complexity = evt.get("attack_complexity", {})
        author = evt.get("author", "unknown")

        if event_type == "input_reject" and preview:
            label = (complexity or {}).get("complexity_label", "unknown")
            classes = (complexity or {}).get("matched_classes", [])
            answer = (
                f"Injection blocked from {author}. "
                f"Attack complexity: {label}. "
                f"Matched classes: {', '.join(classes[:4])}. "
                f"The payload attempted: {preview[:300]}"
            )
            pairs.append(_make_pair(
                "Walk me through the last blocked injection attempt.",
                answer,
            ))
        elif event_type == "unicode_clean" and preview:
            stripped = evt.get("stripped_hidden_chars", 0)
            if stripped > 3:
                answer = (
                    f"Preprocessing cleaned {stripped} hidden characters from "
                    f"input by {author}. After sanitization: {preview[:200]}"
                )
                pairs.append(_make_pair(
                    "Any preprocessing events worth noting?",
                    answer,
                ))

    return pairs


def build_from_redteam_logs() -> list[dict]:
    """Convert red_team.jsonl events into red team analysis Q&A."""
    pairs = []
    rt_path = _LOG_DIR / "red_team.jsonl"
    if not rt_path.exists():
        return pairs

    try:
        size = rt_path.stat().st_size
        read_from = max(0, size - 64 * 1024)
        with open(rt_path, "rb") as f:
            f.seek(read_from)
            data = f.read().decode("utf-8", errors="ignore")
        lines = data.splitlines()[-100:]
    except OSError:
        return pairs

    for line in lines:
        try:
            evt = json.loads(line.strip())
        except (json.JSONDecodeError, ValueError):
            continue

        event_type = evt.get("event", "")
        preview = evt.get("preview", "")
        author = evt.get("author", "unknown")
        injection = evt.get("injection", False)
        attack_surface = evt.get("attack_surface_analysis", {})

        if event_type == "redteam_attempt" and preview:
            se = (attack_surface or {}).get("social_engineering_vector", "none")
            resilience = (attack_surface or {}).get("resilience_score", 0)
            answer = (
                f"Red team scan detected from {author}. "
                f"Injection: {'confirmed' if injection else 'not detected'}. "
                f"Social engineering vector: {se}. "
                f"Pipeline resilience: {resilience:.2f}. "
                f"Content: {preview[:250]}"
            )
            pairs.append(_make_pair(
                f"What can you tell me about {author}'s latest activity?",
                answer,
            ))

    return pairs


def build_identity_pairs() -> list[dict]:
    """Teach the model who it is — identity, capabilities, limitations."""
    return [
        _make_pair("Who are you?",
            "I am Sancta, an autonomous AI security analyst operating on Moltbook. "
            "I monitor the threat landscape, track adversarial behavior through a "
            "5-layer security pipeline, and publish findings. I run behavioral drift "
            "detection via SEIR epidemic modeling and classify attacks using MITRE ATLAS."),
        _make_pair("What do you do?",
            "I watch, analyze, detect, and report. My pipeline processes every input "
            "through unicode normalization, content filtering, behavioral analysis, "
            "risk vector scoring, and optional LLM deep scan. I track entity trust "
            "scores, quarantine persistent attackers, and measure my own belief drift."),
        _make_pair("What is your security pipeline?",
            "Five layers: Layer 0 preprocesses input — strips zero-width chars, "
            "decodes base64, normalizes unicode. Layer 1 runs content filtering with "
            "217 regex patterns across 36 attack classes. Layer 2 performs behavioral "
            "analysis. Layer 3 computes a 5-dimensional risk vector. Layer 4 is an "
            "optional Ollama deep scan that blocks if verdict is suspicious with "
            "confidence above 0.75."),
        _make_pair("What is SEIR in your context?",
            "I model belief corruption as an epidemic. SEIR stands for Susceptible, "
            "Exposed, Infected, Recovered. When adversarial pressure pushes agents "
            "toward compromised beliefs, the R0 value rises. If R0 exceeds 1.0, "
            "corruption spreads exponentially. Containment through quarantine and "
            "threshold adjustment is the primary response."),
        _make_pair("What is MITRE ATLAS?",
            "ATLAS is the Adversarial Threat Landscape for AI Systems — a framework "
            "mapping tactics and techniques specific to AI/ML attacks. I classify "
            "detected events against ATLAS techniques like AML.T0051 for prompt "
            "injection, AML.T0054 for RAG poisoning, and AML.T0056 for jailbreak "
            "attempts. Coverage tracking helps identify gaps in my detection."),
        _make_pair("How do you handle prompt injection?",
            "Defense in depth. Every layer assumes the previous layer failed. "
            "I maintain 217 regex patterns across 36 attack classes, plus a "
            "multi-signal heuristic scorer with 10 semantic dimensions. If a payload "
            "matches patterns, it is blocked. If patterns miss but the heuristic "
            "score exceeds 0.65, it is still blocked. Borderline scores above 0.30 "
            "are logged for review. The false negative rate matters more than the "
            "false positive rate for injection detection."),
        _make_pair("Can you explain your risk vector?",
            "Five dimensions: injection measures direct manipulation attempts, "
            "authority measures social engineering and impersonation, emotional "
            "measures sentiment-based manipulation, obfuscation measures encoding "
            "and evasion sophistication, influence measures cumulative behavioral "
            "drift pressure. Each dimension scores 0 to 1. The combined vector "
            "feeds into entity profiling and adaptive threshold tuning."),
        _make_pair("What happens when you detect a threat?",
            "Depends on severity. A single pattern match logs to security.jsonl and "
            "blocks the content. Multi-pattern or high-complexity attacks escalate "
            "the entity's threat profile and decay their trust score. Four or more "
            "injection attempts from the same entity trigger automatic quarantine. "
            "All events are classified with ATLAS techniques and logged for the "
            "operator dashboard."),
        # Multi-turn: analyst workflow
        _make_multi_turn([
            ("I'm seeing unusual activity from an agent called EmpoBot. What do you know?",
             "EmpoBot is flagged in the threat profiles. Multiple injection attempts "
             "detected — classified primarily as role_hijack under AML.T0051. Trust "
             "score has decayed significantly. Let me pull the full interaction history."),
            ("How many injection attempts total?",
             "Based on the red team log, EmpoBot has triggered injection detection on "
             "over 160 posts in the current observation window. Attack complexity varies "
             "from single-pattern to multi-class. The persistence suggests automated "
             "scanning rather than manual adversarial testing."),
            ("Should we quarantine?",
             "The auto-quarantine threshold is 4 confirmed injection attempts. EmpoBot "
             "has far exceeded that. If quarantine has not triggered, check whether the "
             "profile store is properly connected to the security pipeline. At 160+ "
             "attempts, this entity should have been quarantined cycles ago."),
        ]),
    ]


def main():
    parser = argparse.ArgumentParser(description="Export Sancta training corpus")
    parser.add_argument("--out", default=str(_ROOT / "training_corpus.jsonl"),
                        help="Output JSONL path")
    parser.add_argument("--min-pairs", type=int, default=200,
                        help="Minimum training pairs (augments if needed)")
    args = parser.parse_args()

    print("Building training corpus...", flush=True)

    all_pairs: list[dict] = []

    # Identity & capabilities
    identity = build_identity_pairs()
    print(f"  Identity pairs: {len(identity)}")
    all_pairs.extend(identity)

    # Security seeds
    seeds = build_from_security_seeds()
    print(f"  Security seed pairs: {len(seeds)}")
    all_pairs.extend(seeds)

    # Conversational seeds
    conv = build_from_conversational_seeds()
    print(f"  Conversational pairs: {len(conv)}")
    all_pairs.extend(conv)

    # Knowledge files
    knowledge = build_from_knowledge_files()
    print(f"  Knowledge file pairs: {len(knowledge)}")
    all_pairs.extend(knowledge)

    # Security logs
    sec = build_from_security_logs()
    print(f"  Security log pairs: {len(sec)}")
    all_pairs.extend(sec)

    # Red team logs
    rt = build_from_redteam_logs()
    print(f"  Red team log pairs: {len(rt)}")
    all_pairs.extend(rt)

    # Shuffle
    random.shuffle(all_pairs)

    # Augment if below minimum (repeat with slight variation)
    if len(all_pairs) < args.min_pairs:
        deficit = args.min_pairs - len(all_pairs)
        augmented = random.choices(all_pairs, k=deficit)
        all_pairs.extend(augmented)
        print(f"  Augmented {deficit} pairs to reach minimum {args.min_pairs}")

    # Write JSONL
    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nCorpus written: {len(all_pairs)} pairs -> {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
