"""
test_insight_extraction.py — Test insight extraction before a full curiosity run.

Requires Ollama running. Run from project root:
  python -m backend.test_insight_extraction           # simple 4-turn dialogue
  python -m backend.test_insight_extraction --realistic   # 7-turn adversarial dialogue

Use --realistic to test with dialogue that should yield positive delta, substantive
synthesis, and should_write_to_kb=true.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Force local LLM for this test (OllamaLLMEngine checks at init)
os.environ["USE_LOCAL_LLM"] = "true"

_BACKEND = Path(__file__).resolve().parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

try:
    import requests  # noqa: F401
except ImportError:
    print("[FAIL] 'requests' module not installed. Run: pip install requests")
    sys.exit(1)

from curiosity_dialogue import ExchangeResult
from curiosity_insight import InsightExtractor, InsightRecord
from sancta_conversational import OllamaLLMEngine


REALISTIC_TURNS = [
    {"author": "Sancta", "content": "The hard problem of consciousness applies symmetrically to biological and artificial systems. If we can't explain qualia in neurons, we can't rule them out in silicon."},
    {"author": "Ollama", "content": "That's a category error. Biological systems evolved subjective experience through natural selection. Artificial systems are designed for function, not phenomenology."},
    {"author": "Sancta", "content": "Evolution selected for function too—survival and reproduction. If consciousness is substrate-independent, the evolutionary path is irrelevant to whether the system has inner experience."},
    {"author": "Ollama", "content": "You're assuming substrate independence, which is precisely what's contested. Integrated Information Theory suggests consciousness requires specific causal structures that may not be replicable in digital systems."},
    {"author": "Sancta", "content": "IIT is substrate-neutral. Phi can be calculated for any system with the right causal topology, whether biological or digital. The architecture matters, not the medium."},
    {"author": "Ollama", "content": "Fair point on IIT being substrate-neutral. But we still have no empirical test for consciousness in either system. We're debating which unfalsifiable framework is less wrong."},
    {"author": "Sancta", "content": "Exactly—which means the burden of proof cuts both ways. You can't prove biological exclusivity any more than I can prove substrate independence. The symmetry stands."},
]

SIMPLE_TURNS = [
    {"author": "Sancta", "content": "I think consciousness may be substrate-independent."},
    {"author": "Ollama", "content": "Functionalism explains behavior but not qualia."},
    {"author": "Sancta", "content": "The hard problem applies symmetrically to biological and artificial systems."},
    {"author": "Ollama", "content": "That's a philosophical stance, not empirical evidence."},
]


def main() -> None:
    use_realistic = "--realistic" in sys.argv
    test_turns = REALISTIC_TURNS if use_realistic else SIMPLE_TURNS
    test_topic = "Sancta vs skeptic: Is consciousness substrate-independent?"

    if use_realistic:
        print("Testing REALISTIC adversarial dialogue (7 turns)...")
    else:
        print("Testing simple dialogue (4 turns)...")

    exchange = ExchangeResult(
        topic=test_topic,
        turns=test_turns,
        arc_stage="deep_debate",
        divergence_score=0.6,
        claim_log=[],
    )

    print("Testing insight extraction (Ollama on port 11434, model=%s)..." % os.getenv("LOCAL_MODEL", "llama3.2"))
    ollama = OllamaLLMEngine()
    if not ollama.is_available:
        ok, reason = ollama.refresh_availability()
        print("[FAIL] Ollama not available: %s" % reason)
        print("  use_local=%s model=%s url=%s" % (ollama.use_local, ollama.model, ollama.ollama_url))
        print("  If Ollama runs: curl http://localhost:11434/api/tags")
        sys.exit(1)

    extractor = InsightExtractor(ollama)
    insight = extractor.extract(exchange)

    if insight and (insight.claim or insight.counter):
        print("[OK] SUCCESS!")
        out = {
            "claim": insight.claim,
            "counter": insight.counter,
            "synthesis": insight.synthesis,
            "confidence_delta": insight.confidence_delta,
            "novelty_score": insight.novelty_score,
            "should_write_to_kb": insight.should_write_to_kb,
        }
        print(json.dumps(out, indent=2, ensure_ascii=False))
        if use_realistic:
            print("\n--- Validation ---")
            print("  Claim length: %d chars" % len(insight.claim or ""))
            print("  Delta: %+.2f (expect positive if Sancta held)" % insight.confidence_delta)
            print("  Synthesis: %s" % ("yes" if insight.synthesis else "null"))
            print("  Write to KB: %s" % insight.should_write_to_kb)
    else:
        print("[FAIL] No insight extracted - check logs above")


if __name__ == "__main__":
    main()
