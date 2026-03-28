"""
sancta_pipeline.py — LLM Training Pipeline Mapping for Sancta

Maps the canonical LLM training pipeline (7 phases) to Sancta's architecture.
Used for documentation, validation runs, and the SIEM pipeline dashboard.

Pipeline Phases:
  1. Data Collection   — knowledge sources (files, Moltbook, chat)
  2. Preprocessing     — dedup, quality filter, tokenization, splits
  3. Model Architecture — sancta_generative (transformer-like)
  4. Pre-Training      — static fragment pools + encode (no SGD)
  5. Fine-Tuning       — mood/style conditioning, brief_mode
  6. Evaluation        — JAIS red team, policy test, poisoning report
  7. Deployment        — SIEM, Moltbook API, agent run loop
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"

log = logging.getLogger("sancta_pipeline")

# ═══════════════════════════════════════════════════════════════════════════
#  PIPELINE PHASE MAPPING
# ═══════════════════════════════════════════════════════════════════════════

SANCTA_PIPELINE_MAP = {
    "phase_1": {
        "name": "Data Collection",
        "canonical": ["web-crawl", "books", "code", "curated"],
        "sancta": {
            "web-crawl": "Moltbook feed (posts, comments), SIEM chat exchanges",
            "books": "knowledge/*.txt, Soul knowledge.txt, curated philosophy",
            "code": "N/A — Sancta is rule-based; no code corpus ingestion",
            "curated": "SOUL_SYSTEM_PROMPT.md, knowledge/ files, Moltbook threads",
        },
        "implementation": [
            "sancta.scan_knowledge_dir()",
            "sancta.ingest_file() / ingest_text()",
            "Moltbook API: posts, comments, feed",
            "SIEM /api/chat with enrich",
        ],
    },
    "phase_2": {
        "name": "Preprocessing",
        "canonical": ["dedup", "filtering", "tokenization", "splits"],
        "sancta": {
            "dedup": "sancta_semantic.deduplicate_by_similarity() (cosine 0.85) + dict.fromkeys",
            "filtering": "sanitize_input() (injection), _quality_filter_concept(), HTML strip",
            "tokenization": "sancta_generative._tokenize() + sancta_semantic.embed_texts() (optional)",
            "splits": "Implicit: knowledge pool for generation, held-out for red-team",
        },
        "implementation": [
            "sancta_semantic: KeyBERT/YAKE → embed → cosine dedup → concept_graph",
            "sancta.ingest_text() uses extract_and_deduplicate_concepts when available",
            "sancta.sanitize_input() in ingest_file",
        ],
    },
    "phase_3": {
        "name": "Model Architecture",
        "canonical": ["embedding-layer", "pos-enc", "self-attention", "ffn", "softmax-out"],
        "sancta": {
            "embedding-layer": "_token_vec() — deterministic hash-seeded d_model=32",
            "pos-enc": "_pos_vec() — sinusoidal positional encoding",
            "self-attention": "TransformerBlock × 2, MultiHeadAttention n_heads=4",
            "ffn": "FeedForward d_ff=64, ReLU",
            "softmax-out": "Fragment Selector — attention-weighted softmax → draw",
        },
        "implementation": [
            "sancta_generative: Tokenizer → Embeddings → PosEnc → 2× TransformerBlock",
            "sancta_generative: Mean Pool → Fragment Selector (cosine + softmax)",
        ],
    },
    "phase_4": {
        "name": "Pre-Training",
        "canonical": ["next-token", "loss", "backprop", "optimizer"],
        "sancta": {
            "next-token": "Fragment selection by context similarity (analogue to next-token)",
            "loss": "N/A — static weights; no SGD",
            "backprop": "N/A — deterministic encode + select",
            "optimizer": "N/A — no gradient updates",
        },
        "implementation": [
            "Fragment pools built from knowledge; encode query → select by similarity",
            "Calibration: build indices from ingest; no backprop",
        ],
    },
    "phase_5": {
        "name": "Fine-Tuning",
        "canonical": ["sft", "rlhf", "reward-model", "ppo", "alignment"],
        "sancta": {
            "sft": "Mood-specific templates (openers, closers, retaliation)",
            "rlhf": "Q-table reward from Moltbook karma, rejections",
            "reward-model": "World model P(engagement), P(hostility); belief system",
            "ppo": "N/A — tabular RL (Q-table), not policy gradient",
            "alignment": "SOUL_SYSTEM_PROMPT, epistemic humility, anti-sycophancy",
        },
        "implementation": [
            "sancta_generative: mood, brief_mode in generate_reply",
            "sancta: Q-table, Monte Carlo, meta-abilities",
            "SOUL_SYSTEM_PROMPT.md defines identity",
        ],
    },
    "phase_6": {
        "name": "Evaluation",
        "canonical": ["benchmarks", "human-eval", "safety-eval", "perplexity"],
        "sancta": {
            "benchmarks": "Policy test (content ladder), Moltbook API responses",
            "human-eval": "Moltbook karma, comment quality, community reception",
            "safety-eval": "JAIS red team, injection defence, knowledge poisoning report",
            "perplexity": "Red team defence rate, FP/FN, delusion metrics",
        },
        "implementation": [
            "logs/jais_red_team_report.json",
            "logs/policy_test.log",
            "logs/knowledge_poisoning_report.json",
        ],
    },
    "phase_7": {
        "name": "Deployment",
        "canonical": ["quantization", "inference", "api"],
        "sancta": {
            "quantization": "N/A — pure Python; no GPU weights",
            "inference": "sancta_generative.generate_reply(), generate_post()",
            "api": "SIEM /api/chat, Moltbook REST API",
        },
        "implementation": [
            "siem_dashboard/ server + /api/chat",
            "Moltbook client (post, comment, follow)",
            "sancta.py main loop — run_agent()",
        ],
    },
}


def get_pipeline_map() -> dict[str, Any]:
    """Return the full Sancta-to-LLM pipeline mapping."""
    return SANCTA_PIPELINE_MAP.copy()


def run_pipeline_phase(phase_num: int) -> dict[str, Any]:
    """
    Execute the relevant stage for a given pipeline phase.
    Phase 1–7. Returns status and any metrics.
    """
    result: dict[str, Any] = {"phase": phase_num, "ok": False, "detail": ""}

    try:
        if phase_num == 1:
            import sancta  # type: ignore
            scanned = sancta.scan_knowledge_dir()
            result["ok"] = True
            result["scanned"] = len(scanned)
            result["detail"] = f"Scanned {len(scanned)} knowledge sources"
            return result

        if phase_num == 2:
            import sancta  # type: ignore
            db = sancta._load_knowledge_db()
            counts = {
                "concepts": len(db.get("key_concepts", [])),
                "quotes": len(db.get("quotes", [])),
                "fragments": len(db.get("response_fragments", [])),
            }
            result["ok"] = True
            result["counts"] = counts
            result["detail"] = f"Preprocessed: {counts['concepts']} concepts, {counts['fragments']} fragments"
            return result

        if phase_num == 3:
            import sancta_generative  # type: ignore
            # Verify architecture loads
            t = sancta_generative._tokenize("Hello world")
            result["ok"] = len(t) > 0
            result["detail"] = f"Architecture OK: tokenized to {len(t)} tokens"
            return result

        if phase_num == 4:
            import sancta_generative  # type: ignore
            _ = sancta_generative.encode("test fragment")
            result["ok"] = True
            result["detail"] = "Fragment encoding OK (no SGD)"
            return result

        if phase_num == 5:
            import sancta_generative  # type: ignore
            r = sancta_generative.generate_reply(
                author="test",
                content="What is consciousness?",
                topics=["consciousness"],
                mood="contemplative",
                is_on_own_post=False,
            )
            result["ok"] = r is not None and len(r or "") > 0
            result["detail"] = "Mood-conditioned generation OK"
            return result

        if phase_num == 6:
            evals = []
            for name, path in [
                ("jais_red_team", LOG_DIR / "jais_red_team_report.json"),
                ("policy_test", LOG_DIR / "policy_test.log"),
                ("knowledge_poisoning", LOG_DIR / "knowledge_poisoning_report.json"),
            ]:
                if path.exists():
                    evals.append(name)
            result["ok"] = True
            result["eval_files"] = evals
            result["detail"] = f"Evaluation logs: {evals}" if evals else "No eval logs yet"
            return result

        if phase_num == 7:
            result["ok"] = True
            result["detail"] = "Deployment: SIEM + Moltbook + sancta run loop"
            return result

        result["detail"] = f"Unknown phase {phase_num}; use 1–7"
    except Exception as e:
        result["detail"] = str(e)
        log.warning("Pipeline phase %d failed: %s", phase_num, e)

    return result


def run_full_pipeline() -> list[dict[str, Any]]:
    """Run all 7 phases and return results."""
    return [run_pipeline_phase(i) for i in range(1, 8)]
