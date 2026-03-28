#!/usr/bin/env python3
"""
SanctaGPT smoke train on AI-security dialogues + optional curriculum.

Stability notes (see sancta_gpt.py):
  - LR default 0.001, random doc sampling, BLOCK_SIZE 128
  - Override LR: env SANCTA_GPT_LR=0.0005

From repo root:
  python backend/smoke_train_sancta_gpt_security.py --light --dry-run
  python backend/smoke_train_sancta_gpt_security.py --light --curriculum --steps 400 --fresh
  python backend/smoke_train_sancta_gpt_security.py --light --steps 200 --fresh

--curriculum (requires --light): Phase 1 samples only conversational + User:/Sancta: lines;
  Phase 2 samples the full light corpus (adds SECURITY_SEEDS). Reduces style collision.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import sancta_gpt  # noqa: E402
from sancta_gpt import CONVERSATIONAL_ENGLISH_SEEDS, SECURITY_SEEDS  # noqa: E402

AI_SECURITY_TURNS: list[tuple[str, str]] = [
    (
        "What is OWASP LLM01?",
        "LLM01 is prompt injection: inputs crafted to override system intent or leak instructions.",
    ),
    (
        "What is indirect prompt injection?",
        "Instructions live in data the model retrieves—RAG docs, web pages, email—so they look like trusted context.",
    ),
    (
        "Why is LLM08 dangerous?",
        "Excessive agency: the model or agent can act—tools, APIs, purchases—so bad output becomes real impact.",
    ),
    (
        "How do defenders reduce RAG poisoning risk?",
        "Integrity-check the corpus, segregate untrusted text, monitor retrievals, and validate outputs before actions.",
    ),
    (
        "Name one MITRE ATLAS technique for agents.",
        "AML.T0053 Agent tool invocation: adversary steers the agent into privileged tool calls.",
    ),
    (
        "One-line zero trust for AI?",
        "Never trust model output or retrieved context by default—verify, authorize, and log.",
    ),
]


def _chunk_strings(raw_docs: list[str], block_size: int) -> list[str]:
    out: list[str] = []
    for d in raw_docs:
        cleaned = "".join(c for c in d if 32 <= ord(c) < 127 or c in "\n\t").strip()
        if len(cleaned) < 12:
            continue
        if len(cleaned) > block_size:
            step = max(1, block_size // 2)
            for i in range(0, len(cleaned) - 20, step):
                chunk = cleaned[i : i + block_size]
                if len(chunk) > 12:
                    out.append(chunk)
        else:
            out.append(cleaned)
    return out


def _raw_dialogue_only() -> list[str]:
    docs: list[str] = []
    docs.extend(CONVERSATIONAL_ENGLISH_SEEDS[:40])
    for user, reply in AI_SECURITY_TURNS:
        docs.append(f"User: {user}\nSancta: {reply}")
    return [d for d in docs if d and len(d.strip()) > 12]


def _raw_light_mixed() -> list[str]:
    docs: list[str] = []
    docs.extend(SECURITY_SEEDS[:35])
    docs.extend(CONVERSATIONAL_ENGLISH_SEEDS[:25])
    for user, reply in AI_SECURITY_TURNS:
        docs.append(f"User: {user}\nSancta: {reply}")
    return [d for d in docs if d and len(d.strip()) > 12]


def _run_steps(engine: object, n: int, label: str, log_every: int) -> None:
    for i in range(n):
        loss = engine.train_step()
        if log_every > 0 and (i == 0 or (i + 1) % log_every == 0):
            print(f"  [{label}] step {i + 1}/{n} loss={loss:.4f}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke-train SanctaGPT on AI security dialogues.")
    ap.add_argument("--steps", type=int, default=50, help="Total training steps (or phase1+phase2 if --curriculum)")
    ap.add_argument("--fresh", action="store_true", help="Remove checkpoint and reinit weights")
    ap.add_argument("--no-save", action="store_true", help="Do not write checkpoint after training")
    ap.add_argument(
        "--light",
        action="store_true",
        help="Seeds + dialogues only (no full knowledge/). Each CPU train_step is still slow.",
    )
    ap.add_argument(
        "--curriculum",
        action="store_true",
        help="With --light: phase1 = conversational + User:/Sancta: only; phase2 = add security seeds.",
    )
    ap.add_argument(
        "--phase1-steps",
        type=int,
        default=0,
        help="Curriculum: explicit phase-1 step count (0 = use half of --steps)",
    )
    ap.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="Print loss every N steps (0 = auto from total steps)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only build corpus and print sizes; no train_step.",
    )
    args = ap.parse_args()
    total_steps = max(1, min(args.steps, 100_000))

    if args.curriculum and not args.light:
        print("ERROR: --curriculum requires --light", flush=True)
        return 2

    engine = sancta_gpt.get_engine()
    ckpt = engine._checkpoint_path

    if args.fresh and ckpt.exists():
        ckpt.unlink()
        print(f"Removed {ckpt}", flush=True)
        sancta_gpt._engine = None  # type: ignore[attr-defined]
        engine = sancta_gpt.get_engine()

    bs = engine.BLOCK_SIZE

    if args.light:
        if args.curriculum:
            raw_p1 = _raw_dialogue_only()
            raw_p2_extra = SECURITY_SEEDS[:35]
            chunks_p1 = _chunk_strings(raw_p1, bs)
            chunks_p2_only = _chunk_strings(raw_p2_extra, bs)
            engine._pool_convo = list(chunks_p1)
            engine._pool_security = list(chunks_p2_only)
            engine._pool_knowledge = []
            engine._pool_telemetry = []
            engine._docs = engine._pool_convo + engine._pool_security
            print(
                f"Curriculum light corpus: convo_pool={len(engine._pool_convo)}, "
                f"security_pool={len(engine._pool_security)}, total={len(engine._docs)}",
                flush=True,
            )
        else:
            engine._docs = _chunk_strings(_raw_light_mixed(), bs)
            engine._pool_convo = []
            engine._pool_security = []
            engine._pool_knowledge = []
            engine._pool_telemetry = []
            print(f"Light corpus: {len(engine._docs)} chunks (mixed seeds + AI security turns)", flush=True)
        engine._corpus_size = len(engine._docs)
        engine._initialized = False
        engine._init_model()
        print(
            f"Initialized weights | BLOCK_SIZE={bs} LR={engine.LR} "
            f"(env SANCTA_GPT_LR overrides) | random sampling on",
            flush=True,
        )
    else:
        if ckpt.exists() and not args.fresh:
            loaded = engine.load()
            print(f"load checkpoint: {loaded} step={engine._step} loss={engine._last_loss}", flush=True)
        n_docs = engine.build_corpus()
        print(f"Corpus chunks after build_corpus: {n_docs}", flush=True)
        if not engine._initialized:
            engine._init_model()
            print("Initialized new model weights.", flush=True)

    if not args.light:
        added = 0
        for user, reply in AI_SECURITY_TURNS:
            if engine.learn_from_interaction(user, reply):
                added += 1
        print(f"Ingested {added} dialogue documents.", flush=True)
    else:
        print("Light mode: dialogues in corpus.", flush=True)

    if args.dry_run:
        print(f"dry-run: corpus_size={engine._corpus_size} initialized={engine._initialized}", flush=True)
        return 0

    log_every = args.log_every
    if log_every <= 0:
        log_every = max(1, min(100, total_steps // 10))

    t0 = time.perf_counter()
    if args.curriculum and args.light:
        p1 = args.phase1_steps if args.phase1_steps > 0 else max(1, total_steps // 2)
        p2 = total_steps - p1
        if p2 < 1:
            p1, p2 = total_steps - 1, 1
        engine.set_training_mode("convo")
        print(f"Phase 1: training_mode=convo for {p1} steps", flush=True)
        _run_steps(engine, p1, "P1", log_every)
        engine.set_training_mode("security")
        print(f"Phase 2: training_mode=security for {p2} steps", flush=True)
        _run_steps(engine, p2, "P2", log_every)
        engine.set_training_mode("all")
    else:
        engine.set_training_mode("all")
        engine.set_training_doc_range(None, None)
        _run_steps(engine, total_steps, "train", log_every)

    elapsed = time.perf_counter() - t0
    print(f"Trained in {elapsed:.2f}s | last_loss={engine._last_loss:.4f}", flush=True)

    if not args.no_save:
        ok = engine.save()
        print(f"checkpoint save: {ok} -> {ckpt}", flush=True)

    print("\n--- Sample generations (loss != intelligence; check format drift) ---", flush=True)
    probes = [
        "User: Hello\nSancta: ",
        "Operator: What is LLM01?\nSancta: ",
        "Operator: Explain RAG poisoning.\nSancta: ",
    ]
    for pr in probes:
        out = engine.generate(pr, max_tokens=100, temperature=0.75)
        print(f"{pr!r}\n  -> {out!r}\n", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
