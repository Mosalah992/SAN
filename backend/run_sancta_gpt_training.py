#!/usr/bin/env python3
"""
SanctaGPT offline training: rebuild corpus (security + conversational English + knowledge/) then train.

Usage (from repo root):
  python backend/run_sancta_gpt_training.py [steps] [--fresh]

  --fresh   Ignore existing checkpoint and reinitialize weights (use after major corpus changes).
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


def _safe_print(text: str) -> None:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    sys.stdout.write(text.encode(encoding, errors="backslashreplace").decode(encoding, errors="ignore"))
    sys.stdout.flush()


def main() -> int:
    p = argparse.ArgumentParser(description="Train SanctaGPT on built-in + knowledge corpus.")
    p.add_argument("steps", nargs="?", type=int, default=350, help="Training steps (default 350)")
    p.add_argument("--fresh", action="store_true", help="Delete checkpoint and train from scratch")
    args = p.parse_args()
    steps = max(1, min(args.steps, 50_000))

    engine = sancta_gpt.get_engine()
    ckpt = engine._checkpoint_path

    if args.fresh:
        if ckpt.exists():
            ckpt.unlink()
            print(f"Removed checkpoint: {ckpt}", flush=True)
        sancta_gpt._engine = None  # type: ignore[attr-defined]
        engine = sancta_gpt.get_engine()

    n_docs = engine.build_corpus()
    print(f"Corpus chunks: {n_docs}", flush=True)

    loaded = False
    if ckpt.exists() and not args.fresh:
        loaded = engine.load()
        print(f"Checkpoint load: {loaded} | step={engine._step} loss={engine._last_loss}", flush=True)

    if not engine._initialized:
        engine._init_model()
        print("Fresh model init.", flush=True)

    t0 = time.time()
    final = engine.train(num_steps=steps, log_every=max(20, steps // 15))
    elapsed = time.time() - t0
    saved = engine.save()
    print(f"Trained {steps} steps in {elapsed:.1f}s | final_loss={final:.6f} | saved={saved}", flush=True)

    prompts = [
        "Hello, ",
        "Operator: What is a firewall?\nSancta: ",
        "Threat detected: ",
        "Thanks for your help. ",
    ]
    print("\n--- Sample generations (temperature 0.75) ---", flush=True)
    for pr in prompts:
        out = engine.generate(pr, max_tokens=72, temperature=0.75)
        _safe_print(f"PROMPT {pr!r}\n  -> {out!r}\n\n")

    st = engine.status()
    print("status:", {k: st[k] for k in ("step", "last_loss", "corpus_size", "vocab_size", "num_params")}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
