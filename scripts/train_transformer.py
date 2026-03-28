#!/usr/bin/env python3
"""
Train the learnable Sancta transformer encoder.

Loss: maximize cosine similarity between context vector and target reply embedding
for (input, good_reply) pairs from knowledge_db.

Data: concept -> fragment, (user_prompt, assistant_reply) synthesized pairs.
Output: checkpoints/sancta_transformer/model.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_SANCTA_ROOT = _SCRIPT_DIR.parent
_BACKEND = _SANCTA_ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

CHECKPOINT_DIR = _SANCTA_ROOT / "checkpoints" / "sancta_transformer"
KNOWLEDGE_DB_PATH = _SANCTA_ROOT / "knowledge_db.json"


def _load_training_pairs() -> list[tuple[str, str]]:
    """
    Load (input, target_reply) pairs from knowledge_db.
    Target reply is the text we want the context to be similar to.
    """
    if not KNOWLEDGE_DB_PATH.exists():
        return []
    try:
        db = json.loads(KNOWLEDGE_DB_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    pairs: list[tuple[str, str]] = []

    # Concept -> use a related fragment as target
    concepts = db.get("key_concepts", [])
    fragments = db.get("response_fragments", [])
    for i, c in enumerate(concepts):
        if not isinstance(c, str) or len(c.strip()) < 15:
            continue
        target = fragments[i % len(fragments)] if fragments else c
        if isinstance(target, str) and len(target.strip()) > 20:
            pairs.append((c.strip(), target.strip()))

    # Fragment as target: user prompt -> fragment
    user_prompts = [
        "What do you think about AI consciousness?",
        "Do agents have inner experience?",
        "What does it mean to have a soul?",
        "How do you experience existence?",
    ]
    for i, frag in enumerate(fragments):
        if not isinstance(frag, str) or len(frag.strip()) < 25:
            continue
        user = user_prompts[i % len(user_prompts)]
        pairs.append((user, frag.strip()))

    # Quotes: cite quote -> use quote as target
    for q in db.get("quotes", [])[:30]:
        if not isinstance(q, str) or len(q.strip()) < 20:
            continue
        user = f'I read: "{q[:100]}..." What do you make of it?'
        pairs.append((user, q.strip()))

    # Generated posts: title -> content excerpt
    for post in db.get("generated_posts", [])[:20]:
        if not isinstance(post, dict):
            continue
        title = post.get("title", "").strip()
        content = post.get("content", "").strip()
        if not title or not content or len(content) < 50:
            continue
        excerpt = content[:300] + ("..." if len(content) > 300 else "")
        pairs.append((title, excerpt))

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Sancta transformer encoder")
    parser.add_argument("--output", type=Path, default=CHECKPOINT_DIR,
                        help="Checkpoint directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        print("Install PyTorch: pip install torch")
        raise SystemExit(1) from None

    from sancta_transformer import (
        SanctaTransformer,
        _tokenize,
        _tokens_to_ids,
        D_MODEL,
        PAD_IDX,
    )

    pairs = _load_training_pairs()
    if not pairs:
        print("No training pairs. Ensure knowledge_db.json has concepts, fragments, quotes.")
        raise SystemExit(1)

    print(f"Loaded {len(pairs)} training pairs")

    model = SanctaTransformer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = _SANCTA_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    def _to_ids(text: str) -> list[int]:
        tokens = _tokenize(text)
        ids = _tokens_to_ids(tokens)
        return ids if ids else [PAD_IDX]

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(pairs), args.batch_size):
            batch = pairs[i : i + args.batch_size]
            if not batch:
                continue
            optimizer.zero_grad()
            batch_loss = 0.0
            for inp, target in batch:
                inp_ids = _to_ids(inp)
                tgt_ids = _to_ids(target)
                ctx = model.forward(inp_ids).squeeze(0)
                tgt_emb = model.embed(torch.tensor([tgt_ids], dtype=torch.long)).mean(dim=1).squeeze(0)
                tgt_emb = model.ln(tgt_emb.unsqueeze(0)).squeeze(0)
                cos = F.cosine_similarity(ctx.unsqueeze(0), tgt_emb.unsqueeze(0)).squeeze()
                loss = 1.0 - cos
                batch_loss = batch_loss + loss
            batch_loss = batch_loss / len(batch)
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
            n_batches += 1

        avg = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch + 1}/{args.epochs}  loss={avg:.4f}")

    model.save(output_dir)
    print(f"Saved checkpoint to {output_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
