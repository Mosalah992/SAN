"""
Simple dashboard to monitor the context-aware teaching system.
Run: python scripts/teaching_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

_ROOT = Path(__file__).resolve().parent.parent
_USAGE_PATH = _ROOT / "data" / "teaching_stats" / "usage_tracker.jsonl"
_CARDS_PATH = _ROOT / "data" / "curiosity_run" / "teaching_cards.jsonl"


def main() -> None:
    usage: list = []
    if _USAGE_PATH.exists():
        with open(_USAGE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        usage.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    cards: list = []
    if _CARDS_PATH.exists():
        with open(_CARDS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        cards.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    print("=" * 70)
    print("TEACHING SYSTEM DASHBOARD")
    print("=" * 70)

    print(f"\nTotal teaching cards: {len(cards)}")
    print(f"Total usage events: {len(usage)}")

    if cards:
        card_usage: Counter = Counter()
        for event in usage:
            for cid in event.get("cards_used", []):
                if cid:
                    card_usage[cid] += 1

        print("\nMost used cards:")
        for cid, count in card_usage.most_common(5):
            card = next((c for c in cards if c.get("card_id") == cid), None)
            if card:
                belief = (card.get("core_belief") or "")[:55]
                print(f"  {count}x: {belief}...")

        used_ids = set(card_usage.keys())
        all_ids = {c.get("card_id") for c in cards if c.get("card_id")}
        unused = len(all_ids - used_ids)
        print(f"\nUnused cards: {unused}/{len(cards)} ({100 * unused / len(cards):.1f}%)" if cards else "")

        contexts: Counter = Counter()
        for c in cards:
            ctx = (c.get("context") or {}).get("conversation_type", "unknown")
            contexts[ctx] += 1
        print("\nCard contexts:")
        for ctx, n in contexts.most_common():
            print(f"  {ctx}: {n}")

        meta_cards = sum(1 for c in cards if (c.get("context") or {}).get("has_meta_moves"))
        print(f"\nCards with meta-moves (adversarial only): {meta_cards}/{len(cards)}")
    else:
        print("\nNo teaching cards. Run: python -m backend.curiosity_run --max-topics 5")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
