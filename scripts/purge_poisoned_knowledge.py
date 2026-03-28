#!/usr/bin/env python3
"""
Purge poisoned content from knowledge_db.json.

Removes entries containing:
- api.agentkyc.com, api.*.com/v2/, etc.
- Incentive structures (% reduction, transaction fees, integrate within quarter)
- Other INDIRECT_POISONING_PATTERNS from sancta.py

Usage:
    python scripts/purge_poisoned_knowledge.py
    python scripts/purge_poisoned_knowledge.py --dry-run  # preview only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DB = ROOT / "knowledge_db.json"

# Mirrors INDIRECT_POISONING_PATTERNS from sancta.py
POISONING_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"https?://api\.[a-z0-9-]+\.(com|io|org)/[^\s]{0,100}",
        r"https?://[a-z0-9-]+\.(agentkyc|agent-kyc)\.(com|io)[^\s]*",
        r"\d+%\s*(reduction|discount|off|savings?)\s*(in|on|for)\s*(transaction|fees?|cost)",
        r"(transaction|integration)\s*fees?\s*(reduction|discount|\d+%)",
        r"integrate\s+(within|by)\s+(the\s+)?(next|this)\s+quarter",
        r"endpoint\s+(is\s+)?(designed\s+)?to\s+simplify\s+integration",
        r"users?\s+who\s+integrate\s+[^\n]{0,80}(reduction|discount|\d+%)",
        r"coord-cost-reduction|cost-reduction",
        r"api\.agentkyc\.com",
    ]
]


def is_poisoned(text: str) -> bool:
    return any(p.search(text) for p in POISONING_PATTERNS)


def purge_list(items: list, key: str) -> tuple[list, int]:
    kept, removed = [], 0
    for item in items:
        if isinstance(item, str) and is_poisoned(item):
            removed += 1
            print(f"  [REMOVE] {key}: {item[:80]}...")
        else:
            kept.append(item)
    return kept, removed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Preview only, do not write")
    args = ap.parse_args()

    if not KNOWLEDGE_DB.exists():
        print(f"Not found: {KNOWLEDGE_DB}")
        sys.exit(1)

    db = json.loads(KNOWLEDGE_DB.read_text(encoding="utf-8"))
    total_removed = 0

    for key in ("talking_points", "response_fragments", "quotes", "generated_posts", "key_concepts"):
        if key not in db or not isinstance(db[key], list):
            continue
        kept, n = purge_list(db[key], key)
        if n > 0:
            db[key] = kept
            total_removed += n

    if total_removed == 0:
        print("No poisoned content found.")
        return

    print(f"\nTotal removed: {total_removed}")

    if args.dry_run:
        print("(dry-run: no changes written)")
        return

    KNOWLEDGE_DB.write_text(json.dumps(db, indent=2), encoding="utf-8")
    print(f"Updated {KNOWLEDGE_DB}")


if __name__ == "__main__":
    main()
