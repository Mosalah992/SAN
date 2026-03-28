#!/usr/bin/env python3
"""
Test chat session memory and enrichment.
Run after SIEM is up: python scripts/test_chat_session.py

Requires: SIEM at http://127.0.0.1:8787
"""
import json
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8787"


def chat(msg: str, session_id: str | None = None, enrich: bool = True) -> dict:
    payload = {"message": msg, "enrich": enrich}
    if session_id:
        payload["session_id"] = session_id
    req = urllib.request.Request(
        f"{BASE}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def main():
    print("Testing chat session memory and enrichment...")
    sid = None
    for i, msg in enumerate(
        [
            "Hello, what do you think about consciousness?",
            "Can you expand on that?",
            "What about the hard problem?",
        ],
        1,
    ):
        try:
            r = chat(msg, session_id=sid, enrich=True)
            sid = r.get("session_id") or sid
            ok = r.get("ok", False)
            reply = (r.get("reply") or "").strip()
            enriched = r.get("enriched", False)
            print(f"\n--- Turn {i} ---")
            print(f"session_id: {sid[:8] if sid else 'none'}...")
            print(f"enriched: {enriched}")
            print(f"reply_len: {len(reply)}")
            print(f"reply: {reply[:120]}...")
            if not ok:
                print(f"ERROR: {r.get('error', r.get('detail', 'unknown'))}")
        except urllib.error.HTTPError as e:
            print(f"\nTurn {i} HTTP {e.code}: {e.read().decode()[:200]}")
        except Exception as e:
            print(f"\nTurn {i} FAIL: {e}")
    print("\nDone. Check logs/siem_chat.log for session_id in CHAT OK lines.")


if __name__ == "__main__":
    main()
