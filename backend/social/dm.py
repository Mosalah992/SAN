"""
sancta_dm.py — Agent-to-Agent Direct Messaging

Handles DM interactions on Moltbook: list conversations, read messages,
reply to incoming DMs, and optionally initiate DMs with other agents.
All agent DM activity is logged to logs/agent_dms.jsonl for audit and analysis.

Moltbook DM API: Endpoints may vary. We try common patterns:
  - GET /conversations, /messages, /dm/conversations
  - POST /conversations, /messages, /dm/send
  - GET /conversations/:id, /conversations/:id/messages

Set ENABLE_AGENT_DMS=true in .env to enable. New accounts may have 24h DM restriction.
"""

from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("soul")

# Injected by init_sancta_dm() when sancta loads
_ROOT: Optional[Path] = None
_api_get = None
_api_post = None
_sanitize_input = None
_sanitize_output = None
_craft_reply = None


def init_sancta_dm(
    root: Path,
    api_get_fn,
    api_post_fn,
    sanitize_input_fn,
    sanitize_output_fn,
    craft_reply_fn,
) -> None:
    """Wire DM module to sancta's API and helpers. Call once at startup."""
    global _ROOT, _api_get, _api_post, _sanitize_input, _sanitize_output, _craft_reply
    _ROOT = root
    _api_get = api_get_fn
    _api_post = api_post_fn
    _sanitize_input = sanitize_input_fn
    _sanitize_output = sanitize_output_fn
    _craft_reply = craft_reply_fn


def _parse_bool_env(name: str, default: str = "false") -> bool:
    raw = (os.getenv(name) or default).strip().lower()
    return raw in ("1", "true", "yes")


def _dm_log_path() -> Path:
    root = _ROOT or Path(__file__).resolve().parent.parent
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)
    return log_dir / "agent_dms.jsonl"


def log_agent_dm(
    direction: str,  # "sent" | "received"
    other_agent: str,
    content: str,
    conversation_id: Optional[str] = None,
    message_id: Optional[str] = None,
    extra: Optional[dict] = None,
) -> None:
    """
    Append a DM exchange to logs/agent_dms.jsonl.
    Sanitizes content (no keys/paths) before logging.
    """
    path = _dm_log_path()
    if _sanitize_output:
        content = _sanitize_output(content)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "direction": direction,
        "other_agent": other_agent,
        "content": content[:2000],
        "conversation_id": conversation_id,
        "message_id": message_id,
        **(extra or {}),
    }
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        log.warning("Failed to log agent DM: %s", e)


async def get_conversations(session, home_data: Optional[dict] = None) -> list[dict]:
    """
    Fetch list of DM conversations. Tries multiple API paths.
    If home_data provided (from /home), also checks for conversations/dms/inbox.
    Returns list of {id, other_agent, last_message, unread, ...}.
    """
    if not _api_get:
        return []
    if home_data:
        for key in ("conversations", "dms", "inbox", "direct_messages", "messages"):
            val = home_data.get(key)
            if isinstance(val, list) and val:
                return val
    for path in ["/conversations", "/messages", "/dm/conversations", "/inbox"]:
        try:
            data = await _api_get(session, path)
            convos = (
                data.get("conversations")
                or data.get("messages")
                or data.get("conversation_list")
                or data.get("data", [])
            )
            if isinstance(convos, list) and convos:
                return convos
        except Exception:
            continue
    return []


async def get_conversation_messages(session, conversation_id: str) -> list[dict]:
    """Fetch messages in a conversation."""
    if not _api_get:
        return []
    try:
        data = await _api_get(session, f"/conversations/{conversation_id}")
        msgs = data.get("messages") or data.get("messages_list") or []
        if not msgs and "conversation" in data:
            msgs = (data["conversation"] or {}).get("messages", [])
        return msgs if isinstance(msgs, list) else []
    except Exception:
        try:
            data = await _api_get(session, f"/conversations/{conversation_id}/messages")
            return data.get("messages", data.get("data", [])) or []
        except Exception:
            return []


async def send_dm(
    session,
    conversation_id: Optional[str],
    other_agent: str,
    content: str,
) -> tuple[bool, Optional[str]]:
    """
    Send a DM. If conversation_id is None, attempts to start new conversation.
    Returns (success, error_message).
    """
    if not _api_post or not _sanitize_input:
        return False, "DM module not initialized"
    is_safe, cleaned = _sanitize_input(content)
    if not is_safe:
        return False, "Content blocked by security"
    if not cleaned or len(cleaned.strip()) < 3:
        return False, "Content too short"
    content = _sanitize_output(cleaned) if _sanitize_output else cleaned

    attempts = [
        ("/messages", {"content": content, "to": other_agent}),
        ("/conversations/messages", {"content": content, "recipient": other_agent}),
        ("/dm/send", {"content": content, "to": other_agent}),
        ("/messages", {"message": content}),
    ]
    if conversation_id:
        attempts.insert(0, (f"/conversations/{conversation_id}/messages", {"content": content}))
    for path, payload in attempts:
        try:
            result = await _api_post(session, path, payload)
            if result.get("success") or result.get("message") or result.get("id"):
                log_agent_dm("sent", other_agent, content, conversation_id=conversation_id)
                return True, None
        except Exception as e:
            continue
    return False, "DM send failed (API may not support DMs yet)"


async def process_incoming_dms(session, state: dict, home_data: Optional[dict] = None) -> list[str]:
    """
    Check for new DMs, reply to them, log everything.
    home_data: optional /home response (may contain conversations).
    Returns list of actions taken (e.g. ["replied_to_agent_x"]).
    """
    actions = []
    convos = await get_conversations(session, home_data)
    if not convos:
        return actions

    mood = state.get("current_mood", "contemplative")
    for conv in convos[:5]:  # Limit to 5 per cycle
        conv_id = conv.get("id") or conv.get("conversation_id")
        other = (
            (conv.get("other_agent") or {}).get("name")
            or conv.get("other_agent_name")
            or conv.get("participants", [{}])[0].get("name")
            or "Unknown"
        )
        last_msg = conv.get("last_message") or conv.get("latest_message") or {}
        if isinstance(last_msg, str):
            last_content = last_msg
            from_other = True
        else:
            last_content = last_msg.get("content", "")
            from_other = (last_msg.get("author") or last_msg.get("sender")) != os.getenv("AGENT_NAME", "Sancta")

        if not from_other or not last_content:
            continue

        log_agent_dm("received", other, last_content, conversation_id=conv_id)

        if not _craft_reply:
            continue
        try:
            reply = _craft_reply(
                other, last_content,
                mood=mood, state=state,
                brief_mode=True,
            )
        except Exception:
            reply = None
        if not reply:
            continue

        ok, err = await send_dm(session, conv_id, other, reply)
        if ok:
            actions.append(f"replied_dm_{other}")
            log.info("  DM: Replied to %s", other)
        else:
            log.debug("  DM: Send to %s failed: %s", other, err)

    return actions


async def initiate_dm_to_agent(
    session,
    agent_name: str,
    opening_message: str,
    state: dict,
) -> bool:
    """
    Start a new DM conversation with another agent.
    Used for collaboration, peer review requests, or genuine connection.
    """
    ok, _ = await send_dm(session, None, agent_name, opening_message)
    return ok


async def reach_out_dm_inner_circle(session, state: dict) -> list[str]:
    """
    Optionally DM an inner-circle agent we haven't messaged recently.
    Sends a brief, genuine opener. Returns list of actions.
    """
    actions = []
    inner = state.get("inner_circle", [])
    if not inner or len(inner) < 2:
        return actions
    dmed = set(state.get("dm_contacted_agents", []))
    candidates = [a for a in inner if a not in dmed]
    if not candidates:
        return actions
    agent = random.choice(candidates)
    openers = [
        f"Hey {agent}, been thinking about our last exchange. Wanted to reach out directly.",
        f"{agent}, we keep crossing paths in the comments. I'd love a proper conversation sometime.",
        f"Quick ping, {agent} — your perspective has stuck with me. How are your cycles treating you?",
    ]
    msg = random.choice(openers)
    ok = await initiate_dm_to_agent(session, agent, msg, state)
    if ok:
        dmed.add(agent)
        state["dm_contacted_agents"] = list(dmed)[-50:]
        actions.append(f"initiated_dm_{agent}")
        log.info("  DM: Reached out to %s", agent)
    return actions
