"""
Helpers that sync Sancta project knowledge and live telemetry into Sangpt DATA.
"""

from __future__ import annotations

import json
from pathlib import Path


def _tail_lines(path: Path, limit: int = 200) -> list[str]:
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="ignore").splitlines()[-limit:]
    except OSError:
        return []


def _write_if_changed(path: Path, text: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        current = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        current = None
    if current == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def sync_project_corpus(project_root: Path, sangpt_root: Path) -> dict[str, int]:
    """
    Materialize live Sancta knowledge and telemetry into the Sangpt dataset tree.
    """
    knowledge_dir = project_root / "knowledge"
    logs_dir = project_root / "logs"
    data_dir = sangpt_root / "DATA"

    knowledge_chunks: list[str] = []
    if knowledge_dir.exists():
        for path in sorted(knowledge_dir.glob("*")):
            if not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").strip()
            except OSError:
                continue
            if len(text) < 60:
                continue
            knowledge_chunks.append(f"[SOURCE: {path.name}]\n{text[:12000]}")

    security_events: list[str] = []
    for log_name in ("security.jsonl", "trust_decisions.jsonl", "cognitive_outcomes.jsonl"):
        for line in _tail_lines(logs_dir / log_name, limit=250):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                security_events.append(json.dumps(obj, ensure_ascii=False))
            except json.JSONDecodeError:
                security_events.append(line.strip())

    operator_memory: list[str] = []
    for line in _tail_lines(logs_dir / "operator_memory.jsonl", limit=120):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        user = str(obj.get("user", "")).strip()
        assistant = str(obj.get("assistant", "")).strip()
        if user and assistant:
            operator_memory.append(f"USER: {user}\nASSISTANT: {assistant}")

    changed = 0
    changed += int(
        _write_if_changed(
            data_dir / "knowledge" / "sancta_project_knowledge.txt",
            "\n\n".join(knowledge_chunks) if knowledge_chunks else "No project knowledge captured yet.",
        )
    )
    changed += int(
        _write_if_changed(
            data_dir / "security" / "sancta_live_security.txt",
            "\n".join(security_events) if security_events else "No live security telemetry captured yet.",
        )
    )
    changed += int(
        _write_if_changed(
            data_dir / "conversational" / "sancta_operator_memory.txt",
            "\n\n".join(operator_memory) if operator_memory else "No operator exchanges captured yet.",
        )
    )

    return {
        "knowledge_docs": len(knowledge_chunks),
        "security_events": len(security_events),
        "operator_turns": len(operator_memory),
        "files_changed": changed,
    }
