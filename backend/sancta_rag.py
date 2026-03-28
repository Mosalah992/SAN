"""
sancta_rag.py — Query-conditioned retrieval for Ollama / local LLM prompts.

Indexes prose chunks from knowledge/*.txt and *.md (excludes huge word lists).

Scoring (default): **TF-IDF** over word counts — no sklearn/torch. Set
``SANCTA_RAG_LEXICAL=true`` for legacy substring token overlap only.

Environment:
  SANCTA_RAG=0|false|no     — disable retrieval
  SANCTA_RAG_TOP_K=5        — max chunks (default 5; use 3 for minimal prompts)
  SANCTA_RAG_MAX_TOTAL=8000 — cap on formatted context characters
  SANCTA_RAG_SKIP_FILES     — comma-separated basenames to skip
  SANCTA_RAG_LEXICAL=true   — use old overlap count instead of TF-IDF
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent
_ROOT = _BACKEND_DIR.parent
_KNOWLEDGE_DIR = _ROOT / "knowledge"

_DEFAULT_SKIP_FILES = frozenset(
    {
        "words_alpha.txt",
    }
)

_chunk_rows: list[tuple[str, str]] | None = None
_chunk_signature: float = -1.0


def _skip_files() -> set[str]:
    """Basenames only, all lowercased; ``_scan_corpus`` compares ``f.name.lower()`` to this set."""
    out = set(x.lower() for x in _DEFAULT_SKIP_FILES)
    extra = (os.environ.get("SANCTA_RAG_SKIP_FILES") or "").strip()
    for part in extra.split(","):
        p = part.strip().lower()
        if p:
            out.add(p)
    return out


def _split_chunks(text: str, max_chunk: int = 900, overlap: int = 120) -> list[str]:
    text = (text or "").strip()
    if len(text) < 50:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if len(p.strip()) > 45]
    if not paragraphs:
        paragraphs = [text] if len(text) > 80 else []
    chunks: list[str] = []
    for p in paragraphs:
        if len(p) <= max_chunk:
            chunks.append(p)
            continue
        step = max(200, max_chunk - overlap)
        for i in range(0, len(p), step):
            piece = p[i : i + max_chunk]
            if len(piece) > 50:
                chunks.append(piece)
    return chunks


def _scan_corpus() -> list[tuple[str, str]]:
    """Return [(source_basename, chunk_text), ...]."""
    global _chunk_rows, _chunk_signature
    if not _KNOWLEDGE_DIR.is_dir():
        return []

    files = [f for f in _KNOWLEDGE_DIR.iterdir() if f.is_file() and f.suffix.lower() in (".txt", ".md")]
    skip = _skip_files()
    sig = 0.0
    for f in files:
        if f.name.lower() in skip:
            continue
        try:
            sig += f.stat().st_mtime + f.stat().st_size
        except OSError:
            continue

    if _chunk_rows is not None and sig == _chunk_signature:
        return _chunk_rows

    rows: list[tuple[str, str]] = []
    for f in sorted(files, key=lambda x: x.name.lower()):
        if f.name.lower() in skip:
            continue
        try:
            raw = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for ch in _split_chunks(raw):
            rows.append((f.name, ch))

    _chunk_rows = rows
    _chunk_signature = sig
    return rows


def clear_rag_index_cache() -> None:
    global _chunk_rows, _chunk_signature
    _chunk_rows = None
    _chunk_signature = -1.0


def _query_tokens(query: str) -> set[str]:
    q = re.sub(r"[^\w\s]", " ", (query or "").lower())
    return {w for w in q.split() if len(w) > 2}


def _word_counts(text: str) -> dict[str, int]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    wc: dict[str, int] = {}
    for w in words:
        if len(w) > 2:
            wc[w] = wc.get(w, 0) + 1
    return wc


def _retrieve_lexical(
    query: str,
    corpus: list[tuple[str, str]],
    top_k: int,
    max_chars_per_chunk: int,
) -> list[tuple[str, str, float]]:
    toks = _query_tokens(query)
    if not toks:
        return []
    scored: list[tuple[str, str, float]] = []
    for fname, chunk in corpus:
        low = chunk.lower()
        score = float(sum(1 for t in toks if t in low))
        if score <= 0:
            continue
        excerpt = chunk if len(chunk) <= max_chars_per_chunk else chunk[:max_chars_per_chunk] + "..."
        scored.append((fname, excerpt, score))
    scored.sort(key=lambda x: (-x[2], -len(x[1])))
    return _dedupe_top_k(scored, top_k)


def _dedupe_top_k(
    scored: list[tuple[str, str, float]],
    top_k: int,
) -> list[tuple[str, str, float]]:
    out: list[tuple[str, str, float]] = []
    seen_keys: set[tuple[str, str]] = set()
    for fname, excerpt, sc in scored:
        key = (fname, excerpt[:120])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.append((fname, excerpt, sc))
        if len(out) >= top_k:
            break
    return out


def _retrieve_tfidf(
    query: str,
    corpus: list[tuple[str, str]],
    top_k: int,
    max_chars_per_chunk: int,
) -> list[tuple[str, str, float]]:
    q_terms = _query_tokens(query)
    if not q_terms:
        return []

    rows: list[tuple[str, str, dict[str, int]]] = [
        (fname, chunk, _word_counts(chunk)) for fname, chunk in corpus
    ]
    n_docs = len(rows)
    if n_docs == 0:
        return []

    df: dict[str, int] = {}
    for _, _, wc in rows:
        for w in wc:
            df[w] = df.get(w, 0) + 1

    scored: list[tuple[str, str, float]] = []
    for fname, chunk, wc in rows:
        sc = 0.0
        for t in q_terms:
            if t not in wc:
                continue
            tf = 1.0 + math.log(wc[t])
            idf = math.log((n_docs + 1) / (df.get(t, 0) + 1)) + 1.0
            sc += tf * idf
        if sc <= 0:
            continue
        excerpt = chunk if len(chunk) <= max_chars_per_chunk else chunk[:max_chars_per_chunk] + "..."
        scored.append((fname, excerpt, sc))

    scored.sort(key=lambda x: (-x[2], -len(x[1])))
    return _dedupe_top_k(scored, top_k)


def retrieve(
    query: str,
    top_k: int = 5,
    max_chars_per_chunk: int = 1400,
) -> list[tuple[str, str, float]]:
    """
    Return up to top_k matches: (source_filename, excerpt, score).
    Default scoring: TF-IDF. Score is a relative weight (higher = better match).
    """
    if (os.environ.get("SANCTA_RAG") or "").strip().lower() in ("0", "false", "no", "off"):
        return []

    try:
        top_k = max(1, min(24, int(os.environ.get("SANCTA_RAG_TOP_K", str(top_k)))))
    except ValueError:
        top_k = 5

    corpus = _scan_corpus()
    if not corpus:
        return []

    use_lexical = os.environ.get("SANCTA_RAG_LEXICAL", "").strip().lower() in ("1", "true", "yes", "on")
    if use_lexical:
        return _retrieve_lexical(query, corpus, top_k, max_chars_per_chunk)
    return _retrieve_tfidf(query, corpus, top_k, max_chars_per_chunk)


def format_rag_context(
    query: str,
    top_k: int = 5,
    max_total_chars: int | None = None,
) -> str:
    """
    Block for system prompt injection. Uses top_k ranked chunks (TF-IDF by default).
    """
    if max_total_chars is None:
        try:
            max_total_chars = max(2000, int(os.environ.get("SANCTA_RAG_MAX_TOTAL", "8000")))
        except ValueError:
            max_total_chars = 8000

    hits = retrieve(query, top_k=top_k)
    if not hits:
        return ""

    lines = ["=== RETRIEVED KNOWLEDGE (RAG: cite only from excerpts; do not invent facts) ==="]
    total = 0
    for fname, snippet, sc in hits:
        block = f"\n--- {fname} (score={sc:.4f}) ---\n{snippet}\n"
        if total + len(block) > max_total_chars:
            remain = max_total_chars - total - 100
            if remain > 200:
                lines.append(f"\n--- {fname} (truncated) ---\n{snippet[:remain]}...\n")
            break
        lines.append(block)
        total += len(block)
    lines.append("=== END RETRIEVED KNOWLEDGE ===")
    return "\n".join(lines)


def format_rag_inline(
    query: str,
    top_k: int = 3,
    max_chars_per_chunk: int = 320,
    max_joined: int = 900,
) -> str:
    """
    Compact 'KB line' for tight prompts: chunk1 | chunk2 | chunk3
    """
    hits = retrieve(query, top_k=top_k, max_chars_per_chunk=max_chars_per_chunk)
    if not hits:
        return ""
    parts: list[str] = []
    for fname, snip, _ in hits:
        one = snip.replace("\n", " ").strip()
        if len(one) > max_chars_per_chunk:
            one = one[: max_chars_per_chunk - 3] + "..."
        parts.append(f"{fname}: {one}")
    return " | ".join(parts)[:max_joined]
