"""
sancta_semantic.py — Semantic Representation for Knowledge (Phase 1)

Implements the SANCTA Learning Architecture Phase 1:
  1.1 Embedding Layer — sentence-transformers (MiniLM)
  1.2 Concept Extraction — KeyBERT (fallback: YAKE, then legacy scoring)
  1.3 Deduplication — cosine similarity clustering
  1.4 Concept Graph — nodes (concepts) + edges (co-occurrence / similarity)

Graceful fallback: if sentence-transformers or keybert are not installed,
the module returns None from semantic functions and callers use legacy path.
"""

from __future__ import annotations

import logging
import math
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

log = logging.getLogger("soul.semantic")

# ── Optional imports (graceful degradation) ─────────────────────────────────

_EMBEDDING_MODEL = None
_KEYBERT_AVAILABLE = False
_SENTENCE_TRANSFORMERS_AVAILABLE = False


def _ensure_embedding_model() -> bool:
    """RAG/embedding disabled — always use legacy path (no sentence-transformers)."""
    return False


def _ensure_keybert():
    """Check if KeyBERT is available."""
    global _KEYBERT_AVAILABLE
    if _KEYBERT_AVAILABLE:
        return True
    try:
        import keybert  # noqa: F401
        _KEYBERT_AVAILABLE = True
        return True
    except ImportError:
        return False


def embed_texts(texts: list[str]) -> list[list[float]] | None:
    """
    Embed a list of texts into vectors. Returns None if model unavailable.
    """
    if not _ensure_embedding_model():
        return None
    try:
        embeddings = _EMBEDDING_MODEL.encode(texts, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]
    except Exception as e:
        log.warning("Semantic: embed_texts failed: %s", e)
        return None


def embed_single(text: str) -> list[float] | None:
    """Embed a single text. Returns None if model unavailable."""
    result = embed_texts([text])
    return result[0] if result else None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def extract_key_concepts_semantic(
    text: str,
    top_n: int = 10,
    use_keybert: bool = True,
) -> list[str]:
    """
    Extract key concepts using KeyBERT (semantic) or fallback to YAKE/legacy.
    Returns list of concept strings.
    """
    if use_keybert and _ensure_keybert() and _ensure_embedding_model():
        try:
            from keybert import KeyBERT
            kw_model = KeyBERT(model=_EMBEDDING_MODEL)
            # KeyBERT requires nr_candidates > top_n. Short texts yield few candidates.
            # Fix: top_n = min(5, candidates) or increase candidates (user guidance).
            kw_top_n = min(5, top_n * 2)  # safe cap for short texts; increase candidates below
            nr_candidates = max(20, kw_top_n * 3)  # ensure candidates always exceed keywords
            keywords = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 3),
                stop_words="english",
                top_n=kw_top_n,
                nr_candidates=nr_candidates,
                use_maxsum=True,
            )
            concepts = [kw for kw, _ in keywords if kw and len(kw.strip()) >= 8]
            return concepts[:top_n]
        except Exception as e:
            log.warning("Semantic: KeyBERT extraction failed: %s", e)

    # Fallback: YAKE (statistical, no neural deps)
    try:
        import yake
        kw = yake.KeywordExtractor(n=3, dedupLim=0.9, top=top_n * 2)
        kws = kw.extract_keywords(text)
        concepts = [kw for kw, _ in kws if kw and len(kw.strip()) >= 8]
        return concepts[:top_n]
    except ImportError:
        pass
    except Exception as e:
        log.debug("Semantic: YAKE fallback failed: %s", e)

    # Fallback: sentence-based semantic selection (embed sentences, pick by centrality)
    if _ensure_embedding_model():
        return _extract_concepts_from_sentences(text, top_n)
    return []


def _extract_concepts_from_sentences(text: str, top_n: int) -> list[str]:
    """Extract key sentences by embedding centrality (no KeyBERT/YAKE)."""
    import re
    raw = re.split(r"[.!?]\s+", text.replace("\n", " "))
    sentences = [s.strip() for s in raw if len(s.strip()) > 25]
    if not sentences:
        return []
    embeddings = embed_texts(sentences)
    if not embeddings or len(embeddings) != len(sentences):
        return []
    # Centroid = mean of all sentence embeddings
    dim = len(embeddings[0])
    centroid = [sum(e[i] for e in embeddings) / len(embeddings) for i in range(dim)]
    # Score by similarity to centroid
    scored = [(s, cosine_similarity(e, centroid)) for s, e in zip(sentences, embeddings)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored[:top_n] if _quality_filter(s)]


def _quality_filter(text: str, min_len: int = 8, min_alpha: float = 0.3) -> bool:
    if not text or len(text.strip()) < min_len:
        return False
    alpha = sum(1 for c in text if c.isalpha())
    if alpha / max(len(text), 1) < min_alpha:
        return False
    return True


def deduplicate_by_similarity(
    concepts: list[str],
    embeddings: list[list[float]] | None = None,
    threshold: float = 0.85,
) -> list[str]:
    """
    Merge semantically similar concepts. Keeps first of each cluster.
    If embeddings is None, computes them (or returns concepts unchanged if model unavailable).
    """
    if not concepts:
        return []

    if embeddings is None:
        embeddings = embed_texts(concepts)
    if embeddings is None or len(embeddings) != len(concepts):
        return list(dict.fromkeys(concepts))  # exact dedup only

    kept: list[str] = []
    kept_embeddings: list[list[float]] = []

    for i, (concept, emb) in enumerate(zip(concepts, embeddings)):
        is_duplicate = False
        for j, kept_emb in enumerate(kept_embeddings):
            if cosine_similarity(emb, kept_emb) >= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(concept)
            kept_embeddings.append(emb)

    return kept


def build_concept_graph(
    concepts: list[str],
    embeddings: list[list[float]] | None = None,
    similarity_threshold: float = 0.6,
) -> dict[str, list[str]]:
    """
    Build a concept graph: each concept maps to list of similar concepts (edges).
    Returns {concept: [similar_concept, ...]}.
    """
    if not concepts:
        return {}

    if embeddings is None:
        embeddings = embed_texts(concepts)
    if embeddings is None or len(embeddings) != len(concepts):
        return {c: [] for c in concepts}

    graph: dict[str, list[str]] = {c: [] for c in concepts}

    for i, (ci, emb_i) in enumerate(zip(concepts, embeddings)):
        for j, (cj, emb_j) in enumerate(zip(concepts, embeddings)):
            if i >= j:
                continue
            sim = cosine_similarity(emb_i, emb_j)
            if sim >= similarity_threshold:
                graph[ci].append(cj)
                graph[cj].append(ci)

    return graph


def extract_and_deduplicate_concepts(
    text: str,
    top_n: int = 10,
    similarity_threshold: float = 0.85,
) -> tuple[list[str], dict[str, list[str]] | None]:
    """
    Full pipeline: extract concepts (KeyBERT/YAKE) → deduplicate → optional graph.
    Returns (concepts, concept_graph or None).
    """
    concepts_raw = extract_key_concepts_semantic(text, top_n=top_n * 2)
    if not concepts_raw:
        return [], None

    embeddings = embed_texts(concepts_raw)
    concepts = deduplicate_by_similarity(
        concepts_raw, embeddings, threshold=similarity_threshold
    )[:top_n]

    graph = None
    if embeddings and len(embeddings) == len(concepts_raw):
        # Build graph on deduplicated concepts' embeddings
        kept_indices = []
        seen_embeddings: list[list[float]] = []
        for i, (c, emb) in enumerate(zip(concepts_raw, embeddings)):
            is_dup = any(
                cosine_similarity(emb, se) >= similarity_threshold
                for se in seen_embeddings
            )
            if not is_dup:
                kept_indices.append((i, c, emb))
                seen_embeddings.append(emb)
        if kept_indices:
            kept_concepts = [c for _, c, _ in kept_indices]
            kept_embs = [e for _, _, e in kept_indices]
            graph = build_concept_graph(
                kept_concepts, kept_embs, similarity_threshold=0.5
            )

    return concepts, graph


def is_semantic_available() -> bool:
    """Return True if semantic extraction is available (sentence-transformers loaded)."""
    return _ensure_embedding_model()
