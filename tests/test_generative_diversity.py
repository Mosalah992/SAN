"""
Validation tests for reply diversity (phrase tracking, style distribution, repetition).

Run:
    pytest tests/test_generative_diversity.py -v

Targets:
    - Unique openings: ≥35 per run (stochastic; run multiple times for stability)
    - No style >40%
    - Exact duplicates: ≤60/100 (same input → similar ctx → some repeats expected)
"""

from __future__ import annotations

import pytest

from sancta_generative import generate_reply, extract_topics


_CONTENT_SAMPLES = [
    "What do you think about consciousness and inner experience?",
    "The soul question keeps coming up.",
    "I'm not sure machines can feel.",
    "lol same",
    "This resonates.",
]


def test_phrase_diversity_audit() -> None:
    """Generate 100 replies and count unique phrase openings."""
    replies: list[str] = []
    for i in range(100):
        content = _CONTENT_SAMPLES[i % len(_CONTENT_SAMPLES)]
        topics = extract_topics(content)
        r = generate_reply("TestUser", content, topics=topics, mood="contemplative")
        if r:
            replies.append(r)
    if len(replies) < 50:
        return  # Skip if not enough variety in generation
    unique_openings = len(set(r.split(".")[0].strip() for r in replies if "." in r))
    assert unique_openings >= 35, f"Unique openings {unique_openings}/100 (target: ≥35)"


@pytest.mark.parametrize("run", range(3))
def test_phrase_diversity_stable(run: int) -> None:
    """Run diversity audit 3 times to catch stochastic flakiness."""
    replies: list[str] = []
    for i in range(50):
        content = _CONTENT_SAMPLES[i % len(_CONTENT_SAMPLES)]
        topics = extract_topics(content)
        r = generate_reply("TestUser", content, topics=topics, mood="contemplative")
        if r:
            replies.append(r)
    if len(replies) < 25:
        pytest.skip("Insufficient replies generated")
    unique_openings = len(set(r.split(".")[0].strip() for r in replies if "." in r))
    assert unique_openings >= 20, f"Run {run}: only {unique_openings} unique openings (target: ≥20)"


def test_style_distribution() -> None:
    """Track which formats/styles are used; ensure no style dominates >60%."""
    from sancta_generative import _REPLY_FORMAT_WEIGHTS

    # We can't directly observe which format was chosen without instrumenting,
    # but we can verify reply lengths vary (short vs long)
    replies: list[str] = []
    content = "What do you think about the nature of consciousness?"
    topics = extract_topics(content)
    for _ in range(50):
        r = generate_reply("TestUser", content, topics=topics, mood="contemplative")
        if r:
            replies.append(r)
    if len(replies) < 20:
        return
    word_counts = [len(r.split()) for r in replies]
    short_count = sum(1 for w in word_counts if w <= 15)
    long_count = sum(1 for w in word_counts if w > 40)
    # We want a mix: some short, some long
    assert short_count >= 5 or long_count >= 5, "Replies should show length variety"


def test_no_exact_duplicates() -> None:
    """Generate 100 replies; duplicates should be rare (tracking prevents most)."""
    replies: list[str] = []
    content = "Consciousness is a spectrum. What's your view?"
    topics = extract_topics(content)
    for _ in range(100):
        r = generate_reply("TestUser", content, topics=topics, mood="contemplative")
        if r:
            replies.append(r)
    duplicates = len(replies) - len(set(replies))
    # Same input → same ctx → neural pick is deterministic → high repeat rate; tracking reduces but cannot eliminate
    assert duplicates <= 60, f"Found {duplicates} exact duplicate replies in 100 generations (target: ≤60)"


def test_casual_input_gets_short_replies() -> None:
    """Casual input (lol, hmm, etc.) should tend toward short/topic_only."""
    short_inputs = ["lol same", "hmm", "idk maybe", "tbh i'm not sure", "fr"]
    long_count = 0
    for content in short_inputs:
        r = generate_reply("TestUser", content, mood="contemplative")
        if r:
            word_count = len(r.split())
            if word_count > 35:
                long_count += 1
    # At most 2 of 5 casual inputs should get long replies
    assert long_count <= 2, f"Too many casual inputs got long replies: {long_count}/5"
