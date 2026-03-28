"""Smoke tests for SanctaGPT corpus + training (fast, CPU-only)."""
from __future__ import annotations

import os

import pytest

import sancta_gpt

slow = pytest.mark.skipif(
    os.environ.get("SANCTA_GPT_SLOW_TESTS", "") != "1",
    reason="set SANCTA_GPT_SLOW_TESTS=1 to run (each train_step is slow in pure Python)",
)


def test_conversational_seeds_in_corpus():
    eng = sancta_gpt.SanctaGPT()
    n = eng.build_corpus()
    assert n > 50
    # Chunks are windowed; dialogue markers may not appear in the first N joined strings
    assert any("Operator:" in d or "Sancta:" in d for d in eng._docs)
    assert any("Thanks for checking" in d or "plain English" in d for d in eng._docs)


@slow
def test_train_step_finite_loss():
    eng = sancta_gpt.SanctaGPT()
    eng.build_corpus()
    eng._init_model()
    losses = [eng.train_step() for _ in range(2)]
    assert all(l < float("inf") for l in losses)


@slow
def test_generate_after_min_steps():
    eng = sancta_gpt.SanctaGPT()
    eng.build_corpus()
    eng._init_model()
    eng.train(num_steps=4, log_every=0)
    t = eng.generate("Hi ", max_tokens=20, temperature=0.9)
    assert isinstance(t, str)


def test_add_document():
    eng = sancta_gpt.SanctaGPT()
    eng.build_corpus()
    before = eng._corpus_size
    assert eng.add_document("This is a test document for the corpus. " * 3)
    assert eng._corpus_size >= before
