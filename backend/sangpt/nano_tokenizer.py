"""
Tokenizer wrapper for Sancta's nanoGPT engine.

Priority order:
  1. tiktoken (GPT-2 BPE) — fast, standard, 50257 vocab
  2. Character-level fallback — zero dependencies, works anywhere

The wrapper exposes a uniform interface regardless of backend so the
engine never has to care which tokenizer is active.
"""

import logging
from typing import List, Optional

logger = logging.getLogger("sancta_gpt.tokenizer")

# ─── Try tiktoken first ──────────────────────────────────────────────────────

_tiktoken_enc = None
_HAS_TIKTOKEN = False

try:
    import tiktoken
    _tiktoken_enc = tiktoken.get_encoding("gpt2")
    _HAS_TIKTOKEN = True
    logger.info("Tokenizer backend: tiktoken (GPT-2 BPE, vocab_size=50257)")
except ImportError:
    logger.info("tiktoken not installed — falling back to character-level tokenizer")
except Exception as exc:
    logger.warning("tiktoken init failed (%s) — falling back to character-level", exc)


# ─── Tokenizer interface ─────────────────────────────────────────────────────

class Tokenizer:
    """
    Uniform tokenizer interface for the nanoGPT engine.

    Supports two backends:
      - tiktoken GPT-2 BPE (50257 tokens, subword)
      - character-level (dynamic vocab built from corpus)
    """

    def __init__(self, backend: str = "auto"):
        """
        Args:
            backend: "tiktoken", "char", or "auto" (tiktoken if available, else char)
        """
        if backend == "auto":
            backend = "tiktoken" if _HAS_TIKTOKEN else "char"

        self.backend = backend

        if self.backend == "tiktoken":
            if not _HAS_TIKTOKEN:
                raise RuntimeError("tiktoken requested but not available")
            self._enc = _tiktoken_enc
            self._vocab_size = self._enc.n_vocab  # 50257
            self._eot_token = self._enc.eot_token  # <|endoftext|> = 50256
        else:
            # Character-level: will be built from corpus
            self._char2id = {}
            self._id2char = {}
            self._vocab_size = 0
            self._eot_token = 0  # will be set when vocab is built

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eot_token(self) -> int:
        return self._eot_token

    def build_vocab_from_corpus(self, texts: List[str]) -> int:
        """
        Build vocabulary from corpus. Only meaningful for char backend;
        tiktoken already has a fixed vocabulary.

        Returns the vocab size.
        """
        if self.backend == "tiktoken":
            return self._vocab_size

        all_chars = set()
        for text in texts:
            all_chars.update(text)
        # Sort for deterministic ordering
        chars = sorted(all_chars)
        self._char2id = {ch: i for i, ch in enumerate(chars)}
        self._id2char = {i: ch for i, ch in enumerate(chars)}
        # EOT token is the last index
        self._eot_token = len(chars)
        self._vocab_size = len(chars) + 1  # +1 for EOT
        logger.debug("Char tokenizer built: %d unique chars, vocab_size=%d",
                      len(chars), self._vocab_size)
        return self._vocab_size

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if self.backend == "tiktoken":
            return self._enc.encode(text, allowed_special={"<|endoftext|>"})
        # Character-level
        return [self._char2id.get(ch, self._eot_token) for ch in text]

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to text."""
        if self.backend == "tiktoken":
            return self._enc.decode(tokens)
        # Character-level
        return "".join(self._id2char.get(t, "") for t in tokens if t != self._eot_token)

    def encode_document(self, text: str) -> List[int]:
        """Encode a full document with EOT delimiter."""
        tokens = self.encode(text)
        tokens.append(self._eot_token)
        return tokens

    def encode_corpus(self, documents: List[str]) -> List[int]:
        """Encode an entire corpus into a flat token stream with EOT delimiters."""
        all_tokens = []
        for doc in documents:
            if not doc or not doc.strip():
                continue
            all_tokens.extend(self.encode(doc.strip()))
            all_tokens.append(self._eot_token)
        return all_tokens

    def status(self) -> dict:
        """Return tokenizer status info."""
        return {
            "backend": self.backend,
            "vocab_size": self._vocab_size,
            "eot_token": self._eot_token,
            "has_tiktoken": _HAS_TIKTOKEN,
        }


def get_tokenizer(backend: str = "auto") -> Tokenizer:
    """Factory: create a tokenizer with the best available backend."""
    return Tokenizer(backend=backend)


def has_tiktoken() -> bool:
    """Check if tiktoken is available."""
    return _HAS_TIKTOKEN
