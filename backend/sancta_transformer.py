"""
sancta_transformer.py — Learnable PyTorch Transformer Encoder

PyTorch implementation of the transformer in sancta_generative.py with
learnable parameters. Same architecture: token embed, positional encoding,
2× TransformerBlock (MHA + FFN), mean pool.

API:
  encode(text) -> list[float]   # context vector, compatible with _neural_pick
  forward(text) -> Tensor       # raw forward pass
  save(path) / load(path)      # checkpointing

When no checkpoint exists, sancta_generative falls back to hash-seeded path.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Hyperparameters (match sancta_generative)
D_MODEL = 32
N_HEADS = 4
D_K = D_MODEL // N_HEADS  # 8
D_FF = 64
N_LAYERS = 2
MAX_SEQ = 64
VOCAB_SIZE = 8192  # hash-based token index
PAD_IDX = 0

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT_PATH = _ROOT / "checkpoints" / "sancta_transformer"


def _tokenize(text: str) -> list[str]:
    """Lightweight word + punctuation tokenizer. Matches sancta_generative._tokenize."""
    tokens = re.findall(r"[a-z']+|[0-9]+|[.,!?;:—–\-]", text.lower())
    return tokens[:MAX_SEQ] if tokens else ["<pad>"]


def _token_to_id(token: str) -> int:
    """Map token to vocab index via hash. Stable and deterministic."""
    h = int(hashlib.md5(token.encode()).hexdigest(), 16)
    return 1 + (h % (VOCAB_SIZE - 1))  # 0 reserved for pad


def _tokens_to_ids(tokens: list[str]) -> list[int]:
    return [_token_to_id(t) for t in tokens]


def _get_trained_model(checkpoint_path: Path | None = None):
    """Lazy-load trained model if checkpoint exists."""
    try:
        import torch as _torch
    except ImportError:
        return None
    path = checkpoint_path or DEFAULT_CHECKPOINT_PATH
    if not path.exists() or not (path / "model.pt").exists():
        return None
    model = SanctaTransformer()
    try:
        state = _torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        model.eval()
        return model
    except Exception:
        return None


class SanctaTransformer(nn.Module):
    """
    Learnable transformer encoder. Architecture matches sancta_generative:
      tokenize -> embed -> pos_enc -> 2× (MHA + LN + FFN + LN) -> mean_pool -> LN
    """

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=PAD_IDX)
        self.pos_embed = _SinusoidalPositionalEncoding(D_MODEL, MAX_SEQ)
        self.blocks = nn.ModuleList([
            _TransformerBlock(D_MODEL, N_HEADS, D_FF)
            for _ in range(N_LAYERS)
        ])
        self.ln = nn.LayerNorm(D_MODEL)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    def forward(self, token_ids: list[int]) -> torch.Tensor:
        """Forward pass. token_ids: list of vocab indices."""
        if not token_ids:
            token_ids = [PAD_IDX]
        ids = torch.tensor([token_ids], dtype=torch.long)
        x = self.embed(ids)
        x = x + self.pos_embed(x)
        for block in self.blocks:
            x = block(x)
        pooled = x.mean(dim=1)
        return self.ln(pooled)

    def encode(self, text: str) -> list[float]:
        """
        Encode text to context vector. Returns list[float] compatible with
        sancta_generative._neural_pick (D_MODEL dim).
        """
        tokens = _tokenize(text)
        ids = _tokens_to_ids(tokens)
        with torch.no_grad():
            out = self.forward(ids)
        return out.squeeze(0).tolist()

    def encode_fragment(self, text: str) -> list[float]:
        """
        Lightweight fragment encoding: mean of token embeddings (no full forward).
        Matches _frag_vec semantics for fragment scoring.
        """
        tokens = _tokenize(text)
        if not tokens:
            return [0.0] * D_MODEL
        ids = _tokens_to_ids(tokens)
        with torch.no_grad():
            emb = self.embed(torch.tensor([ids], dtype=torch.long))
            pooled = emb.mean(dim=1).squeeze(0)
            out = self.ln(pooled)
        return out.tolist()

    def save(self, path: Path | str) -> None:
        """Save model state to checkpoint directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "model.pt")

    def load(self, path: Path | str) -> None:
        """Load model state from checkpoint."""
        path = Path(path)
        state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        self.load_state_dict(state, strict=False)


class _SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding. Not learned."""

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class _TransformerBlock(nn.Module):
    """One encoder block: MHA + residual + LN + FFN + residual + LN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=0.0, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff(x))
        return x
