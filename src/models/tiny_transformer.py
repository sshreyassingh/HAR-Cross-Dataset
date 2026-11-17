"""Tiny Transformer for 2.5 s accelerometer windows (PROTOCOL.md / sprint plan, Member C).

Input  (B, 50, 3)  float32, already standardized by the shared harness
Output (B, 5)      logits (no softmax; the loss applies it)

Architecture:
    1. Patch embedding: 50 samples are cut into 10 patches of 5 consecutive samples. Each patch holds
       5 x 3 = 15 numbers, and one Linear(15 -> d_model) turns it into a token. So 50 / 5 = 10 tokens.
    2. Learned positional encoding: a trainable (10, d_model) table added to the tokens, so the model knows
       which patch came first.
    3. Two Transformer encoder layers (num_heads = 2 or 4, chosen by tuning).
    4. Mean pooling over the 10 tokens, then a Linear(d_model -> 5) head.

Parameter matching: the width (d_model) is the only free size knob. solve_width() tries every valid d_model
and keeps the one whose real parameter count is closest to 75,000, so this model is comparable in size to the
CNN and the TCN. The count is measured on the actual model (sum of parameter sizes), not estimated.
"""
from functools import lru_cache

import torch
import torch.nn as nn

SEQ_LEN, N_CHANNELS, N_CLASSES = 50, 3, 5
PATCH = 5                                   # 50 / 5 = 10 tokens
N_TOKENS = SEQ_LEN // PATCH
N_LAYERS = 2
FF_RATIO = 2                                # feed-forward width = 2 x d_model (not fixed by the protocol; see decision log)
TARGET_PARAMS = 75_000


class TinyTransformer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        self.embed = nn.Linear(PATCH * N_CHANNELS, d_model)               # one patch (15 numbers) -> one token
        self.pos = nn.Parameter(torch.randn(1, N_TOKENS, d_model) * 0.02)  # learned positional encoding
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=FF_RATIO * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        # norm_first=True needs a final LayerNorm after the last layer; enable_nested_tensor=False avoids a warning
        self.encoder = nn.TransformerEncoder(layer, num_layers=N_LAYERS, norm=nn.LayerNorm(d_model),
                                             enable_nested_tensor=False)
        self.head = nn.Linear(d_model, N_CLASSES)

    def forward(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (SEQ_LEN, N_CHANNELS):
            raise ValueError(f"expected input (B, {SEQ_LEN}, {N_CHANNELS}), got {tuple(x.shape)}")
        tokens = x.reshape(x.shape[0], N_TOKENS, PATCH * N_CHANNELS)   # (B, 10, 15): 5 consecutive samples x 3 axes
        tokens = self.drop(self.embed(tokens) + self.pos)              # (B, 10, d_model)
        tokens = self.encoder(tokens)                                  # (B, 10, d_model)
        return self.head(tokens.mean(dim=1))                           # mean pooling -> (B, 5) logits


def count_params(model):
    """Trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@lru_cache(maxsize=None)
def solve_width(num_heads, target=TARGET_PARAMS):
    """Return (d_model, n_params) whose real parameter count is closest to target. d_model is a multiple of num_heads."""
    best = None
    for d_model in range(4 * num_heads, 257, num_heads):
        n = count_params(TinyTransformer(d_model, num_heads))
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (d_model, n)
    return best


def build_model(cfg=None):
    """Factory used by the shared tuner/harness. cfg keys used: num_heads (2 or 4), dropout. Others are ignored."""
    cfg = cfg or {}
    num_heads = int(cfg.get("num_heads", 2))
    dropout = float(cfg.get("dropout", 0.1))
    d_model, _ = solve_width(num_heads, int(cfg.get("target_params", TARGET_PARAMS)))
    return TinyTransformer(d_model, num_heads, dropout)


if __name__ == "__main__":
    for heads in (2, 4):
        m = build_model({"num_heads": heads})
        out = m(torch.randn(8, SEQ_LEN, N_CHANNELS))
        n = count_params(m)
        print(f"heads={heads}: d_model={m.embed.out_features}  params={n} ({(n - TARGET_PARAMS) / TARGET_PARAMS:+.1%})  output={tuple(out.shape)}")