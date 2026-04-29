# src/model/attention.py
"""
Memory-efficient multi-head self-attention for ARC-Lite.
No flash-attn dependency — uses PyTorch's scaled_dot_product_attention
which is built into torch >= 2.0 and handles causal masking efficiently.

Causal masking is used because we train the model autoregressively:
predict the next token given all previous tokens. This is how the MDL
principle is applied — the model must compress the entire sequence
(both input and output grid) to predict each next token.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with causal mask.

    Uses torch.nn.functional.scaled_dot_product_attention (SDPA),
    available in PyTorch >= 2.0. On CUDA this dispatches to an optimised
    kernel (similar to flash attention) without requiring the flash-attn
    package.

    Args:
        d_model  : embedding dimension
        n_heads  : number of attention heads
        dropout  : attention dropout probability
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.dropout  = dropout

        # Fused QKV projection: one matrix instead of three saves memory
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x              : (B, T, d_model)
            attention_mask : (B, T) bool tensor — True for real tokens,
                             False for padding. Will be combined with
                             the causal mask inside SDPA.
        Returns:
            out : (B, T, d_model)
        """
        B, T, _ = x.shape

        # 1. Compute Q, K, V via fused projection
        qkv = self.qkv_proj(x)                          # (B, T, 3*d_model)
        q, k, v = qkv.split(self.d_model, dim=-1)       # each (B, T, d_model)

        # 2. Reshape to (B, n_heads, T, d_head)
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # 3. Build attention bias from padding mask
        # SDPA expects attn_mask of shape (B, 1, T, T) or (T, T)
        # where True means "attend" and False means "ignore"
        attn_bias = None
        if attention_mask is not None:
            # (B, T) -> (B, 1, 1, T): broadcast over heads and query positions
            key_mask = attention_mask.unsqueeze(1).unsqueeze(2)   # (B, 1, 1, T)
            # Convert bool to float mask: 0.0 = attend, -inf = ignore
            attn_bias = torch.zeros(B, 1, T, T, device=x.device, dtype=x.dtype)
            attn_bias = attn_bias.masked_fill(~key_mask, float("-inf"))

        # 4. Scaled dot-product attention with causal mask
        # is_causal=True applies the causal (lower-triangular) mask automatically.
        # If we also have a padding attn_bias, we pass it; SDPA adds them.
        # Note: when passing attn_mask, is_causal must be False and we
        # build the causal part manually.
        if attn_bias is not None:
            # Build causal mask manually and merge with padding mask
            causal = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype),
                diagonal=1,
            )                                          # (T, T)
            attn_bias = attn_bias + causal             # broadcast: (B, 1, T, T)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_bias,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )

        # 5. Merge heads: (B, n_heads, T, d_head) -> (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        return self.resid_dropout(self.out_proj(out))


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    d_model, n_heads, T, B = 256, 8, 64, 4
    attn = MultiHeadSelfAttention(d_model, n_heads, dropout=0.1).cuda()

    x    = torch.randn(B, T, d_model).cuda()
    mask = torch.ones(B, T, dtype=torch.bool).cuda()
    mask[0, 50:] = False  # simulate padding in first sequence

    out = attn(x, mask)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == (B, T, d_model)

    total = sum(p.numel() for p in attn.parameters())
    print(f"Attention params: {total/1e6:.3f}M")
    print("MultiHeadSelfAttention OK.")