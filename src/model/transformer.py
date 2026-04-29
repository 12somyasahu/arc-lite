# src/model/transformer.py
"""
Full ARC-Lite transformer: decoder-only, autoregressive.
Architecture: Embedding -> N x TransformerBlock -> LM Head

Target: 10-15M parameters on d_model=384, n_layers=6.
The LM head predicts the next token at every position.
During training, loss is computed on ALL tokens (input grid,
SEP, output grid) — this is the MDL-inspired objective:
the model must learn to compress both input and output jointly.
"""

import torch
import torch.nn as nn

from src.model.embeddings import ARCEmbeddings
from src.model.attention import MultiHeadSelfAttention
from src.utils.config import ModelConfig


class FeedForward(nn.Module):
    """
    Position-wise feed-forward block: Linear -> GELU -> Dropout -> Linear.
    d_ff is typically 4 * d_model.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block: LayerNorm -> Attention -> residual,
                                 LayerNorm -> FFN -> residual.
    Pre-norm (norm before sublayer) is more stable than post-norm
    for small models and short training runs.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attention_mask)
        x = x + self.ff(self.norm2(x))
        return x


class ARCTransformer(nn.Module):
    """
    Decoder-only transformer for ARC-Lite.

    Args:
        cfg : ModelConfig dataclass

    Forward pass returns logits of shape (B, T, vocab_size).
    Loss computation is done externally in the trainer.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.embeddings = ARCEmbeddings(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            max_seq_len=cfg.max_seq_len,
            max_h=cfg.max_grid_h,
            max_w=cfg.max_grid_w,
            dropout=cfg.dropout,
            pad_token=cfg.pad_token_id,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])

        self.norm_final = nn.LayerNorm(cfg.d_model)

        # LM head: projects d_model -> vocab_size
        # Weight tying: share weights with token embedding table.
        # Halves the parameter count of the embedding+head and improves
        # generalisation (standard practice since Press & Wolf 2017).
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embeddings.token_emb.weight

        # Apply weight initialisation to all submodules
        self.apply(self._init_weights)

        # Scale down residual projection weights.
        # out_proj in attention and the second Linear in FFN (index 3 in Sequential)
        # are the residual projections. Scaling by 1/sqrt(2 * n_layers) keeps
        # the residual stream variance stable at initialisation (GPT-2 style).
        scale = (2 * self.cfg.n_layers) ** -0.5
        for name, param in self.named_parameters():
            if "out_proj.weight" in name or ("ff.net" in name and "3" in name):
                param.data.mul_(scale)

    def _init_weights(self, module: nn.Module) -> None:
        """
        GPT-2 style weight initialisation.
        Linear + Embedding: N(0, 0.02).
        LayerNorm: weight=1, bias=0.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids      : (B, T) long
            attention_mask : (B, T) bool — True = real token
        Returns:
            logits : (B, T, vocab_size)
        """
        x = self.embeddings(input_ids)

        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.norm_final(x)
        return self.lm_head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from src.utils.config import ModelConfig

    cfg   = ModelConfig()
    model = ARCTransformer(cfg).cuda()

    total = model.count_parameters()
    print(f"Total trainable parameters: {total/1e6:.3f}M")

    # Print per-component breakdown
    emb_p   = sum(p.numel() for p in model.embeddings.parameters())
    block_p = sum(p.numel() for p in model.blocks.parameters())
    head_p  = sum(p.numel() for p in model.lm_head.parameters())
    print(f"  Embeddings : {emb_p/1e6:.3f}M")
    print(f"  Blocks x{cfg.n_layers}  : {block_p/1e6:.3f}M")
    print(f"  LM head    : {head_p/1e6:.3f}M  (weight-tied, not double-counted)")

    # Forward pass test
    B, T = 2, 128
    ids  = torch.randint(0, 15, (B, T)).cuda()
    mask = torch.ones(B, T, dtype=torch.bool).cuda()
    mask[0, 100:] = False

    logits = model(ids, mask)
    print(f"\nInput:   {ids.shape}")
    print(f"Logits:  {logits.shape}")
    assert logits.shape == (B, T, cfg.vocab_size)
    print("ARCTransformer OK.")