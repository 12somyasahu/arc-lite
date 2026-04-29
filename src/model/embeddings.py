# src/model/embeddings.py
"""
Embeddings for ARC-Lite transformer.

Three components combined:
  1. Token embedding       — maps each token ID (0-14) to a d_model vector
  2. 1D positional embed   — standard learned position for sequence position
  3. 2D positional embed   — learned (row, col) embeddings injected inside
                             grid regions, so the model knows spatial layout

The 2D embed is the key design choice: ARC grids have spatial structure
that a pure 1D position encoding destroys. We detect row/col position
from the token sequence structure and add it on top of the 1D position.
"""

import torch
import torch.nn as nn

from src.data.dataset import ROW_TOKEN, SEP_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN


class ARCEmbeddings(nn.Module):
    """
    Combines token + 1D positional + 2D spatial embeddings.

    Args:
        vocab_size  : number of token types (15)
        d_model     : embedding dimension
        max_seq_len : maximum sequence length
        max_h       : maximum grid height (30)
        max_w       : maximum grid width (30)
        dropout     : dropout applied to final embedding
        pad_token   : token ID used for padding (masked out)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int,
        max_h: int = 30,
        max_w: int = 30,
        dropout: float = 0.1,
        pad_token: int = PAD_TOKEN,
    ):
        super().__init__()
        self.d_model    = d_model
        self.pad_token  = pad_token

        # Standard token embedding
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token)

        # 1D learned positional embedding (one entry per sequence position)
        self.pos_emb_1d = nn.Embedding(max_seq_len, d_model)

        # 2D spatial embeddings: separate tables for row and column indices
        # Row 0..max_h-1, Col 0..max_w-1
        # Each contributes d_model/2 dims; they are concatenated then projected
        self.row_emb = nn.Embedding(max_h, d_model // 2)
        self.col_emb = nn.Embedding(max_w, d_model // 2)
        self.spatial_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(d_model)

    def _build_2d_position_ids(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
     B, T = input_ids.shape
     row_ids = torch.zeros(B, T, dtype=torch.long, device=input_ids.device)
     col_ids = torch.zeros(B, T, dtype=torch.long, device=input_ids.device)

     ids_cpu = input_ids.cpu().tolist()

     for b in range(B):
        row, col = 0, 0
        in_grid = False
        for t, tok in enumerate(ids_cpu[b]):
            if tok == BOS_TOKEN:
                in_grid = True
                row, col = 0, 0
            elif tok == SEP_TOKEN:
                row, col = 0, 0
            elif tok == ROW_TOKEN:
                # Assign safe values — these positions are masked to 0 anyway
                row_ids[b, t] = 0
                col_ids[b, t] = 0
                row += 1
                col = 0
            elif tok == EOS_TOKEN or tok == PAD_TOKEN:
                in_grid = False
            elif in_grid and 0 <= tok <= 9:
                row_ids[b, t] = row
                col_ids[b, t] = col
                col += 1

    # Clamp to valid embedding ranges as a safety net
        row_ids.clamp_(0, self.row_emb.num_embeddings - 1)
        col_ids.clamp_(0, self.col_emb.num_embeddings - 1)

        return row_ids, col_ids

    

    def _spatial_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Returns a (B, T, 1) bool mask: True for color tokens (0-9),
        False for all special tokens. Used to zero out 2D contribution
        on non-grid positions.
        """
        return ((input_ids >= 0) & (input_ids <= 9)).unsqueeze(-1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids : (B, T) long tensor
        Returns:
            embeddings : (B, T, d_model) float tensor
        """
        B, T = input_ids.shape
        device = input_ids.device

        # 1. Token embeddings
        tok_emb = self.token_emb(input_ids)                        # (B, T, d_model)

        # 2. 1D positional embeddings
        positions = torch.arange(T, device=device).unsqueeze(0)    # (1, T)
        pos_emb   = self.pos_emb_1d(positions)                     # (1, T, d_model)

        # 3. 2D spatial embeddings (only on color tokens)
        row_ids, col_ids = self._build_2d_position_ids(input_ids)
        r_emb = self.row_emb(row_ids)                              # (B, T, d_model/2)
        c_emb = self.col_emb(col_ids)                              # (B, T, d_model/2)
        spatial_emb = self.spatial_proj(
            torch.cat([r_emb, c_emb], dim=-1)                     # (B, T, d_model)
        )
        mask = self._spatial_mask(input_ids).float()               # (B, T, 1)
        spatial_emb = spatial_emb * mask                           # zero out specials

        # 4. Combine and normalise
        x = tok_emb + pos_emb + spatial_emb
        x = self.norm(x)
        return self.dropout(x)


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from src.utils.config import ModelConfig
    cfg = ModelConfig()

    emb_layer = ARCEmbeddings(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        max_seq_len=cfg.max_seq_len,
        max_h=cfg.max_grid_h,
        max_w=cfg.max_grid_w,
        dropout=cfg.dropout,
    )

    # Count parameters
    total_params = sum(p.numel() for p in emb_layer.parameters())
    print(f"ARCEmbeddings params: {total_params/1e6:.3f}M")

    # Fake batch: B=2, T=32
    dummy = torch.randint(0, 15, (2, 32))
    dummy[:, 0]  = BOS_TOKEN
    dummy[:, 16] = SEP_TOKEN
    dummy[:, 31] = EOS_TOKEN

    out = emb_layer(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (2, 32, cfg.d_model), "Shape mismatch!"
    print("ARCEmbeddings OK.")