# src/training/loss.py
"""
MDL-inspired joint loss for ARC-Lite.

Core idea (from Vakde's mdlARC, conceptually):
  Standard next-token prediction only computes loss on output grid tokens.
  The MDL principle says: the best model is the one that compresses the
  ENTIRE sequence most efficiently — input grid AND output grid.
  So we compute cross-entropy loss on every non-padding token.

This forces the model to:
  1. Learn to reconstruct input grids (understand structure)
  2. Learn to predict output grids (learn the transformation)
  3. Do both jointly — the shared representation must serve both.

Loss masking:
  - PAD tokens (10): always masked out (no gradient)
  - All other tokens (colors 0-9, BOS, EOS, SEP, ROW): loss included

The loss is mean over all unmasked tokens in the batch (not mean over
sequences), so longer sequences don't get unfairly down-weighted.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dataset import PAD_TOKEN


class MDLLoss(nn.Module):
    """
    Next-token prediction loss over the full sequence.
    Padding tokens are excluded from the loss.

    Args:
        pad_token_id     : token ID to ignore in loss (default: PAD_TOKEN=10)
        label_smoothing  : label smoothing factor (0.0 = off)
    """

    def __init__(self, pad_token_id: int = PAD_TOKEN, label_smoothing: float = 0.1):
        super().__init__()
        self.pad_token_id    = pad_token_id
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            logits    : (B, T, vocab_size) — raw model output
            input_ids : (B, T) — the original input token ids

        Returns:
            loss    : scalar tensor
            metrics : dict with 'loss', 'ppl', 'n_tokens' for logging
        """
        B, T, V = logits.shape

        # Next-token prediction: shift by 1
        # logits[:, :-1] predicts input_ids[:, 1:]
        shift_logits = logits[:, :-1, :].contiguous()     # (B, T-1, V)
        shift_labels = input_ids[:, 1:].contiguous()      # (B, T-1)

        # Flatten for F.cross_entropy
        flat_logits = shift_logits.view(-1, V)             # (B*(T-1), V)
        flat_labels = shift_labels.view(-1)                # (B*(T-1),)

        # Count non-padding tokens for reporting
        n_tokens = (flat_labels != self.pad_token_id).sum().item()

        loss = F.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=self.pad_token_id,
            label_smoothing=self.label_smoothing,
            reduction="mean",
        )

        # Perplexity: exp(loss) — lower is better
        ppl = torch.exp(loss.detach()).item()

        return loss, {
            "loss":     loss.item(),
            "ppl":      ppl,
            "n_tokens": n_tokens,
        }


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from src.data.dataset import PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN

    loss_fn = MDLLoss(pad_token_id=PAD_TOKEN, label_smoothing=0.1)

    B, T, V = 4, 64, 15
    logits   = torch.randn(B, T, V, requires_grad=True)
    ids      = torch.randint(0, 10, (B, T))
    ids[:, 0]   = BOS_TOKEN
    ids[:, 32]  = SEP_TOKEN
    ids[:, -1]  = EOS_TOKEN
    ids[0, 50:] = PAD_TOKEN   # simulate padding

    loss, metrics = loss_fn(logits, ids)
    print(f"Loss:     {metrics['loss']:.4f}")
    print(f"PPL:      {metrics['ppl']:.4f}")
    print(f"N tokens: {metrics['n_tokens']}")
    assert loss.requires_grad, "Loss must have grad for backprop"
    print("MDLLoss OK.")