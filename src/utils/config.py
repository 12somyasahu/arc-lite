# src/utils/config.py
"""
Central configuration dataclass for ARC-Lite.
All hyperparameters live here — model architecture, training, paths.
Import this everywhere instead of scattering magic numbers.
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # ── Vocabulary ─────────────────────────────────────────────────────────
    vocab_size: int = 15          # tokens 0-9 (colors) + PAD/BOS/EOS/SEP/ROW
    pad_token_id: int = 10

    # ── Sequence ───────────────────────────────────────────────────────────
    max_seq_len: int = 1024       # max tokens per sequence (covers 99% of tasks)

    # ── Grid geometry ──────────────────────────────────────────────────────
    max_grid_h: int = 30          # ARC grids are at most 30x30
    max_grid_w: int = 30

    # ── Transformer architecture ───────────────────────────────────────────
    d_model: int = 384            # embedding dimension
    n_heads: int = 8              # attention heads (d_model must be divisible)
    n_layers: int = 6             # transformer blocks
    d_ff: int = 1536              # feed-forward inner dimension (4x d_model)
    dropout: float = 0.1

    # ── Positional encoding ────────────────────────────────────────────────
    use_2d_pos: bool = True       # use 2D row/col embeddings inside grid regions


@dataclass
class TrainingConfig:
    # ── Paths ──────────────────────────────────────────────────────────────
    data_dir: str = "data/raw"
    checkpoint_dir: str = "checkpoints"

    # ── Batch / sequence ───────────────────────────────────────────────────
    batch_size: int = 16          # fits in 6GB VRAM at d_model=256, seq_len=1024
    grad_accum_steps: int = 4     # effective batch = 16 * 4 = 64
    max_seq_len: int = 1024

    # ── Optimiser ──────────────────────────────────────────────────────────
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # ── Schedule ───────────────────────────────────────────────────────────
    warmup_steps: int = 500
    max_steps: int = 20_000

    # ── Logging / checkpointing ────────────────────────────────────────────
    log_every: int = 50           # log loss every N steps
    eval_every: int = 500         # run eval split every N steps
    save_every: int = 1000        # save checkpoint every N steps
    wandb_project: str = "arc-lite"

    # ── Device ─────────────────────────────────────────────────────────────
    device: str = "cuda"          # "cuda" or "cpu"
    num_workers: int = 0          # DataLoader workers (0 = main process, safe on Windows)


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = Config()
    d = cfg.model.d_model
    h = cfg.model.n_heads
    assert d % h == 0, f"d_model {d} must be divisible by n_heads {h}"

    # Rough parameter estimate
    # Embedding table: vocab_size * d_model
    emb = cfg.model.vocab_size * cfg.model.d_model
    # Per transformer block: ~12 * d_model^2 (standard rule of thumb)
    per_block = 12 * cfg.model.d_model ** 2
    total = emb + cfg.model.n_layers * per_block
    print(f"d_model={d}, n_heads={h}, n_layers={cfg.model.n_layers}")
    print(f"Rough param estimate: {total/1e6:.1f}M")
    print(f"Effective batch size: {cfg.training.batch_size * cfg.training.grad_accum_steps}")
    print("Config OK.")