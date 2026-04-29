# scripts/train.py
"""
Entry point for ARC-Lite training.
Usage:
  python scripts/train.py                        # default config
  python scripts/train.py --max_steps 100        # quick smoke run
  python scripts/train.py --batch_size 8         # smaller batch for low VRAM
  python scripts/train.py --resume checkpoints/arc_lite_best.pt
"""

import argparse
import sys
sys.path.insert(0, ".")

from src.utils.config import Config, ModelConfig, TrainingConfig
from src.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ARC-Lite transformer")

    # Training overrides
    p.add_argument("--max_steps",       type=int,   default=None)
    p.add_argument("--batch_size",      type=int,   default=None)
    p.add_argument("--grad_accum",      type=int,   default=None)
    p.add_argument("--learning_rate",   type=float, default=None)
    p.add_argument("--warmup_steps",    type=int,   default=None)

    # Logging overrides
    p.add_argument("--log_every",       type=int,   default=None)
    p.add_argument("--eval_every",      type=int,   default=None)
    p.add_argument("--save_every",      type=int,   default=None)
    p.add_argument("--wandb_project",   type=str,   default=None)

    # Resume
    p.add_argument("--resume",          type=str,   default=None,
                   help="Path to checkpoint to resume from")

    # Device
    p.add_argument("--cpu",             action="store_true",
                   help="Force CPU (for debugging)")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = Config()

    # Apply overrides
    if args.max_steps     is not None: cfg.training.max_steps     = args.max_steps
    if args.batch_size    is not None: cfg.training.batch_size    = args.batch_size
    if args.grad_accum    is not None: cfg.training.grad_accum_steps = args.grad_accum
    if args.learning_rate is not None: cfg.training.learning_rate = args.learning_rate
    if args.warmup_steps  is not None: cfg.training.warmup_steps  = args.warmup_steps
    if args.log_every     is not None: cfg.training.log_every     = args.log_every
    if args.eval_every    is not None: cfg.training.eval_every    = args.eval_every
    if args.save_every    is not None: cfg.training.save_every    = args.save_every
    if args.wandb_project is not None: cfg.training.wandb_project = args.wandb_project
    if args.cpu:                       cfg.training.device        = "cpu"

    trainer = Trainer(cfg)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()