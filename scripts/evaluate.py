# scripts/evaluate.py
"""
Entry point for ARC-Lite evaluation.
Usage:
  python scripts/evaluate.py --checkpoint checkpoints/arc_lite_best.pt
  python scripts/evaluate.py --checkpoint checkpoints/arc_lite_best.pt --split evaluation
  python scripts/evaluate.py --checkpoint checkpoints/arc_lite_best.pt --max_tasks 50
"""

import argparse
import sys
sys.path.insert(0, ".")

import torch

from src.model.transformer import ARCTransformer
from src.evaluation.evaluator import evaluate_split
from src.utils.config import ModelConfig


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ARC-Lite checkpoint")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--split",      type=str, default="evaluation")
    p.add_argument("--data_dir",   type=str, default="data/raw")
    p.add_argument("--max_tasks",  type=int, default=None)
    p.add_argument("--cpu",        action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cpu" if args.cpu else "cuda"

    print(f"[evaluate] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    cfg   = ckpt.get("model_config", ModelConfig())
    model = ARCTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    total_params = model.count_parameters()
    print(f"[evaluate] Model: {total_params/1e6:.3f}M parameters")
    print(f"[evaluate] Trained for {ckpt.get('step', '?')} steps")
    print(f"[evaluate] Split: {args.split}")

    split_dir = f"{args.data_dir}/{args.split}"
    result = evaluate_split(
        model,
        split_dir=split_dir,
        device=device,
        max_tasks=args.max_tasks,
    )

    print(f"\n{'='*40}")
    print(f"FINAL SCORE: {result['solved']}/{result['total']} ({result['accuracy']*100:.1f}%)")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()