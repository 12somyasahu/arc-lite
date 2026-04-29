# src/evaluation/evaluator.py
"""
ARC-AGI exact-match evaluation.

ARC scoring rule (official):
  A task is solved if the predicted output grid matches the ground-truth
  output grid exactly — every cell, every row, correct dimensions.
  Partial credit does not exist. One wrong cell = task failed.

We evaluate on the 'evaluation' split (400 tasks, each with 1 test pair).
The model generates output grids autoregressively given the test input.
"""

import json
import os
import sys
sys.path.insert(0, ".")

import torch

from src.data.dataset import (
    encode_task_pair, tokens_to_grid, grid_to_tokens,
    BOS_TOKEN, SEP_TOKEN, EOS_TOKEN, PAD_TOKEN, ROW_TOKEN, VOCAB_SIZE
)
from src.model.transformer import ARCTransformer
from src.utils.config import ModelConfig


def grids_equal(pred: list[list[int]], gold: list[list[int]]) -> bool:
    """Exact match: dimensions and every cell must match."""
    if len(pred) != len(gold):
        return False
    for row_p, row_g in zip(pred, gold):
        if len(row_p) != len(row_g):
            return False
        if row_p != row_g:
            return False
    return True


@torch.no_grad()
def generate_output_grid(
    model: ARCTransformer,
    input_grid: list[list[int]],
    max_new_tokens: int = 512,
    device: str = "cuda",
) -> list[list[int]]:
    """
    Given an input grid, autoregressively generate the output grid.

    Strategy:
      1. Build prompt: [BOS] + input_tokens + [SEP]
      2. Feed to model, sample greedily (argmax) until [EOS] or max_new_tokens
      3. Decode generated tokens back to a 2D grid
    """
    model.eval()

    # Build prompt sequence
    prompt_tokens = [BOS_TOKEN] + grid_to_tokens(input_grid) + [SEP_TOKEN]
    ids = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

    generated = []

    for _ in range(max_new_tokens):
        # Truncate if exceeding model's max_seq_len
        if ids.shape[1] >= model.cfg.max_seq_len:
            break

        logits = model(ids)                          # (1, T, vocab_size)
        next_token = logits[0, -1, :].argmax().item()  # greedy

        if next_token == EOS_TOKEN:
            break

        generated.append(next_token)
        next_id = torch.tensor([[next_token]], dtype=torch.long, device=device)
        ids = torch.cat([ids, next_id], dim=1)

    return tokens_to_grid(generated, width=30)


def evaluate_split(
    model: ARCTransformer,
    split_dir: str,
    device: str = "cuda",
    max_tasks: int | None = None,
) -> dict:
    """
    Evaluate exact-match accuracy on all tasks in split_dir.

    Each task has one test pair: one input, one ground-truth output.
    Returns a dict with solved count, total count, and accuracy.
    """
    json_files = sorted(f for f in os.listdir(split_dir) if f.endswith(".json"))
    if max_tasks:
        json_files = json_files[:max_tasks]

    solved  = 0
    total   = 0
    results = []

    for fname in json_files:
        task_id = fname.replace(".json", "")
        path    = os.path.join(split_dir, fname)

        with open(path, "r", encoding="utf-8") as f:
            task = json.load(f)

        # Each task has one or more test pairs
        task_solved = False
        for pair in task["test"]:
            input_grid = pair["input"]
            gold_grid  = pair["output"]

            pred_grid = generate_output_grid(model, input_grid, device=device)
            correct   = grids_equal(pred_grid, gold_grid)

            if correct:
                task_solved = True

        if task_solved:
            solved += 1
        total += 1

        results.append({
            "task_id": task_id,
            "solved":  task_solved,
        })

    accuracy = solved / total if total > 0 else 0.0
    print(f"[eval] Solved {solved}/{total} tasks ({accuracy*100:.1f}%)")
    return {
        "solved":   solved,
        "total":    total,
        "accuracy": accuracy,
        "results":  results,
    }


# ── Smoke test (untrained model, expect ~0%) ──────────────────────────────────
if __name__ == "__main__":
    from src.utils.config import ModelConfig

    cfg   = ModelConfig()
    model = ARCTransformer(cfg).cuda()
    model.eval()

    # Test on just 5 tasks so it's fast
    result = evaluate_split(
        model,
        split_dir="data/raw/evaluation",
        device="cuda",
        max_tasks=5,
    )
    print(f"Accuracy (untrained): {result['accuracy']*100:.1f}%  (expect 0%)")