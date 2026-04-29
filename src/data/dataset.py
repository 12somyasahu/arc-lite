# src/data/dataset.py
"""
PyTorch Dataset for ARC-AGI.
Loads raw JSON tasks, tokenizes grids into integer sequences,
and pads them to a fixed max length for batching.

Token vocabulary:
  0-9   : ARC colors (0=black, 1=blue, ..., 9=maroon)
  10    : [PAD]   — padding token
  11    : [BOS]   — beginning of sequence
  12    : [EOS]   — end of sequence
  13    : [SEP]   — separates input grid from output grid
  14    : [ROW]   — marks end of each row within a grid

VOCAB_SIZE = 15
"""

import json
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

# ── Special tokens ────────────────────────────────────────────────────────────
PAD_TOKEN  = 10
BOS_TOKEN  = 11
EOS_TOKEN  = 12
SEP_TOKEN  = 13   # separates input grid from output grid in the sequence
ROW_TOKEN  = 14   # appended at the end of each row

VOCAB_SIZE = 15   # tokens 0..14


def grid_to_tokens(grid: list[list[int]]) -> list[int]:
    """
    Converts a 2D ARC grid (list of lists) into a flat token sequence.
    Each row is followed by a ROW_TOKEN.

    Example: [[0,1],[2,3]] -> [0, 1, ROW, 2, 3, ROW]
    """
    tokens = []
    for row in grid:
        tokens.extend(row)
        tokens.append(ROW_TOKEN)
    return tokens


def tokens_to_grid(tokens: list[int], width: int) -> list[list[int]]:
    """
    Reconstructs a 2D grid from a flat token sequence.
    Stops at EOS_TOKEN or end of list.
    Strips ROW_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN.
    """
    grid = []
    row = []
    special = {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN}

    for tok in tokens:
        if tok == EOS_TOKEN:
            break
        if tok in special:
            continue
        if tok == ROW_TOKEN:
            if row:
                grid.append(row)
            row = []
        else:
            row.append(tok)

    if row:  # flush last row if no trailing ROW_TOKEN
        grid.append(row)

    return grid


def encode_task_pair(
    input_grid: list[list[int]],
    output_grid: list[list[int]],
) -> list[int]:
    """
    Encodes one (input, output) pair into a single flat sequence:
    [BOS] + input_tokens + [SEP] + output_tokens + [EOS]

    This is the MDL-inspired joint sequence: the model sees both
    input and output and learns to compress/reconstruct both.
    """
    seq = [BOS_TOKEN]
    seq += grid_to_tokens(input_grid)
    seq += [SEP_TOKEN]
    seq += grid_to_tokens(output_grid)
    seq += [EOS_TOKEN]
    return seq


class ARCDataset(Dataset):
    """
    Loads all tasks from a given split directory.
    Each ARC task has multiple (input, output) training pairs.
    Each pair becomes one sequence in the dataset.

    Args:
        split_dir   : path to data/raw/{split}/
        max_seq_len : maximum token sequence length; longer sequences are skipped
        split       : human label used for logging only
    """

    def __init__(
        self,
        split_dir: str,
        max_seq_len: int = 1024,
        split: str = "unknown",
    ):
        self.split_dir   = split_dir
        self.max_seq_len = max_seq_len
        self.split       = split

        self.sequences: list[list[int]] = []   # raw token lists
        self.task_ids:  list[str]       = []   # which task each seq came from

        self._load()

    def _load(self) -> None:
        json_files = sorted(f for f in os.listdir(self.split_dir) if f.endswith(".json"))
        skipped = 0

        for fname in json_files:
            task_id = fname.replace(".json", "")
            path = os.path.join(self.split_dir, fname)

            with open(path, "r", encoding="utf-8") as f:
                task = json.load(f)

            # Each task has a "train" list of {input, output} pairs
            for pair in task["train"]:
                seq = encode_task_pair(pair["input"], pair["output"])
                if len(seq) > self.max_seq_len:
                    skipped += 1
                    continue
                self.sequences.append(seq)
                self.task_ids.append(task_id)

        print(
            f"[dataset] split='{self.split}' | "
            f"sequences={len(self.sequences)} | "
            f"skipped(too_long)={skipped} | "
            f"max_seq_len={self.max_seq_len}"
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        return {"input_ids": torch.tensor(seq, dtype=torch.long)}


def collate_fn(batch: list[dict], pad_token: int = PAD_TOKEN) -> dict[str, torch.Tensor]:
    """
    Pads sequences in a batch to the same length (right-padding with PAD_TOKEN).
    Returns:
        input_ids      : (B, T) long tensor
        attention_mask : (B, T) bool tensor — True for real tokens, False for padding
    """
    seqs = [item["input_ids"] for item in batch]
    max_len = max(s.size(0) for s in seqs)

    padded   = torch.full((len(seqs), max_len), pad_token, dtype=torch.long)
    attn_mask = torch.zeros(len(seqs), max_len, dtype=torch.bool)

    for i, s in enumerate(seqs):
        padded[i, : s.size(0)] = s
        attn_mask[i, : s.size(0)] = True

    return {"input_ids": padded, "attention_mask": attn_mask}


# ── Quick stats printout ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import collections
    from torch.utils.data import DataLoader

    raw_dir = "data/raw"

    for split in ["training", "evaluation"]:
        split_dir = os.path.join(raw_dir, split)
        ds = ARCDataset(split_dir, max_seq_len=1024, split=split)

        lengths = [len(s) for s in ds.sequences]
        print(f"  min_len={min(lengths)}  max_len={max(lengths)}  "
              f"mean_len={sum(lengths)/len(lengths):.1f}")

        # Batch test
        loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn, shuffle=True)
        batch = next(iter(loader))
        print(f"  batch input_ids:      {batch['input_ids'].shape}")
        print(f"  batch attention_mask: {batch['attention_mask'].shape}")
        print()