# src/training/scheduler.py
"""
Learning rate scheduler: linear warmup followed by cosine decay.

Schedule:
  Steps 0 to warmup_steps    : LR ramps linearly from 0 to max_lr
  Steps warmup_steps to end  : LR decays via cosine from max_lr to min_lr

This is the standard schedule for transformer training (GPT-style).
min_lr is set to 10% of max_lr — don't decay all the way to zero.
"""

import math
import torch
from torch.optim import Optimizer


class WarmupCosineScheduler:
    """
    Custom LR scheduler (not a torch.optim.lr_scheduler subclass —
    we call .step() manually each optimizer step for full control).

    Args:
        optimizer     : the optimizer whose LR we control
        warmup_steps  : number of steps for linear warmup
        max_steps     : total training steps
        max_lr        : peak learning rate (should match optimizer's initial LR)
        min_lr        : floor LR after cosine decay (default: max_lr * 0.1)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        max_lr: float,
        min_lr: float | None = None,
    ):
        self.optimizer    = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps    = max_steps
        self.max_lr       = max_lr
        self.min_lr       = min_lr if min_lr is not None else max_lr * 0.1
        self.current_step = 0

    def get_lr(self, step: int) -> float:
        """Returns the learning rate for a given step."""
        if step < self.warmup_steps:
            # Linear warmup
            return self.max_lr * (step + 1) / self.warmup_steps

        if step >= self.max_steps:
            return self.min_lr

        # Cosine decay
        progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.max_lr - self.min_lr) * cosine

    def step(self) -> float:
        """Advance one step and update optimizer LR. Returns current LR."""
        lr = self.get_lr(self.current_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.current_step += 1
        return lr

    def state_dict(self) -> dict:
        return {"current_step": self.current_step}

    def load_state_dict(self, state: dict) -> None:
        self.current_step = state["current_step"]


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend, safe on all systems
    import matplotlib.pyplot as plt

    # Dummy optimizer
    dummy_param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([dummy_param], lr=3e-4)

    scheduler = WarmupCosineScheduler(
        optimizer=opt,
        warmup_steps=500,
        max_steps=20_000,
        max_lr=3e-4,
    )

    lrs = [scheduler.step() for _ in range(20_000)]

    print(f"LR at step    0: {lrs[0]:.2e}  (should be ~0)")
    print(f"LR at step  499: {lrs[499]:.2e}  (warmup end ~3e-4)")
    print(f"LR at step 1000: {lrs[1000]:.2e} (cosine start)")
    print(f"LR at step 9999: {lrs[9999]:.2e} (mid decay)")
    print(f"LR at step19999: {lrs[19999]:.2e} (should be ~3e-5)")

    # Save plot to file for visual inspection
    plt.figure(figsize=(8, 3))
    plt.plot(lrs)
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title("WarmupCosine Schedule")
    plt.tight_layout()
    plt.savefig("lr_schedule.png", dpi=100)
    print("Schedule plot saved to lr_schedule.png")
    print("WarmupCosineScheduler OK.")