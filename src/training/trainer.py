# src/training/trainer.py
"""
Training loop for ARC-Lite.

Features:
  - Gradient accumulation (effective batch = batch_size * grad_accum_steps)
  - Gradient clipping
  - WarmupCosine LR schedule
  - W&B logging (loss, ppl, lr, grad_norm, tokens/sec)
  - Checkpoint save/load
  - Eval split loss reported every eval_every steps
"""

import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, ".")

import wandb

from src.model.transformer import ARCTransformer
from src.data.dataset import ARCDataset, collate_fn
from src.training.loss import MDLLoss
from src.training.scheduler import WarmupCosineScheduler
from src.utils.config import Config


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mcfg = cfg.model
        self.tcfg = cfg.training

        self.device = torch.device(self.tcfg.device)
        self._setup_data()
        self._setup_model()
        self._setup_optimizer()
        self.loss_fn = MDLLoss(
            pad_token_id=self.mcfg.pad_token_id,
            label_smoothing=0.1,
        )
        self.global_step = 0
        self.best_eval_loss = float("inf")

    # ── Data ──────────────────────────────────────────────────────────────

    def _setup_data(self) -> None:
        train_dir = os.path.join(self.tcfg.data_dir, "training")
        eval_dir  = os.path.join(self.tcfg.data_dir, "evaluation")

        self.train_ds = ARCDataset(train_dir, self.tcfg.max_seq_len, split="training")
        self.eval_ds  = ARCDataset(eval_dir,  self.tcfg.max_seq_len, split="evaluation")

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=self.tcfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.tcfg.num_workers,
            pin_memory=True,
        )
        self.eval_loader = DataLoader(
            self.eval_ds,
            batch_size=self.tcfg.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.tcfg.num_workers,
            pin_memory=True,
        )

    # ── Model ─────────────────────────────────────────────────────────────

    def _setup_model(self) -> None:
        self.model = ARCTransformer(self.mcfg).to(self.device)
        n_params = self.model.count_parameters()
        print(f"[trainer] Model: {n_params/1e6:.3f}M parameters")
        print(f"[trainer] Device: {self.device}")

    # ── Optimiser ─────────────────────────────────────────────────────────

    def _setup_optimizer(self) -> None:
        # Separate weight decay: apply only to weight matrices, not biases/norms
        decay_params     = []
        no_decay_params  = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params,    "weight_decay": self.tcfg.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.tcfg.learning_rate,
            betas=(self.tcfg.beta1, self.tcfg.beta2),
        )

        self.scheduler = WarmupCosineScheduler(
            optimizer=self.optimizer,
            warmup_steps=self.tcfg.warmup_steps,
            max_steps=self.tcfg.max_steps,
            max_lr=self.tcfg.learning_rate,
        )

    # ── Checkpoint ────────────────────────────────────────────────────────

    def save_checkpoint(self, tag: str) -> None:
        os.makedirs(self.tcfg.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.tcfg.checkpoint_dir, f"arc_lite_{tag}.pt")
        torch.save({
            "step":            self.global_step,
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_eval_loss":  self.best_eval_loss,
            "model_config":    self.mcfg,
        }, path)
        print(f"[trainer] Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.global_step   = ckpt["step"]
        self.best_eval_loss = ckpt["best_eval_loss"]
        print(f"[trainer] Resumed from step {self.global_step}")

    # ── Evaluation ────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        total_loss   = 0.0
        total_tokens = 0

        for batch in self.eval_loader:
            ids  = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)

            logits = self.model(ids, mask)
            loss, metrics = self.loss_fn(logits, ids)

            total_loss   += loss.item() * metrics["n_tokens"]
            total_tokens += metrics["n_tokens"]

        self.model.train()
        return total_loss / max(total_tokens, 1)

    # ── Training loop ─────────────────────────────────────────────────────

    def train(self) -> None:
        wandb.init(
            project=self.tcfg.wandb_project,
            name=f"arc-lite-d{self.mcfg.d_model}-L{self.mcfg.n_layers}",
            config={
                "d_model":          self.mcfg.d_model,
                "n_heads":          self.mcfg.n_heads,
                "n_layers":         self.mcfg.n_layers,
                "d_ff":             self.mcfg.d_ff,
                "max_seq_len":      self.tcfg.max_seq_len,
                "batch_size":       self.tcfg.batch_size,
                "grad_accum":       self.tcfg.grad_accum_steps,
                "learning_rate":    self.tcfg.learning_rate,
                "warmup_steps":     self.tcfg.warmup_steps,
                "max_steps":        self.tcfg.max_steps,
                "train_sequences":  len(self.train_ds),
                "eval_sequences":   len(self.eval_ds),
            },
        )

        self.model.train()
        self.optimizer.zero_grad()

        data_iter    = iter(self.train_loader)
        accum_loss   = 0.0
        accum_tokens = 0
        t0           = time.time()

        print(f"[trainer] Starting training for {self.tcfg.max_steps} steps")
        print(f"[trainer] Grad accum steps: {self.tcfg.grad_accum_steps}")
        print(f"[trainer] Effective batch:  {self.tcfg.batch_size * self.tcfg.grad_accum_steps}")

        while self.global_step < self.tcfg.max_steps:

            # ── Gradient accumulation inner loop ──────────────────────────
            for micro_step in range(self.tcfg.grad_accum_steps):

                # Refill iterator if exhausted (one epoch done, keep going)
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    batch     = next(data_iter)

                ids  = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)

                logits = self.model(ids, mask)
                loss, metrics = self.loss_fn(logits, ids)

                # Scale loss by accum steps so gradients average correctly
                scaled_loss = loss / self.tcfg.grad_accum_steps
                scaled_loss.backward()

                accum_loss   += metrics["loss"]
                accum_tokens += metrics["n_tokens"]

            # ── Optimizer step ────────────────────────────────────────────
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), self.tcfg.grad_clip
            )

            lr = self.scheduler.step()   # update LR and advance step counter
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.global_step += 1

            # ── Logging ───────────────────────────────────────────────────
            if self.global_step % self.tcfg.log_every == 0:
                elapsed      = time.time() - t0
                tokens_per_s = accum_tokens / max(elapsed, 1e-6)
                avg_loss = accum_loss / (self.tcfg.log_every * self.tcfg.grad_accum_steps)
                avg_ppl      = torch.exp(torch.tensor(avg_loss)).item()

                print(
                    f"step {self.global_step:6d} | "
                    f"loss {avg_loss:.4f} | "
                    f"ppl {avg_ppl:.2f} | "
                    f"lr {lr:.2e} | "
                    f"grad_norm {grad_norm:.3f} | "
                    f"tok/s {tokens_per_s:.0f}"
                )

                wandb.log({
                    "train/loss":      avg_loss,
                    "train/ppl":       avg_ppl,
                    "train/lr":        lr,
                    "train/grad_norm": grad_norm,
                    "train/tokens_per_sec": tokens_per_s,
                }, step=self.global_step)

                accum_loss   = 0.0
                accum_tokens = 0
                t0           = time.time()

            # ── Evaluation ────────────────────────────────────────────────
            if self.global_step % self.tcfg.eval_every == 0:
                eval_loss = self.evaluate()
                eval_ppl  = torch.exp(torch.tensor(eval_loss)).item()
                print(f"  [eval] step {self.global_step} | eval_loss {eval_loss:.4f} | eval_ppl {eval_ppl:.2f}")
                wandb.log({
                    "eval/loss": eval_loss,
                    "eval/ppl":  eval_ppl,
                }, step=self.global_step)

                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self.save_checkpoint("best")

            # ── Checkpoint ────────────────────────────────────────────────
            if self.global_step % self.tcfg.save_every == 0:
                self.save_checkpoint(f"step{self.global_step:06d}")

        # Final checkpoint
        self.save_checkpoint("final")
        wandb.finish()
        print(f"[trainer] Training complete. Best eval loss: {self.best_eval_loss:.4f}")