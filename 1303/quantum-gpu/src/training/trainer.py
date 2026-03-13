"""
Training orchestration for classical and hybrid models.

Handles:
    • Training loop with configurable optimizer / scheduler
    • Validation at each epoch
    • Early stopping
    • Checkpoint saving (best + periodic)
    • TensorBoard logging
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utils.logger import QuantumLogger


class Trainer:
    """General-purpose trainer for both classical and hybrid models.

    Parameters
    ----------
    model : nn.Module
        The model to train (classical, hybrid-PennyLane, or hybrid-Qiskit).
    config : dict
        Full experiment config (training, logging, paths sections used).
    logger : QuantumLogger | None
        Optional logger instance.
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        logger: QuantumLogger | None = None,
    ) -> None:
        self.model = model
        self.cfg = config
        self.train_cfg = config.get("training", {})
        self.logger = logger or QuantumLogger(
            name=config.get("experiment", {}).get("name", "train"),
            log_dir=config.get("paths", {}).get("tensorboard_dir", "outputs/tensorboard"),
            use_tensorboard=config.get("logging", {}).get("tensorboard", True),
        )

        # Device — classical parts on accelerator, quantum stays CPU
        self.device = torch.device(
            config.get("experiment", {}).get("device", "cpu")
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # Early stopping
        self.patience = self.train_cfg.get("patience", 20)
        self.min_delta = self.train_cfg.get("min_delta", 0.001)
        self._best_val_loss = float("inf")
        self._patience_counter = 0

        # Tracking
        self.history: dict[str, list[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
        }

    # ── Optimizer & scheduler builders ───────────────────
    def _build_optimizer(self) -> torch.optim.Optimizer:
        opt_name = self.train_cfg.get("optimizer", "adam").lower()
        # Coerce numeric values to floats to handle YAML/string overrides
        def _to_float(v, default: float):
            try:
                return float(v)
            except Exception:
                return default

        lr = _to_float(self.train_cfg.get("learning_rate", 0.01), 0.01)
        wd = _to_float(self.train_cfg.get("weight_decay", 1e-4), 1e-4)

        if opt_name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        elif opt_name == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        else:
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)

    def _build_scheduler(self):
        sched_cfg = self.train_cfg.get("scheduler", {})
        sched_type = sched_cfg.get("type", "cosine")
        params = sched_cfg.get("params", {})

        if sched_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=params.get("T_max", self.train_cfg.get("n_epochs", 100))
            )
        elif sched_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=params.get("step_size", 30),
                gamma=params.get("gamma", 0.5),
            )
        elif sched_type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, patience=params.get("patience", 10), factor=params.get("factor", 0.5)
            )
        return None

    # ── Training loop ────────────────────────────────────
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict[str, list[float]]:
        """Run the full training loop.

        Returns
        -------
        dict
            History of train/val losses and accuracies.
        """
        n_epochs = self.train_cfg.get("n_epochs", 100)
        log_interval = self.cfg.get("logging", {}).get("log_interval", 10)

        self.logger.info(
            f"Starting training: {n_epochs} epochs, device={self.device}, "
            f"lr={self.train_cfg.get('learning_rate')}"
        )

        for epoch in range(1, n_epochs + 1):
            # ── Train step ───────────────────────────────
            train_loss, train_acc = self._train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            # ── Validation step ──────────────────────────
            val_loss, val_acc = self._validate(val_loader)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # ── Scheduler step ───────────────────────────
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # ── Logging ──────────────────────────────────
            if epoch % log_interval == 0 or epoch == 1:
                self.logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc)
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.logger.log_scalar("lr", current_lr, epoch)

            # ── Checkpointing ────────────────────────────
            if val_loss < self._best_val_loss - self.min_delta:
                self._best_val_loss = val_loss
                self._patience_counter = 0
                if self.cfg.get("logging", {}).get("save_best_only", True):
                    self._save_checkpoint(epoch, is_best=True)
            else:
                self._patience_counter += 1

            # ── Early stopping ───────────────────────────
            if self._patience_counter >= self.patience:
                self.logger.info(
                    f"Early stopping at epoch {epoch} (patience={self.patience})"
                )
                break

        self.logger.close()
        return self.history

    # ── Epoch helpers ────────────────────────────────────
    def _train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            loss.backward()
            # Gradient clipping for quantum stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)

        return total_loss / total, correct / total

    # ── Checkpoint ───────────────────────────────────────
    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        models_dir = Path(
            self.cfg.get("paths", {}).get("models_dir", "outputs/models")
        )
        models_dir.mkdir(parents=True, exist_ok=True)

        tag = "best" if is_best else f"epoch_{epoch}"
        path = models_dir / f"checkpoint_{tag}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": self._best_val_loss,
                "history": self.history,
                "config": self.cfg,
            },
            path,
        )
        self.logger.info(f"Checkpoint saved → {path}")
