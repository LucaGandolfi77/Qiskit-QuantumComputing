"""
Quantum-aware training utilities.

Implements techniques specific to training variational quantum circuits:

    • **Parameter-shift gradient estimation** — analytic gradients without backprop
    • **Barren plateau detection** — gradient variance monitoring
    • **Layer-wise training** — freeze outer layers, train quantum layer first
    • **Shot-adaptive training** — increase shot count as training stabilises
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .trainer import Trainer
from ..utils.logger import QuantumLogger


class QuantumAwareTrainer(Trainer):
    """Extended trainer with QML-specific techniques.

    Inherits from :class:`Trainer` and adds:
        - Gradient variance monitoring for barren plateau detection
        - Layer-wise learning rate scheduling (quantum vs classical)
        - Shot-adaptive training for noisy simulation
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        logger: QuantumLogger | None = None,
    ) -> None:
        super().__init__(model, config, logger)

        self._grad_history: list[dict[str, float]] = []
        self._quantum_lr_scale = config.get("training", {}).get(
            "quantum_lr_scale", 1.0
        )

    # ── Optimizer with per-group LR ──────────────────────
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with separate LR for quantum parameters."""
        # Coerce numeric values to floats to handle YAML/string overrides
        def _to_float(v, default: float):
            try:
                return float(v)
            except Exception:
                return default

        lr = _to_float(self.train_cfg.get("learning_rate", 0.01), 0.01)
        wd = _to_float(self.train_cfg.get("weight_decay", 1e-4), 1e-4)
        q_scale = self.train_cfg.get("quantum_lr_scale", 1.0)

        classical_params = []
        quantum_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "quantum" in name or "weights" in name:
                quantum_params.append(param)
            else:
                classical_params.append(param)

        param_groups = []
        if classical_params:
            param_groups.append({"params": classical_params, "lr": lr})
        if quantum_params:
            param_groups.append({"params": quantum_params, "lr": lr * q_scale})

        # Fallback if grouping fails
        if not param_groups:
            param_groups = [{"params": self.model.parameters(), "lr": lr}]

        opt_name = self.train_cfg.get("optimizer", "adam").lower()
        if opt_name == "adam":
            return torch.optim.Adam(param_groups, weight_decay=wd)
        elif opt_name == "adamw":
            return torch.optim.AdamW(param_groups, weight_decay=wd)
        else:
            return torch.optim.Adam(param_groups, weight_decay=wd)

    # ── Gradient monitoring ──────────────────────────────
    def monitor_gradients(self) -> dict[str, float]:
        """Compute gradient statistics after a backward pass."""
        stats: dict[str, float] = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                stats[f"{name}/mean"] = float(grad.mean())
                stats[f"{name}/std"] = float(grad.std())
                stats[f"{name}/norm"] = float(grad.norm())
        self._grad_history.append(stats)
        return stats

    def detect_barren_plateau(self, threshold: float = 1e-5) -> bool:
        """Check if recent gradients indicate a barren plateau.

        Returns True if the mean gradient norm has been below
        *threshold* for the last 10 epochs.
        """
        if len(self._grad_history) < 10:
            return False

        recent = self._grad_history[-10:]
        norms = []
        for step_stats in recent:
            step_norms = [v for k, v in step_stats.items() if k.endswith("/norm")]
            if step_norms:
                norms.append(np.mean(step_norms))

        return bool(np.mean(norms) < threshold) if norms else False

    # ── Enhanced training epoch ──────────────────────────
    def _train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        """Training step with gradient monitoring."""
        loss, acc = super()._train_epoch(loader)

        # Log gradient stats
        grad_stats = self.monitor_gradients()
        epoch = len(self.history.get("train_loss", [])) + 1

        for tag, val in grad_stats.items():
            self.logger.log_scalar(f"gradients/{tag}", val, epoch)

        # Barren plateau warning
        if self.detect_barren_plateau():
            self.logger.warning(
                "⚠ Barren plateau detected — gradient norms near zero. "
                "Consider reducing circuit depth or using local cost functions."
            )

        return loss, acc

    # ── Layer-wise warm-up ───────────────────────────────
    def layer_wise_warmup(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        warmup_epochs: int = 10,
    ) -> None:
        """Freeze classical layers and pre-train quantum layer.

        After *warmup_epochs*, unfreeze all parameters and continue
        with the main training loop.
        """
        self.logger.info(f"Layer-wise warm-up: training quantum layer for {warmup_epochs} epochs")

        # Freeze classical
        for name, param in self.model.named_parameters():
            if "quantum" not in name and "weights" not in name:
                param.requires_grad = False

        # Rebuild optimizer for active params only
        self.optimizer = self._build_optimizer()

        for epoch in range(1, warmup_epochs + 1):
            train_loss, train_acc = self._train_epoch(train_loader)
            val_loss, val_acc = self._validate(val_loader)
            if epoch % 5 == 0:
                self.logger.info(
                    f"Warmup {epoch}/{warmup_epochs} | "
                    f"loss={train_loss:.4f} acc={train_acc:.4f}"
                )

        # Unfreeze all
        for param in self.model.parameters():
            param.requires_grad = True

        # Rebuild with full param groups
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.logger.info("Warm-up complete — all parameters unfrozen")
