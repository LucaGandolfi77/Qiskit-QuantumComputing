#!/usr/bin/env python3
"""
CLI training entry point for Quantum-Classical Hybrid Neural Networks.

Usage::

    python train.py --config config/classical_baseline.yaml
    python train.py --config config/hybrid_pennylane.yaml --epochs 50
    python train.py --config config/hybrid_qiskit.yaml --device cpu
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import torch

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config, get_device
from src.utils.logger import QuantumLogger
from src.utils.quantum_utils import set_seed, count_model_parameters
from src.data.dataloaders import get_data_from_config
from src.models.classical_net import ClassicalNet
from src.models.hybrid_pennylane_net import HybridPennyLaneNet
from src.training.trainer import Trainer
from src.training.quantum_aware_training import QuantumAwareTrainer
from src.evaluation.metrics import evaluate_model
from src.evaluation.visualization import plot_training_curves


# ── Model registry ───────────────────────────────────────

def _build_model(cfg: dict, input_dim: int) -> torch.nn.Module:
    """Dispatch model construction based on framework."""
    framework = cfg.get("experiment", {}).get("framework", "classical")

    # Override input_dim from data
    if "classical" in cfg:
        cfg["classical"]["input_dim"] = input_dim

    if framework == "classical":
        return ClassicalNet.from_config(cfg["classical"])

    elif framework == "pennylane":
        return HybridPennyLaneNet.from_config(cfg)

    elif framework == "qiskit":
        from src.models.hybrid_qiskit_net import HybridQiskitNet
        return HybridQiskitNet.from_config(cfg)

    else:
        raise ValueError(f"Unknown framework '{framework}'.")


# ── CLI ──────────────────────────────────────────────────

@click.command()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to YAML config file.",
)
@click.option("--epochs", "-e", type=int, default=None, help="Override n_epochs.")
@click.option("--lr", type=float, default=None, help="Override learning rate.")
@click.option("--device", "-d", type=str, default=None, help="Override device.")
@click.option("--seed", "-s", type=int, default=None, help="Override random seed.")
@click.option("--quantum-warmup", is_flag=True, help="Use layer-wise quantum warm-up.")
def main(config, epochs, lr, device, seed, quantum_warmup):
    """Train a classical or hybrid quantum-classical model."""
    # Load config
    cfg = load_config(config)

    # CLI overrides
    if epochs is not None:
        cfg["training"]["n_epochs"] = epochs
    if lr is not None:
        cfg["training"]["learning_rate"] = lr
    if device is not None:
        cfg["experiment"]["device"] = device
    if seed is not None:
        cfg["experiment"]["seed"] = seed

    # Reproducibility
    set_seed(cfg["experiment"].get("seed", 42))

    framework = cfg["experiment"].get("framework", "classical")
    exp_name = cfg["experiment"].get("name", "experiment")

    # Logger
    logger = QuantumLogger(
        name=exp_name,
        log_dir=cfg.get("paths", {}).get("tensorboard_dir", "outputs/tensorboard"),
        use_tensorboard=cfg.get("logging", {}).get("tensorboard", True),
    )
    logger.info(f"Framework: {framework} | Experiment: {exp_name}")

    # Data
    train_loader, test_loader, input_dim = get_data_from_config(cfg, PROJECT_ROOT)
    logger.info(f"Dataset loaded: input_dim={input_dim}")

    # Model
    model = _build_model(cfg, input_dim)
    param_info = count_model_parameters(model)
    logger.info(f"Model parameters: {param_info}")

    # Device (classical parts)
    dev = get_device(cfg["experiment"].get("device", "cpu"))
    if framework == "classical":
        model = model.to(dev)
    else:
        # For hybrid models, move only the classical submodules to the accelerator
        # and keep the quantum simulator on CPU to avoid device mismatches.
        try:
            if hasattr(model, "pre_net"):
                model.pre_net.to(dev)
            if hasattr(model, "post_net"):
                model.post_net.to(dev)
        except Exception:
            # best-effort move; if it fails, continue and let runtime reveal issues
            pass

    # Trainer
    if framework != "classical" and quantum_warmup:
        trainer = QuantumAwareTrainer(model, cfg, logger)
        trainer.layer_wise_warmup(train_loader, test_loader, warmup_epochs=10)
    else:
        trainer = Trainer(model, cfg, logger)

    # Train
    history = trainer.train(train_loader, test_loader)

    # Final evaluation
    logger.info("=" * 50)
    logger.info("Final Evaluation on Test Set")
    metrics = evaluate_model(
        model, test_loader,
        device=cfg["experiment"].get("device", "cpu"),
        metric_names=cfg.get("evaluation", {}).get("metrics", ["accuracy", "f1"]),
    )
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # Plot
    plots_dir = Path(cfg.get("paths", {}).get("plots_dir", "outputs/plots"))
    plot_training_curves(
        history,
        title=f"{exp_name} — Training History",
        save_path=plots_dir / f"{exp_name}_training_curves.png",
    )
    logger.info(f"Training curves saved → {plots_dir}")

    logger.close()
    click.echo(f"\n✓ Training complete — {exp_name}")


if __name__ == "__main__":
    main()
