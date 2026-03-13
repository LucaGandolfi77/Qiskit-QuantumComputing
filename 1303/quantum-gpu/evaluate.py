#!/usr/bin/env python3
"""
CLI evaluation entry point.

Load a trained checkpoint and compute classification metrics on the test set.

Usage::

    python evaluate.py --config config/hybrid_pennylane.yaml \\
                       --checkpoint outputs/models/hybrid_pennylane/checkpoint_best.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config
from src.utils.quantum_utils import set_seed
from src.data.dataloaders import get_data_from_config
from src.evaluation.metrics import evaluate_model, compute_confusion_matrix
from src.evaluation.visualization import plot_confusion_matrix


def _build_model(cfg: dict, input_dim: int):
    """Same model builder as train.py."""
    framework = cfg.get("experiment", {}).get("framework", "classical")
    if "classical" in cfg:
        cfg["classical"]["input_dim"] = input_dim

    if framework == "classical":
        from src.models.classical_net import ClassicalNet
        return ClassicalNet.from_config(cfg["classical"])
    elif framework == "pennylane":
        from src.models.hybrid_pennylane_net import HybridPennyLaneNet
        return HybridPennyLaneNet.from_config(cfg)
    elif framework == "qiskit":
        from src.models.hybrid_qiskit_net import HybridQiskitNet
        return HybridQiskitNet.from_config(cfg)
    else:
        raise ValueError(f"Unknown framework '{framework}'.")


@click.command()
@click.option("--config", "-c", type=click.Path(exists=True), required=True)
@click.option("--checkpoint", "-ckpt", type=click.Path(exists=True), required=True)
@click.option("--device", "-d", type=str, default="cpu")
def main(config, checkpoint, device):
    """Evaluate a trained model checkpoint."""
    cfg = load_config(config)
    set_seed(cfg["experiment"].get("seed", 42))

    # Data
    _, test_loader, input_dim = get_data_from_config(cfg, PROJECT_ROOT)

    # Model
    model = _build_model(cfg, input_dim)

    # Load weights
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    click.echo(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Evaluate
    metrics = evaluate_model(
        model, test_loader, device=device,
        metric_names=cfg.get("evaluation", {}).get("metrics", ["accuracy", "f1", "roc_auc"]),
    )

    click.echo("\n─── Evaluation Results ───")
    for k, v in metrics.items():
        click.echo(f"  {k:15s}: {v:.4f}")

    # Confusion matrix
    if cfg.get("evaluation", {}).get("confusion_matrix", False):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y in test_loader:
                preds = model(X.to(device)).argmax(dim=1).cpu()
                all_preds.append(preds)
                all_labels.append(y)
        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()
        cm = compute_confusion_matrix(y_true, y_pred)

        plots_dir = Path(cfg.get("paths", {}).get("plots_dir", "outputs/plots"))
        plot_confusion_matrix(
            cm,
            title=f"{cfg['experiment']['name']} — Confusion Matrix",
            save_path=plots_dir / "confusion_matrix.png",
        )
        click.echo(f"\nConfusion matrix saved → {plots_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
