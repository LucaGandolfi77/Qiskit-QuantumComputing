"""
Visualization utilities — training curves, circuit diagrams, Bloch spheres,
controller comparison plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import seaborn as sns

sns.set_theme(style="whitegrid", palette="colorblind")


# ──────────────────────────────────────────────────────────
#  Training curves
# ──────────────────────────────────────────────────────────

def plot_training_curves(
    history: dict[str, list[float]],
    title: str = "Training History",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot loss and accuracy curves for train and validation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax = axes[0]
    ax.plot(history["train_loss"], label="Train", linewidth=2)
    if history.get("val_loss"):
        ax.plot(history["val_loss"], label="Validation", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()

    # Accuracy
    ax = axes[1]
    ax.plot(history["train_acc"], label="Train", linewidth=2)
    if history.get("val_acc"):
        ax.plot(history["val_acc"], label="Validation", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.legend()

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ──────────────────────────────────────────────────────────
#  Model comparison
# ──────────────────────────────────────────────────────────

def plot_model_comparison(
    histories: dict[str, dict[str, list[float]]],
    metric: str = "val_acc",
    title: str = "Model Comparison",
    save_path: str | Path | None = None,
    smooth_window: int = 5,
) -> plt.Figure:
    """Overlay training curves from multiple models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, history in histories.items():
        values = history.get(metric, [])
        if not values:
            continue
        # Smoothing
        if smooth_window > 1 and len(values) >= smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            smoothed = np.convolve(values, kernel, mode="valid")
            ax.plot(smoothed, label=model_name, linewidth=2)
        else:
            ax.plot(values, label=model_name, linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ──────────────────────────────────────────────────────────
#  Confusion matrix
# ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str] | None = None,
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Heatmap-style confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names or "auto",
        yticklabels=class_names or "auto",
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ──────────────────────────────────────────────────────────
#  Expressibility / entanglement bar chart
# ──────────────────────────────────────────────────────────

def plot_circuit_properties(
    data: dict[str, dict[str, float]],
    title: str = "Circuit Analysis",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart comparing expressibility and entangling capability.

    Parameters
    ----------
    data : dict
        ``{circuit_name: {"expressibility": float, "entangling_capability": float}}``
    """
    names = list(data.keys())
    expr = [data[n].get("expressibility", 0) for n in names]
    ent = [data[n].get("entangling_capability", 0) for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width / 2, expr, width, label="Expressibility (KL ↓)", color="steelblue")
    bars2 = ax2.bar(x + width / 2, ent, width, label="Entangling Cap. (Q ↑)", color="coral")

    ax1.set_xlabel("Circuit")
    ax1.set_ylabel("Expressibility (KL div.)")
    ax2.set_ylabel("Entangling Capability (Q)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ──────────────────────────────────────────────────────────
#  Scalability plot
# ──────────────────────────────────────────────────────────

def plot_scalability(
    qubit_counts: Sequence[int],
    metrics: dict[str, list[float]],
    title: str = "Scalability Analysis",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot metrics vs qubit count."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for metric_name, values in metrics.items():
        ax.plot(qubit_counts, values, marker="o", linewidth=2, label=metric_name)

    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("Metric Value")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
