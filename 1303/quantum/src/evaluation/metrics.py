"""
Classification metrics for model evaluation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    metric_names: Sequence[str] = ("accuracy", "f1", "roc_auc"),
) -> dict[str, float]:
    """Compute a selection of classification metrics.

    Parameters
    ----------
    y_true : array, shape ``(n,)``
    y_pred : array, shape ``(n,)``
    y_prob : array, shape ``(n, n_classes)`` or ``(n,)``
        Class probabilities (needed for ROC-AUC).
    metric_names : list[str]
        Which metrics to compute.

    Returns
    -------
    dict[str, float]
    """
    results: dict[str, float] = {}

    for name in metric_names:
        if name == "accuracy":
            results["accuracy"] = float(accuracy_score(y_true, y_pred))
        elif name == "f1":
            results["f1"] = float(f1_score(y_true, y_pred, average="weighted"))
        elif name == "precision":
            results["precision"] = float(precision_score(y_true, y_pred, average="weighted"))
        elif name == "recall":
            results["recall"] = float(recall_score(y_true, y_pred, average="weighted"))
        elif name == "roc_auc":
            if y_prob is not None:
                try:
                    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                        results["roc_auc"] = float(
                            roc_auc_score(y_true, y_prob[:, 1])
                        )
                    else:
                        results["roc_auc"] = float(
                            roc_auc_score(y_true, y_prob, multi_class="ovr")
                        )
                except ValueError:
                    results["roc_auc"] = float("nan")
            else:
                results["roc_auc"] = float("nan")

    return results


def classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Return sklearn classification report as dict."""
    return classification_report(y_true, y_pred, output_dict=True)


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Return the confusion matrix."""
    return confusion_matrix(y_true, y_pred)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str = "cpu",
    metric_names: Sequence[str] = ("accuracy", "f1", "roc_auc"),
) -> dict[str, float]:
    """Run full evaluation on a DataLoader.

    Returns
    -------
    dict with requested metrics.
    """
    model.eval()
    device = torch.device(device)

    all_preds, all_labels, all_probs = [], [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

        all_preds.append(preds)
        all_labels.append(y_batch.numpy())
        all_probs.append(probs)

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)

    return compute_metrics(y_true, y_pred, y_prob, metric_names)
