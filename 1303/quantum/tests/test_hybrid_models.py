"""
Tests for hybrid models — end-to-end training, data loading, metrics.
"""

import pytest
import numpy as np
import torch

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloaders import get_dataset, get_dataloaders
from src.evaluation.metrics import compute_metrics, evaluate_model
from src.models.classical_net import ClassicalNet
from src.training.trainer import Trainer


# ═══════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════

class TestDataLoading:
    """Test dataset preparation and DataLoader creation."""

    @pytest.mark.parametrize("dataset", ["breast_cancer", "wine", "moons", "circles"])
    def test_get_dataset(self, dataset):
        X_train, X_test, y_train, y_test = get_dataset(
            dataset, test_size=0.2, normalize=True, pca_components=None
        )
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]

    def test_pca_reduction(self):
        X_train, X_test, _, _ = get_dataset(
            "breast_cancer", pca_components=4
        )
        assert X_train.shape[1] == 4
        assert X_test.shape[1] == 4

    def test_get_dataloaders(self):
        X_train, X_test, y_train, y_test = get_dataset("moons")
        train_loader, test_loader = get_dataloaders(
            X_train, X_test, y_train, y_test, batch_size=16
        )
        X_batch, y_batch = next(iter(train_loader))
        assert X_batch.dtype == torch.float32
        assert y_batch.dtype == torch.int64


# ═══════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════

class TestMetrics:
    """Test classification metrics computation."""

    def test_compute_metrics_perfect(self):
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 1])
        y_prob = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [0, 1]], dtype=float)
        m = compute_metrics(y_true, y_pred, y_prob, metric_names=["accuracy", "f1", "roc_auc"])
        assert m["accuracy"] == 1.0
        assert m["f1"] == 1.0

    def test_compute_metrics_no_probs(self):
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 0, 0])
        m = compute_metrics(y_true, y_pred, metric_names=["accuracy"])
        assert 0 <= m["accuracy"] <= 1.0


# ═══════════════════════════════════════════════════════════
#  End-to-end training (classical, fast)
# ═══════════════════════════════════════════════════════════

class TestTraining:
    """Test training loop with classical model (quick integration test)."""

    def test_classical_train_loop(self):
        """Run 3 epochs on moons dataset."""
        X_train, X_test, y_train, y_test = get_dataset("moons")
        train_loader, test_loader = get_dataloaders(
            X_train, X_test, y_train, y_test, batch_size=32
        )

        model = ClassicalNet(
            input_dim=X_train.shape[1],
            hidden_dims=[16],
            output_dim=2,
            dropout=0.0,
            batch_norm=False,
        )

        cfg = {
            "experiment": {"name": "test", "device": "cpu"},
            "training": {
                "n_epochs": 3,
                "learning_rate": 0.01,
                "optimizer": "adam",
                "weight_decay": 0.0,
                "patience": 100,
                "min_delta": 0.0,
            },
            "logging": {"tensorboard": False, "log_interval": 1, "save_best_only": False},
            "paths": {"models_dir": "/tmp/quantum_test_models"},
        }

        trainer = Trainer(model, cfg)
        history = trainer.train(train_loader, test_loader)

        assert len(history["train_loss"]) == 3
        assert len(history["val_acc"]) == 3
        # Loss should be finite
        assert all(np.isfinite(l) for l in history["train_loss"])

    def test_evaluate_model(self):
        """Evaluate returns metrics dict."""
        X_train, X_test, y_train, y_test = get_dataset("circles")
        _, test_loader = get_dataloaders(X_train, X_test, y_train, y_test)

        model = ClassicalNet(input_dim=2, hidden_dims=[8], output_dim=2, batch_norm=False)
        metrics = evaluate_model(model, test_loader, metric_names=["accuracy"])
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1.0
