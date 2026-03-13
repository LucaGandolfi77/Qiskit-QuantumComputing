"""
Dataset preparation and DataLoader factory.

Supports several scikit-learn toy datasets suitable for demonstrating
quantum advantage on small-dimensional problems:

    • **breast_cancer** — 30 features (binary classification)
    • **wine** — 13 features (multi-class, binarised for ≤2 classes)
    • **moons** — 2D synthetic crescent dataset
    • **circles** — 2D concentric circles
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import (
    load_breast_cancer,
    load_wine,
    make_moons,
    make_circles,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ──────────────────────────────────────────────────────────
#  Dataset registry
# ──────────────────────────────────────────────────────────

def _load_breast_cancer():
    data = load_breast_cancer()
    return data.data, data.target


def _load_wine():
    data = load_wine()
    # Binarise: class 0 vs rest
    X, y = data.data, data.target
    y = (y == 0).astype(int)
    return X, y


def _load_moons(n_samples: int = 1000, noise: float = 0.15):
    return make_moons(n_samples=n_samples, noise=noise, random_state=42)


def _load_circles(n_samples: int = 1000, noise: float = 0.1, factor: float = 0.5):
    return make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=42)


_DATASET_MAP = {
    "breast_cancer": _load_breast_cancer,
    "wine": _load_wine,
    "moons": _load_moons,
    "circles": _load_circles,
}


# ──────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────

def get_dataset(
    dataset_name: str,
    test_size: float = 0.2,
    normalize: bool = True,
    pca_components: int | None = None,
    seed: int = 42,
    save_dir: Path | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess a dataset.

    Returns
    -------
    X_train, X_test, y_train, y_test : ndarray
    """
    loader = _DATASET_MAP.get(dataset_name)
    if loader is None:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Choose from {list(_DATASET_MAP)}."
        )

    X, y = loader()
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Standardise
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Dimensionality reduction
    if pca_components is not None and pca_components < X_train.shape[1]:
        pca = PCA(n_components=pca_components, random_state=seed)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    # Optionally save
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / "X_train.npy", X_train)
        np.save(save_dir / "X_test.npy", X_test)
        np.save(save_dir / "y_train.npy", y_train)
        np.save(save_dir / "y_test.npy", y_test)

    return X_train, X_test, y_train, y_test


def get_dataloaders(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Wrap numpy arrays into PyTorch DataLoaders."""
    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long(),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).long(),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_data_from_config(cfg: dict, project_root: Path | None = None):
    """One-call convenience: config dict → DataLoaders.

    Returns
    -------
    train_loader, test_loader, input_dim : DataLoader, DataLoader, int
    """
    data_cfg = cfg.get("data", {})
    save_dir = None
    if project_root is not None:
        save_dir = project_root / data_cfg.get("data_dir", "data") / "processed"

    X_train, X_test, y_train, y_test = get_dataset(
        dataset_name=data_cfg.get("dataset_name", "breast_cancer"),
        test_size=data_cfg.get("test_size", 0.2),
        normalize=data_cfg.get("normalize", True),
        pca_components=data_cfg.get("pca_components"),
        seed=cfg.get("experiment", {}).get("seed", 42),
        save_dir=save_dir,
    )
    train_loader, test_loader = get_dataloaders(
        X_train, X_test, y_train, y_test,
        batch_size=data_cfg.get("batch_size", 32),
    )
    input_dim = X_train.shape[1]
    return train_loader, test_loader, input_dim
