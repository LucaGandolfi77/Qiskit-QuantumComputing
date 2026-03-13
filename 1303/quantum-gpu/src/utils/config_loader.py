"""
Configuration loader — YAML parsing, device auto-detection, path resolution.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
import torch


# ──────────────────────────────────────────────────────────
#  Device detection
# ──────────────────────────────────────────────────────────

def get_device(requested: str = "auto") -> torch.device:
    """Return the best available device.

    Priority: CUDA → MPS → CPU (when *requested* is ``"auto"``).
    Quantum simulations always run on CPU, but classical pre/post
    layers can leverage CUDA (e.g. NVIDIA MX130) or MPS for acceleration.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


# ──────────────────────────────────────────────────────────
#  Config loading
# ──────────────────────────────────────────────────────────

def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config and resolve device + paths."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Auto-detect device
    exp = cfg.get("experiment", {})
    exp["device"] = str(get_device(exp.get("device", "auto")))
    cfg["experiment"] = exp

    # Resolve relative output paths w.r.t. project root
    project_root = path.parent.parent  # config/ → project root
    cfg["_project_root"] = str(project_root)

    paths = cfg.get("paths", {})
    for key, val in paths.items():
        abs_path = project_root / val
        abs_path.mkdir(parents=True, exist_ok=True)
        paths[key] = str(abs_path)
    cfg["paths"] = paths

    return cfg


def merge_configs(base: dict, override: dict) -> dict:
    """Deep-merge *override* into *base* (override wins)."""
    merged = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = merge_configs(merged[k], v)
        else:
            merged[k] = deepcopy(v)
    return merged
