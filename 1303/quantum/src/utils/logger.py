"""
Logging utilities — TensorBoard + Python logging.
"""

from __future__ import annotations

import logging
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class QuantumLogger:
    """Combined Python logger + TensorBoard writer.

    Parameters
    ----------
    name : str
        Logger name (also used as TensorBoard run name).
    log_dir : str | Path
        TensorBoard log directory.
    log_file : str | Path | None
        Optional file path for Python logging.
    use_tensorboard : bool
        Enable TensorBoard writing.
    """

    def __init__(
        self,
        name: str = "quantum",
        log_dir: str | Path = "outputs/tensorboard",
        log_file: str | Path | None = None,
        use_tensorboard: bool = True,
    ) -> None:
        self.name = name
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # Python logger
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.INFO)
        if not self._logger.handlers:
            fmt = logging.Formatter(
                "[%(asctime)s] %(name)s — %(levelname)s — %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            ch = logging.StreamHandler()
            ch.setFormatter(fmt)
            self._logger.addHandler(ch)

            if log_file is not None:
                fh = logging.FileHandler(log_file)
                fh.setFormatter(fmt)
                self._logger.addHandler(fh)

        # TensorBoard
        self._writer = None
        if use_tensorboard:
            self._writer = SummaryWriter(log_dir=str(self._log_dir / name))

    # ── Scalar logging ───────────────────────────────────
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer:
            self._writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int) -> None:
        if self._writer:
            self._writer.add_scalars(main_tag, tag_scalar_dict, step)

    # ── Training logging ─────────────────────────────────
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float | None = None,
        val_acc: float | None = None,
    ) -> None:
        msg = f"Epoch {epoch:4d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f}"
        if val_loss is not None:
            msg += f" | val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        self._logger.info(msg)

        self.log_scalar("loss/train", train_loss, epoch)
        self.log_scalar("accuracy/train", train_acc, epoch)
        if val_loss is not None:
            self.log_scalar("loss/val", val_loss, epoch)
        if val_acc is not None:
            self.log_scalar("accuracy/val", val_acc, epoch)

    # ── General logging ──────────────────────────────────
    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def warning(self, msg: str) -> None:
        self._logger.warning(msg)

    def error(self, msg: str) -> None:
        self._logger.error(msg)

    # ── Cleanup ──────────────────────────────────────────
    def close(self) -> None:
        if self._writer:
            self._writer.close()
