from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    best_acc: float,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_acc": best_acc,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(
    path: Path, model: nn.Module, optimizer: optim.Optimizer, device: torch.device
) -> Tuple[int, float]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint.get("epoch", 0) + 1, checkpoint.get("best_acc", 0.0)
