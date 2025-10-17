from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import load_configs, to_namespace
from src.data import ImageClassificationDataset
from src.models import create_model
from src.utils import get_device
from src.utils.cli import parse_validate_args


def create_loader(
    split_cfg: Dict[str, str],
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    augment_cfg: Dict[str, Any] | None,
) -> DataLoader:
    dataset = ImageClassificationDataset(
        split_cfg["images"],
        split_cfg["labels"],
        train=False,
        augment_config=augment_cfg,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def load_model_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)


def main() -> int:
    args = parse_validate_args()

    config_paths = [
        args.train_config,
        args.data_config,
        args.model_config,
        args.aug_config,
    ]
    config = load_configs(config_paths)

    train_cfg = config.get("train", {})
    data_cfg = config["data"]
    augment_cfg = config.get("augment")

    batch_size = args.batch_size or int(train_cfg.get("batch_size", 32))
    num_workers = int(train_cfg.get("num_workers", 4))

    device_name = args.device or get_device(train_cfg.get("device_preference"))
    device = torch.device(device_name)

    loader = create_loader(
        data_cfg[args.split],
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        augment_cfg=augment_cfg,
    )

    model_cfg = to_namespace(config["model"])
    model = create_model(model_cfg).to(device)
    load_model_checkpoint(model, args.checkpoint, device)
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            total_correct += outputs.argmax(dim=1).eq(targets).sum().item()
            total_samples += inputs.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)

    print(
        f"Split '{args.split}' - Loss: {avg_loss:.4f} "
        f"Accuracy: {accuracy * 100:.2f}% ({total_correct}/{total_samples})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
