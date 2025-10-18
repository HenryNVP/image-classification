from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from src.config import load_configs, to_namespace
from src.data import ImageClassificationDataset
from src.engine import fit
from src.models import create_model
from src.utils import get_device, seed_all
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.cli import parse_train_args


def create_loader(
    split_cfg: Dict[str, str],
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    augment_cfg: Dict[str, Any] | None,
    train: bool,
) -> DataLoader:
    dataset = ImageClassificationDataset(
        split_cfg["images"],
        split_cfg["labels"],
        train=train,
        augment_config=augment_cfg,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def main() -> int:
    args = parse_train_args()

    config_paths = [
        args.train_config,
        args.data_config,
        args.model_config,
        args.aug_config,
    ]
    config = load_configs(config_paths)

    if "seed" in config:
        seed_all(int(config["seed"]))

    model_cfg = to_namespace(config["model"])
    train_cfg = config.get("train", {})
    data_cfg = config["data"]
    augment_cfg = config.get("augment")

    batch_size = args.batch_size or int(train_cfg.get("batch_size", 32))
    epochs = args.epochs or int(train_cfg.get("epochs", 10))
    lr = float(train_cfg.get("lr", 5e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    num_workers = int(train_cfg.get("num_workers", 4))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.1))
    log_interval = train_cfg.get("log_interval")
    early_cfg = train_cfg.get("early_stopping", {})
    patience = early_cfg.get("patience")
    patience = int(patience) if patience is not None else None
    min_delta = float(early_cfg.get("min_delta", 0.0)) if early_cfg else 0.0

    device_name = args.device or get_device(train_cfg.get("device_preference"))
    device = torch.device(device_name)

    if args.amp == "auto":
        amp = bool(train_cfg.get("amp", device.type == "cuda"))
    else:
        amp = args.amp == "on"

    output_dir = Path(args.output_dir or train_cfg.get("output_dir", "checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = create_loader(
        data_cfg["train"],
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        augment_cfg=augment_cfg,
        train=True,
    )
    val_loader = create_loader(
        data_cfg["val"],
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        augment_cfg=augment_cfg,
        train=False,
    )

    model = create_model(model_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        start_epoch, best_acc = load_checkpoint(args.resume, model, optimizer, device)
        print(
            f"Resumed from {args.resume} at epoch {start_epoch} "
            f"with best accuracy {best_acc:.4f}"
        )

    best, history = fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        epochs=epochs,
        device=device,
        save_dir=output_dir,
        amp=amp,
        start_epoch=start_epoch,
        best_accuracy=best_acc,
        label_smoothing=label_smoothing,
        log_interval=log_interval,
        early_stopping_patience=patience,
        early_stopping_min_delta=min_delta,
    )

    save_checkpoint(model, optimizer, epochs, best, output_dir / "last.pt")
    actual_epochs = history[-1]["epoch"] if history else 0
    print(
        f"Training complete after {actual_epochs} epochs. "
        f"Best validation accuracy: {best * 100:.2f}%"
    )

    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2))
    print(f"Saved training history to {history_path}")

    if args.plot and history:
        plot_history(history, output_dir)

    return 0


def plot_history(history: list[dict], output_dir: Path) -> None:
    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]
    train_acc = [entry["train_acc"] for entry in history]
    val_acc = [entry["val_acc"] for entry in history]

    try:
        plt.style.use("seaborn-v0_8")
    except OSError:
        pass
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_loss, label="Train")
    axes[0].plot(epochs, val_loss, label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy")
    axes[0].legend()

    axes[1].plot(epochs, [acc * 100 for acc in train_acc], label="Train")
    axes[1].plot(epochs, [acc * 100 for acc in val_acc], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Top-1 Accuracy (%)")
    axes[1].legend()

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "training_curves.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved training curves to {plot_path}")


if __name__ == "__main__":
    raise SystemExit(main())
