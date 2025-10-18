from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
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
) -> Tuple[ImageClassificationDataset, DataLoader]:
    dataset = ImageClassificationDataset(
        split_cfg["images"],
        split_cfg["labels"],
        train=False,
        augment_config=augment_cfg,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    return dataset, loader


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

    dataset, loader = create_loader(
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
    total_samples = 0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            probs = torch.softmax(outputs, dim=1)

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            all_probs.append(probs.cpu())
            all_targets.append(targets.cpu())

    if total_samples == 0:
        print(f"Split '{args.split}' is empty. No metrics computed.")
        return 0

    probs = torch.cat(all_probs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    preds = probs.argmax(dim=1)

    avg_loss = total_loss / total_samples

    targets_np = targets.numpy()
    probs_np = probs.numpy()
    preds_np = preds.numpy()

    accuracy = accuracy_score(targets_np, preds_np)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        targets_np, preds_np, average="macro", zero_division=0
    )
    precision_per, recall_per, f1_per, support = precision_recall_fscore_support(
        targets_np, preds_np, average=None, zero_division=0
    )

    try:
        roc_auc = roc_auc_score(targets_np, probs_np, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc = None

    class_names = list(getattr(dataset, "classes", []))
    if not class_names:
        class_names = [str(i) for i in range(probs_np.shape[1])]

    per_class = []
    for idx, name in enumerate(class_names):
        per_class.append(
            {
                "class": name,
                "precision": float(precision_per[idx]),
                "recall": float(recall_per[idx]),
                "f1": float(f1_per[idx]),
                "support": int(support[idx]),
            }
        )

    metrics = {
        "split": args.split,
        "samples": int(total_samples),
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "roc_auc_macro": float(roc_auc) if roc_auc is not None else None,
        "per_class": per_class,
    }

    print(
        f"Split '{args.split}' - Loss: {avg_loss:.4f} "
        f"Acc: {accuracy * 100:.2f}% MacroP: {precision_macro * 100:.2f}% "
        f"MacroR: {recall_macro * 100:.2f}% MacroF1: {f1_macro * 100:.2f}% "
        + (f"ROC AUC: {roc_auc * 100:.2f}%" if roc_auc is not None else "ROC AUC: N/A")
    )

    output_path = args.output or args.checkpoint.with_suffix(".metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Saved metrics to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
