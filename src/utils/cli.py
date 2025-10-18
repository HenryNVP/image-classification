from __future__ import annotations

import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_CONFIG = _ROOT / "configs" / "model" / "regnety_016.yaml"
DEFAULT_DATA_CONFIG = _ROOT / "configs" / "data.yaml"
DEFAULT_TRAIN_CONFIG = _ROOT / "configs" / "train.yaml"
DEFAULT_AUG_CONFIG = _ROOT / "configs" / "aug.yaml"


def parse_train_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an image classifier.")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
        help=f"Model configuration (default: {DEFAULT_MODEL_CONFIG})",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=DEFAULT_DATA_CONFIG,
        help=f"Dataset configuration (default: {DEFAULT_DATA_CONFIG})",
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=DEFAULT_TRAIN_CONFIG,
        help=f"Training configuration (default: {DEFAULT_TRAIN_CONFIG})",
    )
    parser.add_argument(
        "--aug-config",
        type=Path,
        default=DEFAULT_AUG_CONFIG,
        help=f"Augmentation configuration (default: {DEFAULT_AUG_CONFIG})",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Override checkpoint directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--device", type=str, default=None, help="Override device selection.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint.")
    parser.add_argument(
        "--amp",
        type=str,
        choices=("auto", "on", "off"),
        default="auto",
        help="Automatic mixed precision setting.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate training curves (loss/accuracy) at the end of training.",
    )
    return parser.parse_args()


def parse_validate_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a trained image classifier.")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
        help=f"Model configuration (default: {DEFAULT_MODEL_CONFIG})",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=DEFAULT_DATA_CONFIG,
        help=f"Dataset configuration (default: {DEFAULT_DATA_CONFIG})",
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=DEFAULT_TRAIN_CONFIG,
        help=f"Training configuration (default: {DEFAULT_TRAIN_CONFIG})",
    )
    parser.add_argument(
        "--aug-config",
        type=Path,
        default=DEFAULT_AUG_CONFIG,
        help=f"Augmentation configuration (default: {DEFAULT_AUG_CONFIG})",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint to evaluate.")
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="val",
        help="Dataset split to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--device", type=str, default=None, help="Override device selection.")
    return parser.parse_args()
