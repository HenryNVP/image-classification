from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    class_name: str


class ImageClassificationDataset(Dataset[tuple]):
    """Dataset that reads image paths and labels from a CSV file."""

    def __init__(
        self,
        image_dir: str | Path,
        labels_csv: str | Path,
        *,
        train: bool = False,
        transform: Callable | None = None,
        augment_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.labels_csv = Path(labels_csv)
        self.train = train
        self.transform = transform or build_transforms(train=train, augment_config=augment_config)

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.labels_csv.exists():
            raise FileNotFoundError(f"Label file not found: {self.labels_csv}")

        self.samples: List[Sample] = []
        self._class_names: List[str] = []

        self._load_labels()

    def _load_labels(self) -> None:
        class_names: set[str] = set()

        with open(self.labels_csv, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                image_name = row["image"]
                class_name = row["class_name"]
                class_id_raw = int(row["class_id"])
                label = class_id_raw - 1  # Convert to zero-based index.

                image_path = self.image_dir / image_name
                if not image_path.exists():
                    raise FileNotFoundError(f"Image missing on disk: {image_path}")

                self.samples.append(Sample(image_path, label, class_name))
                class_names.add(class_name)

        self._class_names = sorted(class_names)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = image.convert("RGB")
            if self.transform:
                image = self.transform(image)
        return image, sample.label

    @property
    def classes(self) -> Sequence[str]:
        return self._class_names

    @property
    def num_classes(self) -> int:
        return len(self._class_names)


def build_transforms(
    *,
    train: bool,
    augment_config: Optional[Dict[str, Any]] = None,
) -> transforms.Compose:
    cfg = augment_config or {}
    ops: List[Any] = []

    size = cfg.get("size")

    if train:
        train_cfg = cfg.get("train", {})
        rrc_cfg = train_cfg.get("random_resized_crop")
        if rrc_cfg:
            if isinstance(rrc_cfg, dict):
                crop_size = int(rrc_cfg.get("size", 224))
                scale = tuple(rrc_cfg.get("scale", (0.8, 1.0)))
                ratio = tuple(rrc_cfg.get("ratio", (3 / 4, 4 / 3)))
                ops.append(
                    transforms.RandomResizedCrop(
                        crop_size,
                        scale=scale,  # type: ignore[arg-type]
                        ratio=ratio,  # type: ignore[arg-type]
                    )
                )
            else:
                ops.append(transforms.RandomResizedCrop(int(rrc_cfg)))
        elif size:
            ops.append(transforms.Resize((size, size)))

        if train_cfg.get("random_horizontal_flip", True):
            ops.append(transforms.RandomHorizontalFlip())
        jitter_cfg = train_cfg.get("color_jitter")
        if jitter_cfg:
            ops.append(transforms.ColorJitter(**jitter_cfg))
    else:
        eval_cfg = cfg.get("eval", {})
        if size:
            ops.append(transforms.Resize((size, size)))
        if eval_cfg.get("center_crop") and size:
            ops.append(transforms.CenterCrop(size))

    ops.append(transforms.ToTensor())

    norm_cfg = cfg.get("normalize", {})
    mean = tuple(norm_cfg.get("mean", IMAGENET_MEAN))
    std = tuple(norm_cfg.get("std", IMAGENET_STD))
    ops.append(transforms.Normalize(mean, std))

    return transforms.Compose(ops)
