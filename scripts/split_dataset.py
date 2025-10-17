from __future__ import annotations

import re
import csv
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_IMAGES_DIR = ROOT_DIR / "data" / "raw" / "images"
ANNOTATIONS_DIR = ROOT_DIR / "data" / "raw" / "annotations"

TRAIN_LIST = ANNOTATIONS_DIR / "trainval.txt"
TEST_LIST = ANNOTATIONS_DIR / "test.txt"
METADATA_LIST = ANNOTATIONS_DIR / "list.txt"

TRAIN_DIR = ROOT_DIR / "data" / "train"
VAL_DIR = ROOT_DIR / "data" / "val"
TEST_DIR = ROOT_DIR / "data" / "test"
TRAIN_LABELS = ROOT_DIR / "data" / "train_labels.csv"
VAL_LABELS = ROOT_DIR / "data" / "val_labels.csv"
TEST_LABELS = ROOT_DIR / "data" / "test_labels.csv"

IMAGE_SIZE = (224, 224)


def read_metadata() -> Dict[str, Dict[str, int | str]]:
    metadata: Dict[str, Dict[str, int | str]] = {}
    pattern = re.compile(r"_[0-9]+$")

    with open(METADATA_LIST, "r", encoding="utf-8") as file:
        for line in file:
            if not line or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            name, class_id, species, breed_id = parts[:4]
            class_name = pattern.sub("", name)
            metadata[name] = {
                "class_id": int(class_id),
                "species": int(species),
                "breed_id": int(breed_id),
                "class_name": class_name,
            }
    return metadata


def read_split(file_path: Path) -> List[str]:
    names: List[str] = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            names.append(stripped.split()[0])
    return names


def ensure_clean_dir(directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def split_test_set(test_names: Iterable[str], metadata: Dict[str, Dict[str, int | str]]) -> Tuple[List[str], List[str]]:
    per_class: Dict[str, List[str]] = defaultdict(list)
    for name in test_names:
        class_name = str(metadata[name]["class_name"])
        per_class[class_name].append(name)

    val_split: List[str] = []
    remaining_test: List[str] = []

    for class_name, entries in per_class.items():
        entries.sort()
        mid = len(entries) // 2
        val_split.extend(entries[:mid])
        remaining_test.extend(entries[mid:])

    return val_split, remaining_test


def resize_and_save(image_name: str, destination: Path) -> None:
    source = RAW_IMAGES_DIR / f"{image_name}.jpg"
    if not source.exists():
        raise FileNotFoundError(f"Missing image file: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(source) as img:
        resized = img.convert("RGB").resize(IMAGE_SIZE, Image.BILINEAR)
        resized.save(destination, format="JPEG", quality=95)


def copy_split(
    names: Iterable[str], metadata: Dict[str, Dict[str, int | str]], destination_root: Path
) -> None:
    for name in names:
        target = destination_root / f"{name}.jpg"
        resize_and_save(name, target)


def write_split_metadata(
    names: Iterable[str],
    metadata: Dict[str, Dict[str, int | str]],
    destination: Path,
) -> None:
    with open(destination, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "class_name", "class_id", "species", "breed_id"])
        for name in names:
            info = metadata[name]
            writer.writerow(
                [
                    f"{name}.jpg",
                    info["class_name"],
                    info["class_id"],
                    info["species"],
                    info["breed_id"],
                ]
            )


def main() -> None:
    metadata = read_metadata()
    train_names = read_split(TRAIN_LIST)
    test_names = read_split(TEST_LIST)

    for directory in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        ensure_clean_dir(directory)

    val_names, remaining_test_names = split_test_set(test_names, metadata)

    copy_split(train_names, metadata, TRAIN_DIR)
    copy_split(val_names, metadata, VAL_DIR)
    copy_split(remaining_test_names, metadata, TEST_DIR)

    write_split_metadata(train_names, metadata, TRAIN_LABELS)
    write_split_metadata(val_names, metadata, VAL_LABELS)
    write_split_metadata(remaining_test_names, metadata, TEST_LABELS)

    print(f"Prepared {len(train_names)} training images in {TRAIN_DIR}")
    print(f"Prepared {len(val_names)} validation images in {VAL_DIR}")
    print(f"Prepared {len(remaining_test_names)} test images in {TEST_DIR}")


if __name__ == "__main__":
    main()
