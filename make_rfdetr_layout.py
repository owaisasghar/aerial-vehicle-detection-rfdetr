#!/usr/bin/env python3
"""Create RF-DETR's expected COCO directory layout without duplicating images."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "prepared" / "aerial_vehicle_coco"
DESTINATION = ROOT / "prepared" / "aerial_vehicle_rfdetr"
SPLITS = {"train": "train", "val": "valid", "test": "test"}


def main() -> None:
    if DESTINATION.exists():
        raise SystemExit(f"Refusing to overwrite existing directory: {DESTINATION}")

    for source_split, target_split in SPLITS.items():
        source_images = SOURCE / "images" / source_split
        source_annotations = SOURCE / "annotations" / f"instances_{source_split}.json"
        target_dir = DESTINATION / target_split
        target_dir.mkdir(parents=True)

        with source_annotations.open(encoding="utf-8") as file:
            coco = json.load(file)

        expected = {image["file_name"] for image in coco["images"]}
        present = {path.name for path in source_images.iterdir() if path.is_file()}
        if expected != present:
            raise RuntimeError(
                f"{source_split}: annotation/image mismatch: "
                f"missing={len(expected - present)}, extra={len(present - expected)}"
            )

        for name in sorted(expected):
            os.link(source_images / name, target_dir / name)

        shutil.copy2(source_annotations, target_dir / "_annotations.coco.json")
        print(
            f"{target_split}: {len(coco['images']):,} images, "
            f"{len(coco['annotations']):,} annotations"
        )

    print(f"RF-DETR dataset created at: {DESTINATION}")


if __name__ == "__main__":
    main()
