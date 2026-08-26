#!/usr/bin/env python3
"""Remove exact duplicate training images that also occur in val/test."""

import hashlib
import json
from pathlib import Path


ROOT = Path("prepared/aerial_vehicle_coco")


def digest(path: Path) -> str:
    value = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


protected = set()
for split in ("val", "test"):
    for path in (ROOT / "images" / split).iterdir():
        protected.add(digest(path))

annotation_path = ROOT / "annotations" / "instances_train.json"
data = json.loads(annotation_path.read_text())
remove_ids = set()
removed_names = []
for image in data["images"]:
    path = ROOT / "images" / "train" / image["file_name"]
    if digest(path) in protected:
        remove_ids.add(image["id"])
        removed_names.append(image["file_name"])
        path.unlink()

data["images"] = [image for image in data["images"] if image["id"] not in remove_ids]
data["annotations"] = [ann for ann in data["annotations"] if ann["image_id"] not in remove_ids]
annotation_path.write_text(json.dumps(data, separators=(",", ":")))
(ROOT / "REMOVED_CROSS_SPLIT_DUPLICATES.txt").write_text("\n".join(removed_names) + "\n")
print(f"Removed {len(removed_names)} training duplicates:")
print("\n".join(removed_names))
