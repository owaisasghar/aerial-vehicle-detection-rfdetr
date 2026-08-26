#!/usr/bin/env python3
"""Build a leakage-aware, one-class RF-DETR dataset from the downloaded data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from PIL import Image


CATEGORY = {"id": 1, "name": "vehicle", "supercategory": "vehicle"}
VISDRONE_VEHICLE_IDS = {4, 5, 6, 9, 10}  # car, van, truck, bus, motor
DRONEVEHICLE_CLASSES = {
    "car", "truck", "truvk", "bus", "van", "feright_car", "feright car", "feright"
}
SPLIT_FRACTIONS = {"train": 0.8, "valid": 0.1, "test": 0.1}


@dataclass
class Record:
    source: Path
    materialized_source: Path
    output_name: str
    width: int
    height: int
    boxes: list[list[float]]
    source_dataset: str
    sequence_group: str
    raw_split: str
    force_train: bool = False
    crop_border: bool = False


def clip_box(x: float, y: float, width: float, height: float, iw: int, ih: int):
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(iw), x + width), min(float(ih), y + height)
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(x1, 3), round(y1, 3), round(x2 - x1, 3), round(y2 - y1, 3)]


def normalize_group(value: str) -> str:
    value = value.strip().lower().replace(" ", "")
    value = re.sub(r"\(1\)$", "", value)
    return re.sub(r"[^a-z0-9()_-]+", "_", value).strip("_") or "unknown"


def uavdt_group(stem: str) -> str:
    patterns = (
        r"^(DJI_\d+ \([^)]*\))_n\d+$",
        r"^(DJI_\d+\([^)]*\))_n\d+$",
        r"^(DJI_\d+)_n\d+$",
        r"^(DJI-\d+-\d+p)\d+$",
        r"^(det_ucystr__)\d+$",
        r"^(pavilion_\d+_)\d+$",
        r"^(dji_\d+_\d+m)\d+_\d+$",
        r"^(dji\d+)\d+_\d+$",
        r"^(djibridge\d*)_?\d+.*$",
        r"^(lim_?\d*)\D*\d+$",
        r"^(dhi_\d{4})\d+_\d+$",
        r"^(traina?)\d+$",
    )
    for pattern in patterns:
        match = re.match(pattern, stem, flags=re.IGNORECASE)
        if match:
            group = normalize_group(match.group(1))
            # These names are generic exports from one shared capture source.
            # Treating their numeric suffixes as separate sequences leaks nearly
            # identical frames across splits.
            if group.startswith("lim_") or group.startswith("dji_"):
                group = group.split("_", 1)[0]
            return f"uavdt:{group}"
    return f"uavdt:singleton:{normalize_group(stem)}"


def dronevehicle_group(root: ET.Element, raw_split: str) -> tuple[str, bool]:
    folder = (root.findtext("folder") or "").strip()
    filename = (root.findtext("filename") or "").strip().lstrip("/")
    original_path = (root.findtext("path") or "").strip()

    dated_flight = re.match(r"^(DJI_20\d{8})_\d+", filename, flags=re.IGNORECASE)
    if dated_flight:
        return f"dronevehicle:flight:{normalize_group(dated_flight.group(1))}", False

    generic_folders = {"a", "b", "image", "modala", "madala", "modela_1", "valimg"}
    normalized_folder = normalize_group(folder)
    if normalized_folder not in generic_folders and normalized_folder != "unknown":
        return f"dronevehicle:capture:{normalized_folder}", False

    if original_path:
        parts = [part for part in PureWindowsPath(original_path).parts if part not in {"\\", "/"}]
        if parts:
            parts = parts[:-1]
        removable = {
            "a", "b", "jpg", "jpeg", "image", "images", "modala", "madala", "modela_1"
        }
        while parts and normalize_group(parts[-1]) in removable:
            parts.pop()
        if parts:
            return f"dronevehicle:capture:{normalize_group(parts[-1])}", False

    # These records have only reused camera filenames such as DJI_0582.jpg.
    # Their capture sequence cannot be recovered reliably. Keep the downloaded
    # source split intact instead of mixing them into evaluation arbitrarily.
    if raw_split == "train":
        return "dronevehicle:unresolved-train-metadata", True
    return "dronevehicle:unresolved-validation-metadata", False


def read_visdrone(root: Path, stats: Counter) -> list[Record]:
    records = []
    for raw_split in ("train", "val"):
        base = root / f"VisDrone2019-DET-{raw_split}"
        for image_path in sorted((base / "images").glob("*")):
            if not image_path.is_file():
                continue
            try:
                with Image.open(image_path) as image:
                    image.load()
                    width, height = image.size
            except (OSError, ValueError):
                stats["visdrone_corrupt_images"] += 1
                continue
            boxes = []
            label_path = base / "annotations" / f"{image_path.stem}.txt"
            if not label_path.exists():
                stats["visdrone_missing_labels"] += 1
                continue
            for line in label_path.read_text(errors="replace").splitlines():
                values = line.strip().split(",")
                if len(values) < 8:
                    stats["visdrone_bad_rows"] += 1
                    continue
                try:
                    x, y, box_width, box_height, score, class_id, _, _ = map(int, values[:8])
                except ValueError:
                    stats["visdrone_bad_rows"] += 1
                    continue
                if score == 0 or class_id not in VISDRONE_VEHICLE_IDS:
                    continue
                box = clip_box(x, y, box_width, box_height, width, height)
                if box and box[2] >= 2 and box[3] >= 2:
                    boxes.append(box)
                    stats[f"visdrone_class_{class_id}_boxes"] += 1
                else:
                    stats["visdrone_invalid_boxes"] += 1
            sequence = image_path.name.split("_", 1)[0]
            records.append(Record(
                source=image_path,
                materialized_source=image_path,
                output_name=f"visdrone_{raw_split}_{image_path.name}",
                width=width,
                height=height,
                boxes=boxes,
                source_dataset="visdrone",
                sequence_group=f"visdrone:{normalize_group(sequence)}",
                raw_split=raw_split,
            ))
    return records


def read_uavdt(root: Path, stats: Counter) -> list[Record]:
    records = []
    for raw_split in ("train", "val", "test"):
        base = root / raw_split
        for image_path in sorted((base / "images").glob("*.jpg")):
            try:
                with Image.open(image_path) as image:
                    image.load()
                    width, height = image.size
            except (OSError, ValueError):
                stats["uavdt_corrupt_images"] += 1
                continue
            label_path = base / "labels" / f"{image_path.stem}.txt"
            if not label_path.exists():
                stats["uavdt_missing_labels"] += 1
                continue
            boxes = []
            for line in label_path.read_text(errors="replace").splitlines():
                values = line.split()
                if len(values) != 5:
                    stats["uavdt_bad_rows"] += 1
                    continue
                try:
                    class_id, cx, cy, box_width, box_height = map(float, values)
                except ValueError:
                    stats["uavdt_bad_rows"] += 1
                    continue
                box = clip_box(
                    (cx - box_width / 2) * width,
                    (cy - box_height / 2) * height,
                    box_width * width,
                    box_height * height,
                    width,
                    height,
                )
                if box and box[2] >= 2 and box[3] >= 2:
                    boxes.append(box)
                    stats[f"uavdt_source_class_{int(class_id)}_boxes"] += 1
                else:
                    stats["uavdt_invalid_boxes"] += 1
            records.append(Record(
                source=image_path,
                materialized_source=image_path,
                output_name=f"uavdt_{raw_split}_{image_path.name}",
                width=width,
                height=height,
                boxes=boxes,
                source_dataset="uavdt",
                sequence_group=uavdt_group(image_path.stem),
                raw_split=raw_split,
            ))
    return records


def read_dronevehicle(root: Path, reuse_root: Path, stats: Counter) -> list[Record]:
    records = []
    for raw_split, image_folder, label_folder in (
        ("train", "trainimg", "trainlabel"),
        ("val", "valimg", "vallabel"),
    ):
        image_dir = root / raw_split / image_folder
        label_dir = root / raw_split / label_folder
        old_split = "train" if raw_split == "train" else "val"
        for image_path in sorted(image_dir.glob("*.jpg")):
            try:
                with Image.open(image_path) as image:
                    image.load()
            except (OSError, ValueError):
                stats["dronevehicle_corrupt_images"] += 1
                continue
            label_path = label_dir / f"{image_path.stem}.xml"
            if not label_path.exists():
                stats["dronevehicle_missing_labels"] += 1
                continue
            try:
                tree = ET.parse(label_path)
            except ET.ParseError:
                stats["dronevehicle_bad_xml"] += 1
                continue
            group, force_train = dronevehicle_group(tree.getroot(), raw_split)
            boxes = []
            for obj in tree.findall(".//object"):
                name = (obj.findtext("name") or "").strip().lower()
                if name not in DRONEVEHICLE_CLASSES:
                    stats[f"dronevehicle_skipped_class:{name}"] += 1
                    continue
                polygon = obj.find("polygon")
                bounding_box = obj.find("bndbox")
                if polygon is not None:
                    try:
                        xs = [float(polygon.findtext(f"x{index}")) - 100 for index in range(1, 5)]
                        ys = [float(polygon.findtext(f"y{index}")) - 100 for index in range(1, 5)]
                    except (TypeError, ValueError):
                        stats["dronevehicle_bad_polygon"] += 1
                        continue
                    box = clip_box(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys), 640, 512)
                elif bounding_box is not None:
                    try:
                        x1 = float(bounding_box.findtext("xmin")) - 100
                        y1 = float(bounding_box.findtext("ymin")) - 100
                        x2 = float(bounding_box.findtext("xmax")) - 100
                        y2 = float(bounding_box.findtext("ymax")) - 100
                    except (TypeError, ValueError):
                        stats["dronevehicle_bad_bndbox"] += 1
                        continue
                    box = clip_box(x1, y1, x2 - x1, y2 - y1, 640, 512)
                    stats["dronevehicle_bndbox_converted"] += 1
                else:
                    stats["dronevehicle_point_only_skipped"] += 1
                    continue
                if box and box[2] >= 2 and box[3] >= 2:
                    boxes.append(box)
                else:
                    stats["dronevehicle_invalid_boxes"] += 1

            output_name = f"dronevehicle_{raw_split}_{image_path.name}"
            reusable = reuse_root / old_split / output_name
            records.append(Record(
                source=image_path,
                materialized_source=reusable if reusable.exists() else image_path,
                output_name=output_name,
                width=640,
                height=512,
                boxes=boxes,
                source_dataset="dronevehicle",
                sequence_group=group,
                raw_split=raw_split,
                force_train=force_train,
                crop_border=not reusable.exists(),
            ))
    return records


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_signature(path: Path) -> tuple[str, list[float]]:
    with Image.open(path) as image:
        grayscale = image.convert("L")
        hash_image = grayscale.resize((9, 8), Image.Resampling.BILINEAR)
        pixels = list(
            getattr(hash_image, "get_flattened_data", hash_image.getdata)()
        )
        review_image = grayscale.resize((32, 24), Image.Resampling.BILINEAR)
        review_pixels = list(
            map(float, getattr(review_image, "get_flattened_data", review_image.getdata)())
        )
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(
                pixels[offset + column + 1] > pixels[offset + column]
            )
    return f"{value:016x}", review_pixels


def pixel_similarity(left: list[float], right: list[float]) -> tuple[float, float, float, float]:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    left_variance = sum(value * value for value in left_centered) / len(left)
    right_variance = sum(value * value for value in right_centered) / len(right)
    left_std = math.sqrt(left_variance)
    right_std = math.sqrt(right_variance)
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    correlation = (
        sum(a * b for a, b in zip(left_centered, right_centered)) / denominator
        if denominator else 0.0
    )
    mean_absolute_error = sum(abs(a - b) for a, b in zip(left, right)) / len(left)
    return correlation, mean_absolute_error, left_std, right_std


def remove_exact_duplicates(records: list[Record], stats: Counter) -> tuple[list[Record], list[dict]]:
    by_hash: dict[str, list[Record]] = defaultdict(list)
    for index, record in enumerate(records, 1):
        # DroneVehicle reusable files are already cropped exactly as this build needs.
        # The rare no-reuse case is not deduplicated before materialization.
        if record.crop_border:
            key = f"uncropped:{record.source_dataset}:{record.output_name}"
        else:
            key = file_sha256(record.materialized_source)
        by_hash[key].append(record)
        if index % 5000 == 0:
            print(f"Hashed {index:,}/{len(records):,} source images", flush=True)

    unique_records = []
    duplicate_log = []
    for digest, group in by_hash.items():
        if len(group) == 1:
            unique_records.append(group[0])
            continue
        # Prefer the most complete annotation, then a stable name.
        ordered = sorted(group, key=lambda record: (-len(record.boxes), record.output_name))
        kept = ordered[0]
        unique_records.append(kept)
        removed = ordered[1:]
        stats["exact_duplicate_images_removed"] += len(removed)
        duplicate_log.append({
            "sha256": digest,
            "kept": kept.output_name,
            "kept_boxes": len(kept.boxes),
            "removed": [
                {"file": record.output_name, "boxes": len(record.boxes)} for record in removed
            ],
        })
    return unique_records, duplicate_log


def merge_perceptually_identical_groups(
    records: list[Record], stats: Counter
) -> list[dict]:
    """Merge same-source sequence groups connected by an identical dHash."""

    parent: dict[str, str] = {record.sequence_group: record.sequence_group for record in records}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    by_fingerprint: dict[tuple[str, str], list[tuple[Record, list[float]]]] = defaultdict(list)
    for index, record in enumerate(records, 1):
        fingerprint, pixels = perceptual_signature(record.materialized_source)
        by_fingerprint[(record.source_dataset, fingerprint)].append((record, pixels))
        if index % 5000 == 0:
            print(f"Perceptually screened {index:,}/{len(records):,} images", flush=True)

    merge_log = []
    for (source, fingerprint), candidates in by_fingerprint.items():
        groups = {record.sequence_group for record, _ in candidates}
        if len(groups) < 2:
            continue
        for left_index, (left_record, left_pixels) in enumerate(candidates):
            for right_record, right_pixels in candidates[left_index + 1:]:
                if left_record.sequence_group == right_record.sequence_group:
                    continue
                correlation, error, left_std, right_std = pixel_similarity(
                    left_pixels, right_pixels
                )
                # A dHash collision by itself is unreliable for near-black frames.
                # Merge only high-contrast, highly correlated visual matches.
                if min(left_std, right_std) < 10 or correlation < .95 or error > 10:
                    continue
                union(left_record.sequence_group, right_record.sequence_group)
                merge_log.append({
                    "source_dataset": source,
                    "difference_hash": fingerprint,
                    "left_image": left_record.output_name,
                    "right_image": right_record.output_name,
                    "left_sequence_group": left_record.sequence_group,
                    "right_sequence_group": right_record.sequence_group,
                    "correlation": round(correlation, 6),
                    "mean_absolute_error": round(error, 6),
                    "minimum_contrast_stddev": round(min(left_std, right_std), 6),
                })

    force_train_components = {
        find(record.sequence_group) for record in records if record.force_train
    }
    changed_groups = set()
    for record in records:
        original = record.sequence_group
        merged = find(original)
        record.sequence_group = merged
        if merged != original:
            changed_groups.add(original)
        if merged in force_train_components:
            record.force_train = True
    stats["perceptual_fingerprint_merge_events"] = len(merge_log)
    stats["sequence_groups_merged_by_fingerprint"] = len(changed_groups)
    return merge_log


def assign_groups(records: list[Record], seed: int) -> tuple[dict[str, str], dict]:
    assignments: dict[str, str] = {}
    report: dict = {}
    rng = random.Random(seed)
    for source in sorted({record.source_dataset for record in records}):
        source_records = [record for record in records if record.source_dataset == source]
        groups: dict[str, list[Record]] = defaultdict(list)
        forced_records = []
        for record in source_records:
            if record.force_train:
                forced_records.append(record)
            else:
                groups[record.sequence_group].append(record)

        total_images = len(source_records)
        total_boxes = sum(len(record.boxes) for record in source_records)
        target_images = {split: total_images * fraction for split, fraction in SPLIT_FRACTIONS.items()}
        target_boxes = {split: total_boxes * fraction for split, fraction in SPLIT_FRACTIONS.items()}
        current_images = Counter({"train": len(forced_records), "valid": 0, "test": 0})
        current_boxes = Counter({"train": sum(len(record.boxes) for record in forced_records), "valid": 0, "test": 0})

        sortable = []
        for group_name, group_records in groups.items():
            sortable.append((
                len(group_records),
                sum(len(record.boxes) for record in group_records),
                rng.random(),
                group_name,
                group_records,
            ))
        sortable.sort(key=lambda item: (-item[0], -item[1], item[2]))

        for image_count, box_count, _, group_name, _ in sortable:
            scores = {}
            for split in SPLIT_FRACTIONS:
                image_ratio = (current_images[split] + image_count) / max(target_images[split], 1)
                box_ratio = (current_boxes[split] + box_count) / max(target_boxes[split], 1)
                scores[split] = image_ratio + box_ratio
            chosen = min(scores, key=lambda split: (scores[split], split))
            assignments[group_name] = chosen
            current_images[chosen] += image_count
            current_boxes[chosen] += box_count

        report[source] = {
            "total_images": total_images,
            "total_boxes": total_boxes,
            "assignable_sequence_groups": len(groups),
            "forced_train_images_unresolved_metadata": len(forced_records),
            "images_by_split": dict(current_images),
            "boxes_by_split": dict(current_boxes),
            "groups_by_split": dict(Counter(assignments[group] for group in groups)),
        }
    return assignments, report


def materialize(
    records: list[Record],
    assignments: dict[str, str],
    output: Path,
    seed: int,
    stats: Counter,
) -> None:
    by_split: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        split = "train" if record.force_train else assignments[record.sequence_group]
        by_split[split].append(record)

    for split_index, split in enumerate(("train", "valid", "test")):
        split_dir = output / split
        split_dir.mkdir(parents=True)
        split_records = by_split[split]
        random.Random(seed + split_index).shuffle(split_records)
        images = []
        annotations = []
        annotation_id = 1
        for image_id, record in enumerate(split_records, 1):
            destination = split_dir / record.output_name
            if record.crop_border:
                with Image.open(record.source) as image:
                    image.convert("RGB").crop((100, 100, 740, 612)).save(
                        destination, quality=95, subsampling=0
                    )
            else:
                try:
                    os.link(record.materialized_source, destination)
                except OSError:
                    shutil.copy2(record.materialized_source, destination)
            images.append({
                "id": image_id,
                "file_name": record.output_name,
                "width": record.width,
                "height": record.height,
                "source_dataset": record.source_dataset,
                "sequence_group": record.sequence_group,
                "raw_split": record.raw_split,
            })
            for box in record.boxes:
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": box,
                    "area": round(box[2] * box[3], 3),
                    "iscrowd": 0,
                })
                annotation_id += 1
            stats[f"{split}_images"] += 1
            stats[f"{split}_boxes"] += len(record.boxes)
            stats[f"{split}_{record.source_dataset}_images"] += 1
            stats[f"{split}_{record.source_dataset}_boxes"] += len(record.boxes)
            if not record.boxes:
                stats[f"{split}_negative_images"] += 1
                stats[f"{split}_{record.source_dataset}_negative_images"] += 1

        payload = {
            "info": {
                "description": "Leakage-aware unified aerial motor-vehicle dataset v2",
                "seed": seed,
                "split_policy": "capture-sequence-grouped 80/10/10",
            },
            "licenses": [],
            "categories": [CATEGORY],
            "images": images,
            "annotations": annotations,
        }
        (split_dir / "_annotations.coco.json").write_text(
            json.dumps(payload, separators=(",", ":"))
        )
        print(f"{split}: {len(images):,} images, {len(annotations):,} boxes", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=Path, default=Path("datasets"))
    parser.add_argument("--reuse-crops", type=Path,
                        default=Path("prepared/aerial_vehicle_coco/images"))
    parser.add_argument("--output", type=Path,
                        default=Path("prepared/aerial_vehicle_rfdetr_v2"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {args.output}")

    stats = Counter()
    records = []
    records.extend(read_visdrone(args.datasets / "visdrone" / "raw", stats))
    records.extend(read_uavdt(args.datasets / "uavdt" / "raw" / "UAVDT", stats))
    records.extend(read_dronevehicle(
        args.datasets / "dronevehicle" / "raw", args.reuse_crops, stats
    ))
    stats["source_images_before_deduplication"] = len(records)
    print(f"Loaded {len(records):,} valid source images", flush=True)

    records, duplicate_log = remove_exact_duplicates(records, stats)
    stats["unique_images_after_deduplication"] = len(records)
    perceptual_merge_log = merge_perceptually_identical_groups(records, stats)
    assignments, split_report = assign_groups(records, args.seed)
    materialize(records, assignments, args.output, args.seed, stats)

    (args.output / "PREPARATION_REPORT.json").write_text(
        json.dumps(dict(sorted(stats.items())), indent=2)
    )
    (args.output / "SPLIT_REPORT.json").write_text(json.dumps(split_report, indent=2))
    (args.output / "EXACT_DUPLICATES_REMOVED.json").write_text(
        json.dumps(duplicate_log, indent=2)
    )
    (args.output / "PERCEPTUAL_GROUP_MERGES.json").write_text(
        json.dumps(perceptual_merge_log, indent=2)
    )
    (args.output / "SEQUENCE_ASSIGNMENTS.json").write_text(
        json.dumps(dict(sorted(assignments.items())), indent=2)
    )
    print(json.dumps(dict(sorted(stats.items())), indent=2))


if __name__ == "__main__":
    main()
