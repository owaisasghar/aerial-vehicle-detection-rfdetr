#!/usr/bin/env python3
"""Audit the prepared one-class aerial vehicle COCO dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SPLITS = ("train", "valid", "test")


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    values.sort()
    result = {}
    for name, fraction in (
        ("min", 0), ("p01", .01), ("p05", .05), ("p10", .10),
        ("p50", .50), ("p90", .90), ("p95", .95), ("p99", .99), ("max", 1),
    ):
        position = fraction * (len(values) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            value = values[lower]
        else:
            value = values[lower] * (upper - position) + values[upper] * (position - lower)
        result[name] = round(value, 6)
    return result


def difference_hash(image: Image.Image) -> str:
    resized = image.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    pixels = list(getattr(resized, "get_flattened_data", resized.getdata)())
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(pixels[offset + column + 1] > pixels[offset + column])
    return f"{value:016x}"


def visual_comparison(left_path: Path, right_path: Path) -> dict[str, float]:
    pixels = []
    for path in (left_path, right_path):
        with Image.open(path) as image:
            resized = image.convert("L").resize((32, 24), Image.Resampling.BILINEAR)
            pixels.append(list(map(
                float, getattr(resized, "get_flattened_data", resized.getdata)()
            )))
    left, right = pixels
    left_mean, right_mean = statistics.mean(left), statistics.mean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    left_std = math.sqrt(sum(value * value for value in left_centered) / len(left))
    right_std = math.sqrt(sum(value * value for value in right_centered) / len(right))
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    correlation = (
        sum(a * b for a, b in zip(left_centered, right_centered)) / denominator
        if denominator else 0.0
    )
    error = sum(abs(a - b) for a, b in zip(left, right)) / len(left)
    return {
        "correlation": round(correlation, 6),
        "mean_absolute_error": round(error, 6),
        "minimum_contrast_stddev": round(min(left_std, right_std), 6),
    }


def source_name(filename: str) -> str:
    return filename.split("_", 1)[0]


def make_contact_sheet(
    root: Path,
    candidates: list[tuple[str, dict, list[dict]]],
    destination: Path,
    title: str,
    seed: int,
    count: int = 12,
) -> None:
    rng = random.Random(seed)
    selected = rng.sample(candidates, min(count, len(candidates)))
    cell_width, cell_height = 420, 315
    sheet = Image.new("RGB", (cell_width * 3, cell_height * 4 + 34), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill="black", font=ImageFont.load_default())
    for index, (split, metadata, annotations) in enumerate(selected):
        image_path = root / split / metadata["file_name"]
        with Image.open(image_path) as original:
            image = original.convert("RGB")
        scale = min(cell_width / image.width, (cell_height - 25) / image.height)
        resized = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
        x_offset = (index % 3) * cell_width
        y_offset = 34 + (index // 3) * cell_height
        sheet.paste(resized, (x_offset, y_offset + 25))
        cell_draw = ImageDraw.Draw(sheet)
        label = f"{split} | {metadata['file_name']} | boxes={len(annotations)}"
        cell_draw.text((x_offset + 3, y_offset + 5), label[:66], fill="black", font=ImageFont.load_default())
        for annotation in annotations:
            x, y, width, height = annotation["bbox"]
            cell_draw.rectangle(
                (
                    x_offset + x * scale,
                    y_offset + 25 + y * scale,
                    x_offset + (x + width) * scale,
                    y_offset + 25 + (y + height) * scale,
                ),
                outline=(255, 32, 32),
                width=2,
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("prepared/aerial_vehicle_rfdetr"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = args.dataset.resolve()

    report: dict = {"dataset": str(root), "splits": {}}
    exact_hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    perceptual_hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    visual_candidates: dict[str, list[tuple[str, dict, list[dict]]]] = defaultdict(list)

    for split in SPLITS:
        directory = root / split
        annotation_path = directory / "_annotations.coco.json"
        coco = json.loads(annotation_path.read_text())
        images = coco.get("images", [])
        annotations = coco.get("annotations", [])
        categories = coco.get("categories", [])
        image_by_id = {image["id"]: image for image in images}
        annotations_by_image: dict[int, list[dict]] = defaultdict(list)
        issues: Counter = Counter()
        issue_examples: dict[str, list] = defaultdict(list)

        if categories != [{"id": 1, "name": "vehicle", "supercategory": "vehicle"}]:
            issues["unexpected_categories"] += 1
        if len(image_by_id) != len(images):
            issues["duplicate_image_ids"] += len(images) - len(image_by_id)
        annotation_ids = [annotation["id"] for annotation in annotations]
        if len(set(annotation_ids)) != len(annotation_ids):
            issues["duplicate_annotation_ids"] += len(annotation_ids) - len(set(annotation_ids))

        widths: list[float] = []
        heights: list[float] = []
        relative_areas: list[float] = []
        boxes_per_image: list[float] = []
        source_images: Counter = Counter()
        source_boxes: Counter = Counter()
        negative_images: Counter = Counter()
        resolutions: Counter = Counter()
        small_medium_large: Counter = Counter()
        tiny_min_side: Counter = Counter()
        boundary_boxes = 0

        for annotation in annotations:
            image = image_by_id.get(annotation.get("image_id"))
            if image is None:
                issues["orphan_annotations"] += 1
                continue
            annotations_by_image[image["id"]].append(annotation)
            if annotation.get("category_id") != 1:
                issues["wrong_category"] += 1
            try:
                x, y, width, height = map(float, annotation["bbox"])
            except (KeyError, TypeError, ValueError):
                issues["malformed_boxes"] += 1
                continue
            if not all(math.isfinite(value) for value in (x, y, width, height)):
                issues["nonfinite_boxes"] += 1
                continue
            if width <= 0 or height <= 0:
                issues["nonpositive_boxes"] += 1
            if x < 0 or y < 0 or x + width > image["width"] + .001 or y + height > image["height"] + .001:
                issues["out_of_bounds_boxes"] += 1
            expected_area = width * height
            if abs(float(annotation.get("area", -1)) - expected_area) > .01:
                issues["incorrect_areas"] += 1
            source = source_name(image["file_name"])
            source_boxes[source] += 1
            widths.append(width)
            heights.append(height)
            relative_areas.append(expected_area / (image["width"] * image["height"]))
            if expected_area < 32**2:
                small_medium_large["small"] += 1
            elif expected_area < 96**2:
                small_medium_large["medium"] += 1
            else:
                small_medium_large["large"] += 1
            minimum_side = min(width, height)
            if minimum_side < 4:
                tiny_min_side["lt_4px"] += 1
            if minimum_side < 8:
                tiny_min_side["lt_8px"] += 1
            if minimum_side < 16:
                tiny_min_side["lt_16px"] += 1
            if x <= 0 or y <= 0 or x + width >= image["width"] or y + height >= image["height"]:
                boundary_boxes += 1

        annotation_names = {image["file_name"] for image in images}
        disk_names = {
            path.name for path in directory.iterdir()
            if path.is_file() and path.name != "_annotations.coco.json"
        }
        issues["missing_image_files"] += len(annotation_names - disk_names)
        issues["unreferenced_image_files"] += len(disk_names - annotation_names)

        for index, image_metadata in enumerate(images, 1):
            filename = image_metadata["file_name"]
            source = source_name(filename)
            source_images[source] += 1
            resolutions[f"{image_metadata['width']}x{image_metadata['height']}"] += 1
            image_annotations = annotations_by_image[image_metadata["id"]]
            boxes_per_image.append(float(len(image_annotations)))
            if not image_annotations:
                negative_images[source] += 1
            visual_candidates[source].append((split, image_metadata, image_annotations))
            path = directory / filename
            if not path.exists():
                continue
            try:
                file_hash = hashlib.sha256()
                with path.open("rb") as file:
                    for chunk in iter(lambda: file.read(1024 * 1024), b""):
                        file_hash.update(chunk)
                exact_hashes[file_hash.hexdigest()].append((split, filename))
                with Image.open(path) as image:
                    image.load()
                    if image.size != (image_metadata["width"], image_metadata["height"]):
                        issues["dimension_mismatches"] += 1
                        if len(issue_examples["dimension_mismatches"]) < 10:
                            issue_examples["dimension_mismatches"].append(
                                [filename, list(image.size), [image_metadata["width"], image_metadata["height"]]]
                            )
                    perceptual_hashes[difference_hash(image)].append((split, filename))
            except (OSError, ValueError) as error:
                issues["decode_errors"] += 1
                if len(issue_examples["decode_errors"]) < 10:
                    issue_examples["decode_errors"].append([filename, repr(error)])

            if index % 5000 == 0:
                print(f"{split}: decoded {index:,}/{len(images):,}", flush=True)

        report["splits"][split] = {
            "images": len(images),
            "annotations": len(annotations),
            "images_by_source": dict(source_images),
            "annotations_by_source": dict(source_boxes),
            "negative_images_by_source": dict(negative_images),
            "top_resolutions": resolutions.most_common(12),
            "boxes_per_image": quantiles(boxes_per_image),
            "box_width_pixels": quantiles(widths),
            "box_height_pixels": quantiles(heights),
            "box_relative_area": quantiles(relative_areas),
            "coco_size_buckets": dict(small_medium_large),
            "tiny_boxes_by_minimum_side": dict(tiny_min_side),
            "boxes_touching_image_boundary": boundary_boxes,
            "issues": {key: value for key, value in issues.items() if value},
            "issue_examples": dict(issue_examples),
        }
        print(f"{split}: decoded all {len(images):,} images", flush=True)

    def crossing(groups: dict[str, list[tuple[str, str]]]) -> list[list[tuple[str, str]]]:
        return [items for items in groups.values() if len({item[0] for item in items}) > 1]

    exact_cross_split = crossing(exact_hashes)
    perceptual_cross_split = crossing(perceptual_hashes)
    high_confidence_visual_matches = []
    for group in perceptual_cross_split:
        for left_index, (left_split, left_name) in enumerate(group):
            for right_split, right_name in group[left_index + 1:]:
                if left_split == right_split:
                    continue
                # Different source datasets can collide on a compact hash but
                # cannot be leaked copies of one another.
                if source_name(left_name) != source_name(right_name):
                    continue
                metrics = visual_comparison(
                    root / left_split / left_name, root / right_split / right_name
                )
                if (
                    metrics["minimum_contrast_stddev"] >= 10
                    and metrics["correlation"] >= .95
                    and metrics["mean_absolute_error"] <= 10
                ):
                    high_confidence_visual_matches.append({
                        "left": [left_split, left_name],
                        "right": [right_split, right_name],
                        **metrics,
                    })
    report["duplicates"] = {
        "exact_cross_split_groups": len(exact_cross_split),
        "exact_cross_split_images": sum(len(group) for group in exact_cross_split),
        "exact_cross_split_examples": exact_cross_split[:20],
        "identical_dhash_cross_split_groups": len(perceptual_cross_split),
        "identical_dhash_cross_split_images": sum(len(group) for group in perceptual_cross_split),
        "identical_dhash_cross_split_examples": perceptual_cross_split[:20],
        "high_confidence_visual_cross_split_pairs": high_confidence_visual_matches,
        "note": (
            "Identical dHash values are review candidates, not proof. High-confidence "
            "pairs additionally require same source, sufficient contrast, correlation "
            ">=0.95, and pixel MAE <=10."
        ),
    }

    output_dir = root / "audit"
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / "FULL_AUDIT_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2))
    for index, source in enumerate(sorted(visual_candidates)):
        positives = [item for item in visual_candidates[source] if item[2]]
        negatives = [item for item in visual_candidates[source] if not item[2]]
        make_contact_sheet(root, positives, output_dir / f"{source}_positive_samples.jpg",
                           f"{source}: random annotated positive images", args.seed + index)
        if negatives:
            make_contact_sheet(root, negatives, output_dir / f"{source}_negative_samples.jpg",
                               f"{source}: random images with zero retained vehicle boxes", args.seed + 100 + index)
    print(f"Audit report: {report_path}")


if __name__ == "__main__":
    main()
