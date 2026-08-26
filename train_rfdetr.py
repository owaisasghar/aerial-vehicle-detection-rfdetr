#!/usr/bin/env python3
"""Train RF-DETR Small on the audited aerial-vehicle dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from rfdetr import RFDETRSmall
from rfdetr.datasets.aug_configs import AUG_AERIAL


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "prepared" / "aerial_vehicle_rfdetr",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs" / "rfdetr_small_aerial_vehicle",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=768)
    # RF-DETR 1.9.2 maps an indexed device such as ``cuda:0`` to ``[0]`` and
    # then incorrectly treats that list as a string in its trainer builder.
    # ``cuda`` selects the only visible GPU without triggering that code path.
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    annotation = args.dataset / "train" / "_annotations.coco.json"
    if not annotation.exists():
        raise SystemExit(f"RF-DETR COCO dataset not found: {annotation}")

    model = RFDETRSmall()
    training_options = {
        "dataset_dir": str(args.dataset),
        "output_dir": str(args.output),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "resolution": args.resolution,
        "device": args.device,
        "lr": 1e-4,
        "lr_encoder": 1.5e-4,
        "early_stopping": True,
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 0.001,
        "checkpoint_interval": 5,
        "use_ema": True,
        "run_test": True,
        "aug_config": AUG_AERIAL,
        "augmentation_backend": "cpu",
        "seed": 42,
        "num_workers": 4,
        "save_dataset_grids": True,
        "notes": {
            "dataset": "VisDrone + UAVDT + DroneVehicle, leakage-aware v2",
            "task": "single-class aerial vehicle detection",
        },
    }
    if args.resume:
        training_options["resume"] = str(args.resume)
    model.train(**training_options)


if __name__ == "__main__":
    main()
