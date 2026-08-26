#!/usr/bin/env python3
"""Build RF-DETR's actual data loaders and fetch one batch without the model."""

from __future__ import annotations

from pathlib import Path

from rfdetr.config import RFDETRSmallConfig, TrainConfig
from rfdetr.datasets.aug_configs import AUG_AERIAL
from rfdetr.training import RFDETRDataModule


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "prepared" / "aerial_vehicle_rfdetr"


def main() -> None:
    model_config = RFDETRSmallConfig(resolution=768, positional_encoding_size=48)
    train_config = TrainConfig(
        dataset_dir=DATASET,
        dataset_file="roboflow",
        output_dir=ROOT / "runs" / "loader_validation",
        batch_size=1,
        grad_accum_steps=16,
        num_workers=0,
        aug_config=AUG_AERIAL,
        augmentation_backend="cpu",
        seed=42,
        accelerator="cpu",
    )
    data = RFDETRDataModule(model_config, train_config)
    data.setup("fit")
    train_batch = next(iter(data.train_dataloader()))
    valid_batch = next(iter(data.val_dataloader()))

    def describe(name: str, batch) -> None:
        samples, targets = batch
        tensor = samples.tensors if hasattr(samples, "tensors") else samples
        box_counts = [len(target["boxes"]) for target in targets]
        print(
            f"{name}: tensor_shape={tuple(tensor.shape)}, "
            f"targets_per_image={box_counts}"
        )

    describe("train", train_batch)
    describe("valid", valid_batch)
    print("RF-DETR DATA LOADER VALIDATION: PASS")


if __name__ == "__main__":
    main()
