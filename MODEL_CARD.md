---
tags:
  - object-detection
  - aerial-imagery
  - vehicle-detection
  - rfdetr
  - pytorch
datasets:
  - VisDrone/VisDrone2019-DET
library_name: rfdetr
pipeline_tag: object-detection
---

# RF-DETR Small — Aerial Vehicle Detection

Fine-tuned RF-DETR Small checkpoint for detecting vehicles in drone and
elevated-camera imagery. The model predicts one broad `vehicle` class covering
cars, vans, trucks, buses/freight vehicles, and motorcycles.

## Results

Evaluated once on a held-out, sequence-aware test split containing 3,083 images
and 52,561 vehicle boxes:

| Metric | Score |
|---|---:|
| COCO mAP@50:95 | 0.6477 |
| mAP@50 | 0.9180 |
| mAP@75 | 0.7498 |
| mAR@500 | 0.7082 |
| F1 (threshold sweep) | 0.8905 |
| Precision | 0.9247 |
| Recall | 0.8587 |

## Download

```bash
hf download uwaisasghar/aerial-vehicle-detection-rfdetr \
  checkpoint_best_total.pth \
  --local-dir runs/rfdetr_small_aerial_vehicle
```

SHA-256:
`b7d80101d36bce349a09e3ee2965bc99195eafcb73054dd552e4b486d909202e`

## Inference

Clone the [source repository](https://github.com/owaisasghar/aerial-vehicle-detection-rfdetr),
install its requirements, download the checkpoint as shown above, and run:

```bash
python infer_rfdetr_video.py \
  --source input.mp4 \
  --output output_tracked.mp4 \
  --weights runs/rfdetr_small_aerial_vehicle/checkpoint_best_total.pth \
  --mode hybrid \
  --track-threshold 0.50 \
  --box-style corner
```

## Training data

The unified one-class dataset uses VisDrone2019-DET train/validation, a
YOLO-format UAVDT repack, and DroneVehicle RGB train/validation. Capture or
video sequences are kept in a single split to reduce leakage. See the source
repository README for dataset links, exact preparation steps, and directory
layout. Upstream datasets are not redistributed; users must review and comply
with their respective terms.

## Limitations

- The model does not distinguish vehicle subclasses.
- Occlusion, shadows, motion blur, and unseen viewpoints can cause errors.
- Reported results are from the same public dataset families used in training;
  validate separately on intended production cameras.
- Tracking and speed estimation are implemented by the accompanying inference
  application, not by this detector checkpoint.
- Image-plane speed is not real-world speed unless the scene is calibrated.
