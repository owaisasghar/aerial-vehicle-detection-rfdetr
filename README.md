# Aerial Vehicle Detection, Tracking, and Speed Estimation

Production-oriented aerial vehicle analytics built with a fine-tuned
[RF-DETR Small](https://github.com/roboflow/rf-detr), Supervision-style sliced
inference, ByteTrack, and OpenCV/FFmpeg video processing.

The detector is trained as a single `vehicle` class covering cars, vans, trucks,
buses/freight vehicles, and motorcycles in drone and elevated-camera imagery.

![RF-DETR vehicle tracking with corner boxes and estimated speed](assets/demo_corner_speed.jpg)

## Highlights

- Fine-tuned RF-DETR Small detector for small aerial vehicles.
- Hybrid inference: full-frame context plus overlapping 960 px image slices.
- NMS merges full-frame and sliced predictions.
- ByteTrack assigns persistent vehicle IDs and motion traces.
- Corner-style or conventional full bounding boxes.
- Smoothed image-plane speed in px/s.
- Optional km/h conversion using a measured meters-per-pixel calibration.
- FP16 CUDA inference and high-quality H.264 output.
- Deterministic, leakage-aware dataset splitting by capture/video sequence.

## Model accuracy

The selected artifact is the best EMA checkpoint:
`checkpoint_best_total.pth`. Early stopping ended training after epoch 33 when
the monitored metric had not improved for 10 validation records. The best EMA
validation mAP@50:95 was **0.5843** at epoch 32.

### Held-out test results

The best checkpoint was loaded once and evaluated on the held-out test split of
3,083 images and 52,561 vehicle boxes.

| Metric | Score | Percentage |
|---|---:|---:|
| mAP@50:95 | 0.6477 | 64.77% |
| mAP@50 | 0.9180 | 91.80% |
| mAP@75 | 0.7498 | 74.98% |
| mAR@500 | 0.7082 | 70.82% |
| F1 (threshold sweep) | 0.8905 | 89.05% |
| Precision | 0.9247 | 92.47% |
| Recall | 0.8587 | 85.87% |

Object detectors do not have one universal classification-style “accuracy.”
The primary score here is COCO **mAP@50:95**, which averages performance across
IoU thresholds from 0.50 through 0.95. mAP@50 is more forgiving and describes
whether the detector generally found the vehicle; mAP@75 and mAP@50:95 place
more weight on tight box localization.

These test images come from the same three public dataset families as training,
although capture sequences were kept in only one split. A separate test set from
the intended production cameras is still required before deployment.

## Approach

```mermaid
flowchart LR
    A[Video frame] --> B[RGB conversion]
    B --> C[Full-frame RF-DETR]
    B --> D[Overlapping sliced RF-DETR]
    C --> E[NMS merge]
    D --> E
    E --> F[ByteTrack IDs]
    F --> G[Smoothed trajectories]
    G --> H[Corner boxes, speed labels, H.264 video]
```

1. **Detection** — RF-DETR Small predicts a single vehicle class at 768 px
   inference resolution.
2. **Small-object recovery** — `supervision.InferenceSlicer` processes
   overlapping 960×960 crops. This is a SAHI-style slicing approach without a
   separate SAHI dependency.
3. **Context recovery** — a full-frame prediction runs alongside the slices so
   medium/large vehicles are not lost when a crop removes useful context.
4. **Merge** — class-agnostic NMS at 0.50 IoU removes duplicate full/slice boxes.
5. **Tracking** — ByteTrack uses confidence and box motion to assign IDs.
6. **Speed** — least-squares velocity is calculated from one second of tracked
   box-center history, followed by exponential smoothing and stationary-jitter
   suppression.

## Dataset

The unified COCO dataset combines:

- VisDrone2019-DET train and validation
- UAVDT community YOLO repack
- DroneVehicle RGB train and validation

| Split | Images | Boxes | Negative images |
|---|---:|---:|---:|
| Train | 21,858 | 497,226 | 476 |
| Validation | 3,326 | 49,246 | 159 |
| Test | 3,083 | 52,561 | 93 |
| **Total** | **28,267** | **599,033** | **728** |

Source image totals are 19,445 DroneVehicle, 7,016 VisDrone, and 1,806 UAVDT
images. Valid negative images are deliberately retained to reduce false
positives.

### Dataset preparation and QA

- Maps supported vehicle categories into one `vehicle` class.
- Converts VisDrone text, UAVDT YOLO, and DroneVehicle XML/polygon annotations
  into COCO bounding boxes.
- Removes the documented DroneVehicle border and translates/clips annotations.
- Rejects corrupt images, invalid boxes, and zero-area boxes.
- Removes exact duplicate files.
- Groups likely frames from the same capture/video before splitting.
- Forces high-confidence perceptual matches into the same split group.
- Uses deterministic seed `42` and audits every final image and annotation.

Raw and prepared datasets are not included in this repository. Review and
follow each upstream dataset's terms before training or commercial use.

## Training configuration

| Setting | Value |
|---|---|
| Model | RF-DETR Small |
| Task | One-class aerial vehicle detection |
| Resolution | 768 |
| Maximum epochs | 50 |
| Completed epoch | 33 (early stopped) |
| Batch size | 1 |
| Gradient accumulation | 16 |
| Effective batch size | 16 |
| Learning rate | 1e-4 |
| Encoder learning rate | 1.5e-4 |
| Augmentation | `AUG_AERIAL` |
| EMA | Enabled |
| Early-stopping patience | 10 |
| Seed | 42 |

Training was performed locally on an NVIDIA GeForce RTX 4060 Laptop GPU with
8 GB VRAM.

## Project structure

```text
.
├── infer_rfdetr_video.py       # Detection, slicing, tracking, speed, encoding
├── train_rfdetr.py             # RF-DETR training and resume entry point
├── prepare_dataset_v2.py       # Leakage-aware unified COCO preparation
├── audit_dataset.py            # Full dataset QA/audit
├── make_rfdetr_layout.py       # RF-DETR/Roboflow folder layout conversion
├── remove_split_duplicates.py  # Exact cross-split duplicate cleanup
├── validate_rfdetr_loader.py   # Data-loader validation
├── results/test_metrics.json   # Final measured test metrics
└── assets/demo_corner_speed.jpg
```

## Installation

Python 3.11 or 3.12 is recommended. FFmpeg is required for H.264 output.

```bash
sudo apt-get install ffmpeg

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA, install a PyTorch build compatible with the local NVIDIA driver. If a
Conda or robotics environment injects incompatible CUDA libraries, launch with
`env -u LD_LIBRARY_PATH` as shown below.

## Model checkpoint

The best checkpoint is approximately 124 MB and exceeds GitHub's normal 100 MB
file limit, so model weights are not committed. Download the private release
asset when available, or copy the trained checkpoint to:

```text
runs/rfdetr_small_aerial_vehicle/checkpoint_best_total.pth
```

You may also pass any checkpoint path explicitly with `--weights`.

## Video inference

```bash
env -u LD_LIBRARY_PATH .venv/bin/python infer_rfdetr_video.py \
  --source input.mp4 \
  --output output_tracked.mp4 \
  --weights runs/rfdetr_small_aerial_vehicle/checkpoint_best_total.pth \
  --mode hybrid \
  --track-threshold 0.50 \
  --box-style corner
```

Useful modes:

- `--mode hybrid`: full frame plus slices; best quality and default.
- `--mode sliced`: slices only; strong small-object detail but less context.
- `--mode full`: fastest; may miss very small vehicles.
- `--box-style corner`: unobtrusive corner boxes.
- `--box-style full`: conventional rectangles.
- `--max-frames N`: short preview or tuning run.
- `--fp32`: disable CUDA FP16 optimization.

Run `python infer_rfdetr_video.py --help` for all options.

## Speed estimation

Without scene calibration, the script reports px/s. For km/h, measure a known
ground-plane distance visible in the video:

```text
meters_per_pixel = known_distance_in_meters / measured_distance_in_pixels
```

Then run:

```bash
env -u LD_LIBRARY_PATH .venv/bin/python infer_rfdetr_video.py \
  --source input.mp4 \
  --output output_speed.mp4 \
  --meters-per-pixel 0.038
```

The `0.038` value used in the included demonstration was estimated from a
typical 4.5 m car spanning about 118 pixels. It is appropriate for a visual
demo, not enforcement or safety decisions. Accurate deployment needs surveyed
ground control points and, for non-top-down cameras, a road-plane homography.

## Rebuilding the dataset and training

After placing the upstream datasets beneath `datasets/`:

```bash
source .venv/bin/activate
python prepare_dataset_v2.py
python audit_dataset.py
python validate_rfdetr_loader.py

env -u LD_LIBRARY_PATH python train_rfdetr.py
```

Resume from a Lightning checkpoint:

```bash
env -u LD_LIBRARY_PATH python train_rfdetr.py \
  --resume runs/rfdetr_small_aerial_vehicle/last.ckpt
```

## Limitations

- The model predicts one broad vehicle category rather than car/truck/bus
  subclasses.
- Severe tree cover, shadows, motion blur, truncation, or viewpoints unlike the
  training data can cause misses and false positives.
- ByteTrack IDs can change after long occlusion or leaving/re-entering the frame;
  cumulative IDs are not guaranteed to equal the physical vehicle count.
- Speed is image-plane motion until the scene is calibrated.
- The held-out public-dataset score does not replace production-camera testing.

## Repository data policy

The Git repository contains source code, documentation, a compressed demo image,
and small metric files only. The following remain local or are distributed as
separate authorized artifacts:

- upstream datasets and prepared images/annotations;
- training checkpoints and exported engines;
- source, preview, and rendered videos;
- virtual environments, caches, and logs.
