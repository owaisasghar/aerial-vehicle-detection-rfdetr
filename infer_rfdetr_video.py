#!/usr/bin/env python3
"""Run RF-DETR aerial-vehicle detection and ByteTrack tracking on a video."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import supervision as sv
import torch
from rfdetr import RFDETRSmall


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = (
    ROOT / "runs" / "rfdetr_small_aerial_vehicle" / "checkpoint_best_total.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RF-DETR + sliced inference + Supervision ByteTrack"
    )
    parser.add_argument("--source", type=Path, default=ROOT / "video.MP4")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "output_rfdetr_tracked.mp4"
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.15,
        help="Detector threshold. Kept below ByteTrack's activation threshold.",
    )
    parser.add_argument("--track-threshold", type=float, default=0.50)
    parser.add_argument(
        "--mode",
        choices=("hybrid", "sliced", "full"),
        default="hybrid",
        help=(
            "Use full-frame + SAHI-style tiled inference, tiled inference only, "
            "or one full-frame prediction."
        ),
    )
    parser.add_argument("--slice-size", type=int, default=960)
    parser.add_argument("--slice-overlap", type=int, default=160)
    parser.add_argument("--slice-batch-size", type=int, default=2)
    parser.add_argument(
        "--box-style",
        choices=("corner", "full"),
        default="corner",
        help="Draw unobtrusive corner boxes or conventional full rectangles.",
    )
    parser.add_argument(
        "--meters-per-pixel",
        type=float,
        help=(
            "Ground-plane calibration for km/h. Without it, speed is displayed "
            "in pixels/second."
        ),
    )
    parser.add_argument(
        "--speed-window",
        type=float,
        default=1.0,
        help="Seconds of track history used for smoothed speed estimation.",
    )
    parser.add_argument(
        "--stationary-threshold-px-s",
        type=float,
        default=5.0,
        help="Pixel speed below this value is displayed as zero.",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Disable FP16 inference optimization on CUDA.",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--max-frames", type=int, help="Process only this many frames (for previews)."
    )
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument(
        "--preset",
        default="medium",
        choices=(
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.source.is_file():
        raise SystemExit(f"Source video not found: {args.source}")
    if not args.weights.is_file():
        raise SystemExit(f"Checkpoint not found: {args.weights}")
    if args.output.resolve() == args.source.resolve():
        raise SystemExit("Output must not overwrite the source video.")
    if not 0 < args.confidence < 1:
        raise SystemExit("--confidence must be between 0 and 1.")
    if not 0 < args.track_threshold < 1:
        raise SystemExit("--track-threshold must be between 0 and 1.")
    if args.confidence > args.track_threshold:
        raise SystemExit("--confidence must be <= --track-threshold for ByteTrack.")
    if args.slice_overlap >= args.slice_size:
        raise SystemExit("--slice-overlap must be smaller than --slice-size.")
    if args.slice_batch_size < 1:
        raise SystemExit("--slice-batch-size must be at least 1.")
    if args.meters_per_pixel is not None and args.meters_per_pixel <= 0:
        raise SystemExit("--meters-per-pixel must be greater than zero.")
    if args.speed_window <= 0:
        raise SystemExit("--speed-window must be greater than zero.")
    if args.stationary_threshold_px_s < 0:
        raise SystemExit("--stationary-threshold-px-s cannot be negative.")
    if args.start_frame < 0:
        raise SystemExit("--start-frame cannot be negative.")
    if args.max_frames is not None and args.max_frames < 1:
        raise SystemExit("--max-frames must be at least 1.")
    if not 0 <= args.crf <= 51:
        raise SystemExit("--crf must be between 0 and 51.")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required to encode the annotated video.")
    args.output.parent.mkdir(parents=True, exist_ok=True)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class VehicleDetector:
    def __init__(self, args: argparse.Namespace) -> None:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit(
                "CUDA was requested but is unavailable. Run with a working NVIDIA "
                "driver or pass --device cpu."
            )

        print(f"Loading checkpoint: {args.weights}", flush=True)
        self.model = RFDETRSmall.from_checkpoint(
            str(args.weights), device=args.device, resolution=args.resolution
        )
        self.threshold = args.confidence
        self.mode = args.mode

        if args.device == "cuda" and not args.fp32:
            print("Enabling FP16 inference optimization...", flush=True)
            self.model.inference(compile=False, dtype=torch.float16, inplace=True)

        self.slicer: sv.InferenceSlicer | None = None
        if args.mode in {"hybrid", "sliced"}:
            self.slicer = sv.InferenceSlicer(
                callback=self._predict_slice,
                slice_wh=(args.slice_size, args.slice_size),
                overlap_wh=(args.slice_overlap, args.slice_overlap),
                iou_threshold=0.50,
                thread_workers=1,
                batch_size=args.slice_batch_size,
            )

    def _predict_slice(
        self, images: np.ndarray | Sequence[np.ndarray]
    ) -> sv.Detections | list[sv.Detections]:
        if isinstance(images, (list, tuple)):
            rgb_images = [bgr_to_rgb(image) for image in images]
            predictions = self.model.predict(
                rgb_images,
                threshold=self.threshold,
                include_source_image=False,
            )
            if not isinstance(predictions, list):
                raise RuntimeError("RF-DETR returned an unexpected batched result.")
            return predictions

        return self.model.predict(
            bgr_to_rgb(images),
            threshold=self.threshold,
            include_source_image=False,
        )

    def predict(self, frame: np.ndarray) -> sv.Detections:
        full_predictions: sv.Detections | None = None
        if self.mode in {"hybrid", "full"}:
            result = self._predict_slice(frame)
            if isinstance(result, list):
                raise RuntimeError("RF-DETR returned an unexpected result.")
            full_predictions = result

        if self.slicer is None:
            if full_predictions is None:
                raise RuntimeError("No inference mode was configured.")
            return full_predictions

        sliced_predictions = self.slicer(frame)
        if self.mode == "sliced":
            return sliced_predictions
        if full_predictions is None:
            raise RuntimeError("RF-DETR returned an unexpected result.")
        return sv.Detections.merge(
            [full_predictions, sliced_predictions]
        ).with_nms(threshold=0.50, class_agnostic=True)


def open_video(path: Path, start_frame: int) -> tuple[cv2.VideoCapture, int, int, float, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open source video: {path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0 or fps <= 0:
        capture.release()
        raise SystemExit("Source video has invalid width, height, or frame rate.")
    if start_frame >= total_frames:
        capture.release()
        raise SystemExit(
            f"--start-frame {start_frame} is outside the {total_frames}-frame video."
        )
    if start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    return capture, width, height, fps, total_frames


def open_encoder(
    output: Path, width: int, height: int, fps: float, crf: int, preset: str
) -> subprocess.Popen[bytes]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        f"{fps:.8f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


class SpeedEstimator:
    """Estimate image-plane speed from smoothed ByteTrack center trajectories."""

    def __init__(
        self,
        fps: float,
        window_seconds: float,
        stationary_threshold_px_s: float,
        meters_per_pixel: float | None,
    ) -> None:
        self.fps = fps
        self.window_frames = max(3, round(window_seconds * fps))
        self.stationary_threshold_px_s = stationary_threshold_px_s
        self.meters_per_pixel = meters_per_pixel
        self.histories: dict[int, deque[tuple[int, float, float]]] = defaultdict(
            deque
        )
        self.smoothed_speeds: dict[int, float] = {}
        self.last_seen: dict[int, int] = {}

    def update(
        self, detections: sv.Detections, frame_number: int
    ) -> dict[int, float | None]:
        speeds: dict[int, float | None] = {}
        if detections.tracker_id is None:
            return speeds

        centers = np.column_stack(
            (
                (detections.xyxy[:, 0] + detections.xyxy[:, 2]) / 2.0,
                (detections.xyxy[:, 1] + detections.xyxy[:, 3]) / 2.0,
            )
        )
        for tracker_id_value, center in zip(detections.tracker_id, centers):
            tracker_id = int(tracker_id_value)
            if tracker_id < 0:
                continue
            history = self.histories[tracker_id]
            history.append((frame_number, float(center[0]), float(center[1])))
            self.last_seen[tracker_id] = frame_number

            while history and frame_number - history[0][0] > self.window_frames:
                history.popleft()

            if len(history) < 4:
                speeds[tracker_id] = None
                continue

            samples = np.asarray(history, dtype=np.float64)
            times = (samples[:, 0] - samples[0, 0]) / self.fps
            if times[-1] <= 0:
                speeds[tracker_id] = None
                continue

            # Least-squares velocity is less sensitive to detector box jitter than
            # frame-to-frame displacement.
            design = np.column_stack((times, np.ones_like(times)))
            velocity_x = np.linalg.lstsq(design, samples[:, 1], rcond=None)[0][0]
            velocity_y = np.linalg.lstsq(design, samples[:, 2], rcond=None)[0][0]
            raw_speed = float(np.hypot(velocity_x, velocity_y))
            if raw_speed < self.stationary_threshold_px_s:
                raw_speed = 0.0

            previous = self.smoothed_speeds.get(tracker_id)
            smoothed = raw_speed if previous is None else 0.25 * raw_speed + 0.75 * previous
            self.smoothed_speeds[tracker_id] = smoothed
            speeds[tracker_id] = smoothed

        stale_before = frame_number - 5 * round(self.fps)
        stale_ids = [
            tracker_id
            for tracker_id, last_frame in self.last_seen.items()
            if last_frame < stale_before
        ]
        for tracker_id in stale_ids:
            self.histories.pop(tracker_id, None)
            self.smoothed_speeds.pop(tracker_id, None)
            self.last_seen.pop(tracker_id, None)
        return speeds

    def format_speed(self, speed_px_s: float | None) -> str:
        if self.meters_per_pixel is None:
            return "-- px/s" if speed_px_s is None else f"{speed_px_s:.0f} px/s"
        if speed_px_s is None:
            return "-- km/h"
        speed_kmh = speed_px_s * self.meters_per_pixel * 3.6
        return f"{speed_kmh:.1f} km/h"

    @property
    def unit_name(self) -> str:
        if self.meters_per_pixel is None:
            return "PX/S"
        return f"KM/H @ {self.meters_per_pixel:.4f} M/PX"


def annotate(
    frame: np.ndarray,
    detections: sv.Detections,
    box_annotator: sv.BoxCornerAnnotator | sv.BoxAnnotator,
    label_annotator: sv.LabelAnnotator,
    trace_annotator: sv.TraceAnnotator,
    speed_estimator: SpeedEstimator,
    unique_ids: set[int],
    frame_number: int,
    mode: str,
) -> np.ndarray:
    labels = []
    tracker_ids = detections.tracker_id
    speeds = speed_estimator.update(detections, frame_number)
    for index in range(len(detections)):
        tracker_id = int(tracker_ids[index]) if tracker_ids is not None else -1
        labels.append(
            f"#{tracker_id} | {speed_estimator.format_speed(speeds.get(tracker_id))}"
        )
        if tracker_id >= 0:
            unique_ids.add(tracker_id)

    scene = trace_annotator.annotate(scene=frame.copy(), detections=detections)
    scene = box_annotator.annotate(scene=scene, detections=detections)
    scene = label_annotator.annotate(
        scene=scene, detections=detections, labels=labels
    )

    active = len(detections)
    status = (
        f"ACTIVE: {active}   UNIQUE: {len(unique_ids)}   "
        f"FRAME: {frame_number}   MODE: {mode.upper()}   "
        f"SPEED: {speed_estimator.unit_name}"
    )
    (text_width, text_height), _ = cv2.getTextSize(
        status, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2
    )
    cv2.rectangle(scene, (16, 14), (36 + text_width, 38 + text_height), (0, 0, 0), -1)
    cv2.putText(
        scene,
        status,
        (26, 30 + text_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return scene


def main() -> None:
    args = parse_args()
    validate_args(args)
    capture, width, height, fps, total_frames = open_video(
        args.source, args.start_frame
    )
    available_frames = total_frames - args.start_frame
    target_frames = min(available_frames, args.max_frames or available_frames)

    print(
        f"Source: {args.source} | {width}x{height} | {fps:.3f} FPS | "
        f"processing {target_frames}/{total_frames} frames",
        flush=True,
    )
    detector = VehicleDetector(args)

    # Keep using Supervision's bundled ByteTrack for compatibility with this
    # environment. Supervision 0.30 marks it deprecated in favor of a separate
    # package, but its tracking behavior remains stable for this pipeline.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The `ByteTrack` was deprecated", category=FutureWarning
        )
        tracker = sv.ByteTrack(
            track_activation_threshold=args.track_threshold,
            lost_track_buffer=30,
            minimum_matching_threshold=0.80,
            frame_rate=fps,
            minimum_consecutive_frames=2,
        )

    if args.box_style == "corner":
        box_annotator: sv.BoxCornerAnnotator | sv.BoxAnnotator = (
            sv.BoxCornerAnnotator(
                thickness=4,
                corner_length=18,
                color_lookup=sv.ColorLookup.TRACK,
            )
        )
    else:
        box_annotator = sv.BoxAnnotator(
            thickness=3, color_lookup=sv.ColorLookup.TRACK
        )
    label_annotator = sv.LabelAnnotator(
        text_scale=0.58,
        text_thickness=2,
        text_padding=5,
        smart_position=True,
        color_lookup=sv.ColorLookup.TRACK,
    )
    trace_annotator = sv.TraceAnnotator(
        trace_length=max(30, round(fps)),
        thickness=3,
        smooth=True,
        color_lookup=sv.ColorLookup.TRACK,
    )
    speed_estimator = SpeedEstimator(
        fps=fps,
        window_seconds=args.speed_window,
        stationary_threshold_px_s=args.stationary_threshold_px_s,
        meters_per_pixel=args.meters_per_pixel,
    )

    encoder = open_encoder(
        args.output, width, height, fps, args.crf, args.preset
    )
    if encoder.stdin is None:
        raise RuntimeError("Could not open ffmpeg input pipe.")

    unique_ids: set[int] = set()
    processed = 0
    started = time.perf_counter()
    try:
        while processed < target_frames:
            ok, frame = capture.read()
            if not ok:
                break

            detections = detector.predict(frame)
            tracked = tracker.update_with_detections(detections)
            absolute_frame = args.start_frame + processed
            annotated = annotate(
                frame,
                tracked,
                box_annotator,
                label_annotator,
                trace_annotator,
                speed_estimator,
                unique_ids,
                absolute_frame,
                args.mode,
            )
            encoder.stdin.write(annotated.tobytes())
            processed += 1

            if processed == 1 or processed % 25 == 0 or processed == target_frames:
                elapsed = time.perf_counter() - started
                rate = processed / elapsed
                eta = (target_frames - processed) / rate if rate else 0.0
                print(
                    f"[{processed:4d}/{target_frames}] {rate:.2f} frames/s | "
                    f"ETA {eta:.1f}s | active {len(tracked)} | "
                    f"unique tracks {len(unique_ids)}",
                    flush=True,
                )
    except (BrokenPipeError, KeyboardInterrupt):
        print("Inference interrupted; finalizing partial output...", file=sys.stderr)
    finally:
        capture.release()
        encoder.stdin.close()
        return_code = encoder.wait()

    if return_code != 0:
        raise SystemExit(f"ffmpeg failed with exit code {return_code}.")
    if processed == 0:
        raise SystemExit("No frames were processed.")

    elapsed = time.perf_counter() - started
    print(
        f"Done: {args.output} | {processed} frames | "
        f"{len(unique_ids)} unique tracks | {elapsed:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
