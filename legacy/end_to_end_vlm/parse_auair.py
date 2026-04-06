#!/usr/bin/env python3
"""
AU-AIR dataset parser → JSONL for temporal student training.

Downloads annotations + images, groups them into temporal sequences of T
consecutive frames, derives ground-truth drone actions from the recorded
velocity telemetry, and writes a JSONL file where each record contains:

  sample_id       : str
  clip_id         : str   (datetime prefix from image_name)
  frame_index     : int   (0-based position within clip)
  image_paths     : list[str]   (T absolute paths, oldest→newest)
  action_index    : int   (0-7 in ACTION_VOCABULARY order)
  halt_step       : int   (1-6)
  telemetry       : dict  (linear_xyz, angles, altitude_m, lat, lon at t)
  n_objects       : int
  categories      : list[str]

Usage:
    python scripts/parse_auair.py \
        --images-dir /data/auair/images \
        --annotations /data/auair/annotations.json \
        --output data/auair_sequences.jsonl \
        --temporal-window 4 \
        --action-chunk-size 4
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Action vocabulary (mirrors student_training.ACTION_VOCABULARY) ──────────
# 0: north (+x fwd), 1: south (-x bwd), 2: east (+y right), 3: west (-y left)
# 4: up (+z), 5: down (-z), 6: yaw_right (+psi), 7: yaw_left (-psi)

AU_AIR_CATEGORIES = ["Human", "Car", "Truck", "Van", "Motorbike", "Bicycle", "Bus", "Trailer"]


def _velocity_to_action(linear_x: float, linear_y: float, linear_z: float, dpsi: float) -> int:
    """Maps recorded drone velocity to the nearest ACTION_VOCABULARY index.

    AU-AIR telemetry uses NED-like body frame:
      linear_x  = forward velocity (positive = forward = north)
      linear_y  = lateral velocity (positive = right = east)
      linear_z  = vertical velocity (positive = up for Bebop 2 convention)
      dpsi      = yaw rate (rad/s, positive = clockwise = yaw_right)

    We pick the axis with the largest absolute magnitude.
    """
    candidates = [
        (abs(linear_x), 0 if linear_x >= 0 else 1),   # north / south
        (abs(linear_y), 2 if linear_y >= 0 else 3),   # east / west
        (abs(linear_z), 4 if linear_z >= 0 else 5),   # up / down
        (abs(dpsi),     6 if dpsi >= 0 else 7),        # yaw_right / yaw_left
    ]
    return max(candidates, key=lambda t: t[0])[1]


def _velocity_to_halt(linear_x: float, linear_y: float, linear_z: float, dpsi: float) -> int:
    """Maps velocity magnitude to halt confidence step (1=confident, 6=uncertain).

    High speed → lower step (less confident about stopping).
    Near-stationary → high step (almost ready to halt).
    """
    speed = math.sqrt(linear_x**2 + linear_y**2 + linear_z**2)
    # Empirical range from dataset: ~0 to ~0.5 m/s typical
    if speed < 0.02:
        return 6   # hovering, very confident
    elif speed < 0.05:
        return 5
    elif speed < 0.10:
        return 4
    elif speed < 0.20:
        return 3
    elif speed < 0.35:
        return 2
    else:
        return 1   # fast movement, least confident


def _yaw_delta(rec_a: dict, rec_b: dict) -> float:
    """Raw heading change between two consecutive records (radians, wrapped to [-pi, pi]).

    AU-AIR timestamps have integer-second resolution so dt is often 0 — using a
    rate (dpsi/dt) would blow up.  Instead we return the raw angular delta and
    compare it directly against linear velocities; the magnitudes are empirically
    on the same order (~0.05-0.15 for both).
    """
    dpsi = rec_b["angle_psi"] - rec_a["angle_psi"]
    return (dpsi + math.pi) % (2 * math.pi) - math.pi


def _record_time_s(rec: dict) -> float:
    """Returns absolute time in seconds from a record's time dict."""
    t = rec["time"]
    return (
        t["hour"] * 3600.0
        + t["min"] * 60.0
        + t["sec"]
        + t["ms"] / 1_000_000.0  # ms field is in microseconds in practice
    )


def _clip_id(image_name: str) -> str:
    """Extracts clip identifier from image filename."""
    # pattern: frame_YYYYMMDDHHMMSS_x_NNNNNNN.jpg
    parts = image_name.split("_")
    return parts[1] if len(parts) >= 2 else "unknown"


def load_annotations(annotations_path: Path) -> list[dict]:
    t0 = time.perf_counter()
    data = json.loads(annotations_path.read_bytes().decode("utf-8", errors="replace"))
    records = data["annotations"]
    logger.info("Loaded annotations from %s — %d records in %.2fs", annotations_path, len(records), time.perf_counter() - t0)
    return records


def group_by_clip(annotations: list[dict]) -> dict[str, list[dict]]:
    clips: dict[str, list[dict]] = defaultdict(list)
    for rec in annotations:
        cid = _clip_id(rec["image_name"])
        clips[cid].append(rec)
    # Sort each clip by time
    for cid in clips:
        clips[cid].sort(key=_record_time_s)
    result = dict(clips)
    logger.info("Grouped into %d clips", len(result))
    return result


def build_sequences(
    clips: dict[str, list[dict]],
    images_dir: Path,
    temporal_window: int,
    action_chunk_size: int,
    stride: int,
) -> list[dict]:
    """Slides a window of T frames over each clip to produce training sequences."""
    sequences: list[dict] = []
    skipped_clips = 0

    for clip_id, frames in clips.items():
        n = len(frames)
        # Need T frames as input + C frames ahead for action chunk targets
        min_len = temporal_window + action_chunk_size
        if n < min_len:
            logger.warning("Skipping clip %s: only %d frames, need %d (T=%d + C=%d)", clip_id, n, min_len, temporal_window, action_chunk_size)
            skipped_clips += 1
            continue

        for start in range(0, n - min_len + 1, stride):
            window = frames[start: start + temporal_window]
            chunk = frames[start + temporal_window - 1: start + temporal_window - 1 + action_chunk_size]

            # Verify all images exist
            image_paths = []
            all_exist = True
            for rec in window:
                p = images_dir / rec["image_name"]
                if not p.exists():
                    logger.debug("Missing image, skipping sequence %s-%06d: %s", clip_id, start, p)
                    all_exist = False
                    break
                image_paths.append(str(p))
            if not all_exist:
                continue

            # Action chunk: derive from each frame in chunk using velocity telemetry
            action_indices: list[int] = []
            halt_steps: list[int] = []
            for i, rec in enumerate(chunk):
                # Heading delta vs next frame (raw radians, no rate division)
                if i + 1 < len(chunk):
                    dpsi = _yaw_delta(rec, chunk[i + 1])
                else:
                    dpsi = 0.0
                a = _velocity_to_action(
                    rec["linear_x"], rec["linear_y"], rec["linear_z"], dpsi
                )
                h = _velocity_to_halt(
                    rec["linear_x"], rec["linear_y"], rec["linear_z"], dpsi
                )
                action_indices.append(a)
                halt_steps.append(h)

            # Telemetry snapshot at the last frame of the input window
            anchor = window[-1]
            telemetry = {
                "linear_x": anchor["linear_x"],
                "linear_y": anchor["linear_y"],
                "linear_z": anchor["linear_z"],
                "angle_phi": anchor["angle_phi"],
                "angle_theta": anchor["angle_theta"],
                "angle_psi": anchor["angle_psi"],
                "altitude_m": anchor["altitude"] / 1000.0,
                "latitude": anchor["latitude"],
                "longitude": anchor["longtitude"],  # note: typo in source data
            }

            bboxes = anchor.get("bbox", [])
            cat_names = [AU_AIR_CATEGORIES[b["class"]] for b in bboxes if 0 <= b["class"] < 8]

            seq_id = f"{clip_id}-{start:06d}"
            sequences.append({
                "sample_id": seq_id,
                "clip_id": clip_id,
                "frame_index": start,
                "image_paths": image_paths,
                "action_indices": action_indices,    # (chunk_size,) ground truth
                "halt_steps": halt_steps,            # (chunk_size,)
                # Convenience: current action = action_indices[0]
                "action_index": action_indices[0],
                "halt_step": halt_steps[0],
                "telemetry": telemetry,
                "n_objects": len(bboxes),
                "categories": cat_names,
            })

    logger.info(
        "Built %d sequences from %d clips (skipped %d short clips)",
        len(sequences), len(clips), skipped_clips,
    )
    return sequences


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse AU-AIR into temporal sequences JSONL.")
    parser.add_argument("--images-dir", required=True, type=str, help="Path to extracted AU-AIR images/ folder.")
    parser.add_argument("--annotations", required=True, type=str, help="Path to annotations.json.")
    parser.add_argument("--output", type=str, default="data/auair_sequences.jsonl")
    parser.add_argument("--temporal-window", type=int, default=4, help="Number of input frames per sequence.")
    parser.add_argument("--action-chunk-size", type=int, default=4, help="Number of future actions to predict.")
    parser.add_argument("--stride", type=int, default=2, help="Frame stride when sliding the window (1=max overlap).")
    parser.add_argument("--stats", action="store_true", help="Print dataset statistics and exit.")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    annotations_path = Path(args.annotations)

    logger.info("images_dir=%s  annotations=%s", images_dir, annotations_path)
    annotations = load_annotations(annotations_path)

    clips = group_by_clip(annotations)
    for cid, frames in clips.items():
        logger.debug("Clip %s: %d frames", cid, len(frames))

    if args.stats:
        action_hist = defaultdict(int)
        for rec in annotations:
            a = _velocity_to_action(rec["linear_x"], rec["linear_y"], rec["linear_z"], 0.0)
            action_hist[a] += 1
        print("\nAction distribution (from velocity):")
        labels = ["north", "south", "east", "west", "up", "down", "yaw+", "yaw-"]
        for idx, label in enumerate(labels):
            print(f"  {idx} {label}: {action_hist[idx]}")
        return

    logger.info(
        "Building sequences (T=%d, C=%d, stride=%d)...",
        args.temporal_window, args.action_chunk_size, args.stride,
    )
    t0 = time.perf_counter()
    sequences = build_sequences(
        clips=clips,
        images_dir=images_dir,
        temporal_window=args.temporal_window,
        action_chunk_size=args.action_chunk_size,
        stride=args.stride,
    )
    logger.info("Sequence building done in %.2fs — %d sequences", time.perf_counter() - t0, len(sequences))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for seq in sequences:
            f.write(json.dumps(seq) + "\n")

    logger.info("Written to: %s", output_path)

    # Action distribution — keep as print for user-facing summary
    from collections import Counter
    action_counts = Counter(s["action_index"] for s in sequences)
    labels = ["north", "south", "east", "west", "up", "down", "yaw+", "yaw-"]
    print("\nAction distribution in sequences:")
    total = sum(action_counts.values())
    for idx, label in enumerate(labels):
        n = action_counts.get(idx, 0)
        print(f"  {idx} {label}: {n} ({100*n/max(total,1):.1f}%)")


if __name__ == "__main__":
    main()
