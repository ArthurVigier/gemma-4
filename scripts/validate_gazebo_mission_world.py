"""Validates the local Gazebo mission world against ARC-Drone-Bench expectations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if SRC_PATH.as_posix() not in sys.path:
    sys.path.insert(0, SRC_PATH.as_posix())

from arc_drone.live_validation import validate_mission_world, validation_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Gazebo mission marker assets for ARC-Drone-Bench.")
    parser.add_argument(
        "--world-path",
        default=Path("assets/gazebo/worlds/arc_drone_bench_mission.world").as_posix(),
        help="Path to the Gazebo world file to validate.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_mission_world(args.world_path)
    print(validation_summary(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
