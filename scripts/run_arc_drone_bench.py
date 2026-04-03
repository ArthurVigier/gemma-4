"""Small CLI entrypoint for dry-running ARC-Drone-Bench generation."""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.as_posix() not in sys.path:
    sys.path.insert(0, SRC.as_posix())

from arc_drone.arc_drone_bench import ARCDroneBench


def main() -> None:
    bench = ARCDroneBench()
    tasks = bench.generate_tasks()
    avg_opening = mean(
        float(task.metadata["opening_row"])
        for task in tasks
        if task.family == "path_planning"
    )
    print(f"Generated {len(tasks)} tasks across {len(bench.config.task_families)} families.")
    print(f"Average path-planning opening row: {avg_opening:.2f}")


if __name__ == "__main__":
    main()
