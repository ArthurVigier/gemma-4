"""One-command CUDA smoke test for a cloud GPU environment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if SRC_PATH.as_posix() not in sys.path:
    sys.path.insert(0, SRC_PATH.as_posix())

from arc_drone.cloud_gpu import format_gpu_smoke_report, run_gpu_smoke


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a CUDA smoke test and optional ONNX export for ARC-drone.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--onnx-output-path", default="artifacts/onnx/trm_reasoner.onnx")
    parser.add_argument("--engine-output-path", default="artifacts/trt/trm_reasoner.plan")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_gpu_smoke(
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        export_onnx=args.export_onnx,
        onnx_output_path=args.onnx_output_path,
        engine_output_path=args.engine_output_path,
    )
    print(format_gpu_smoke_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
