"""Command-line entrypoint for lightweight Gemma hidden-layer sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if SRC_PATH.as_posix() not in sys.path:
    sys.path.insert(0, SRC_PATH.as_posix())

from arc_drone.gemma_layer_sweep import LayerSweepConfig, format_layer_sweep_summary, run_layer_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight Gemma hidden-layer sweep.")
    parser.add_argument("--foundation-model-id", default="google/gemma-4-e2b")
    parser.add_argument("--task-count", type=int, default=1024)
    parser.add_argument("--eval-task-count", type=int, default=256)
    parser.add_argument("--probe-epochs", type=int, default=5)
    parser.add_argument("--probe-batch-size", type=int, default=32)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="artifacts/layer_sweeps/gemma_layers")
    parser.add_argument("--layers", type=int, nargs="*", default=[])
    parser.add_argument("--layer-fractions", type=float, nargs="*", default=[0.25, 0.5, 0.75, 0.9])
    parser.add_argument("--max-length", type=int, default=768)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_layer_sweep(
        LayerSweepConfig(
            foundation_model_id=args.foundation_model_id,
            task_count=args.task_count,
            eval_task_count=args.eval_task_count,
            probe_epochs=args.probe_epochs,
            probe_batch_size=args.probe_batch_size,
            probe_learning_rate=args.probe_learning_rate,
            seed=args.seed,
            device=args.device,
            output_dir=args.output_dir,
            layers=tuple(args.layers),
            layer_fractions=tuple(args.layer_fractions),
            max_length=args.max_length,
        )
    )
    print(format_layer_sweep_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
