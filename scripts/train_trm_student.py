"""Command-line entrypoint for training the TRM-like student."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if SRC_PATH.as_posix() not in sys.path:
    sys.path.insert(0, SRC_PATH.as_posix())

from arc_drone.student_training import StudentTrainingConfig, format_training_summary, train_student


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TRM-like student on ARC-Drone-Bench.")
    parser.add_argument("--foundation-model-id", default="google/gemma-4-e2b")
    parser.add_argument("--task-count", type=int, default=4096)
    parser.add_argument("--eval-task-count", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="artifacts/checkpoints/trm_student")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--onnx-output-path")
    parser.add_argument("--trt-engine-output-path")
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--refinement-steps", type=int, default=6)
    parser.add_argument("--halting-threshold", type=float, default=0.82)
    parser.add_argument("--action-loss-weight", type=float, default=1.0)
    parser.add_argument("--halt-loss-weight", type=float, default=0.25)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = train_student(
        StudentTrainingConfig(
            foundation_model_id=args.foundation_model_id,
            task_count=args.task_count,
            eval_task_count=args.eval_task_count,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=args.device,
            output_dir=args.output_dir,
            export_onnx=args.export_onnx,
            onnx_output_path=args.onnx_output_path,
            trt_engine_output_path=args.trt_engine_output_path,
            hidden_size=args.hidden_size,
            refinement_steps=args.refinement_steps,
            halting_threshold=args.halting_threshold,
            action_loss_weight=args.action_loss_weight,
            halt_loss_weight=args.halt_loss_weight,
            num_workers=args.num_workers,
        )
    )
    print(format_training_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
