"""Command-line entrypoint for training the TRM-like student."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if SRC_PATH.as_posix() not in sys.path:
    sys.path.insert(0, SRC_PATH.as_posix())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

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
    parser.add_argument("--action-regression-weight", type=float, default=0.5)
    parser.add_argument("--halt-loss-weight", type=float, default=0.25)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--temporal-window", type=int, default=4,
                        help="Number of consecutive frames per input sequence (T).")
    parser.add_argument("--action-chunk-size", type=int, default=4,
                        help="Number of future actions predicted per forward pass (C).")
    parser.add_argument("--auair-path", type=str, default=None,
                        help="JSONL from parse_auair.py. When set, trains on real AU-AIR "
                             "sequences instead of synthetic ARC tasks.")
    parser.add_argument("--auair-images-path", type=str, default=None,
                        help="Root directory for AU-AIR images. Filenames from JSONL will be resolved relative to this.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger.info(
        "Starting student training | model=%s  device=%s  epochs=%d  lr=%.2e  batch=%d  auair=%s",
        args.foundation_model_id, args.device, args.epochs, args.learning_rate, args.batch_size,
        args.auair_path or "synthetic",
    )
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
            action_regression_weight=args.action_regression_weight,
            halt_loss_weight=args.halt_loss_weight,
            num_workers=args.num_workers,
            temporal_window=args.temporal_window,
            action_chunk_size=args.action_chunk_size,
            auair_path=args.auair_path,
            auair_images_path=args.auair_images_path,
        )
    )
    logger.info(
        "Training complete | best_eval_action_accuracy=%.4f  best_eval_halt_mae=%.4f  output=%s",
        summary.best_eval_action_accuracy, summary.best_eval_halt_step_mae, summary.output_dir,
    )
    print(format_training_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
