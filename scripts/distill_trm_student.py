"""Command-line entrypoint for Gemma-guided student distillation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if SRC_PATH.as_posix() not in sys.path:
    sys.path.insert(0, SRC_PATH.as_posix())

from arc_drone.student_distillation import DistillationConfig, distill_student
from arc_drone.student_training import format_training_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill the TRM-like student from a Gemma hidden layer.")
    parser.add_argument("--foundation-model-id", default="google/gemma-4-e2b")
    parser.add_argument("--teacher-layer-index", type=int, default=17)
    parser.add_argument("--teacher-layer-indices", type=int, nargs="*", default=None)
    parser.add_argument("--teacher-feature-pooling", choices=("mean", "concat"), default="mean")
    parser.add_argument("--cache-dir")
    parser.add_argument("--task-count", type=int, default=4096)
    parser.add_argument("--eval-task-count", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="artifacts/checkpoints/trm_student_distilled")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--onnx-output-path")
    parser.add_argument("--trt-engine-output-path")
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--refinement-steps", type=int, default=6)
    parser.add_argument("--halting-threshold", type=float, default=0.82)
    parser.add_argument("--action-loss-weight", type=float, default=2.0)
    parser.add_argument("--action-regression-weight", type=float, default=0.5)
    parser.add_argument("--halt-loss-weight", type=float, default=0.5)
    parser.add_argument("--teacher-representation-weight", type=float, default=0.0)
    parser.add_argument("--teacher-kl-weight", type=float, default=0.25)
    parser.add_argument("--teacher-probe-epochs", type=int, default=5)
    parser.add_argument("--teacher-probe-learning-rate", type=float, default=1e-3)
    parser.add_argument("--teacher-probe-batch-size", type=int, default=64)
    parser.add_argument("--teacher-temperature", type=float, default=2.0)
    parser.add_argument("--teacher-max-length", type=int, default=768)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    teacher_layer_indices = tuple(args.teacher_layer_indices) if args.teacher_layer_indices else (args.teacher_layer_index,)
    summary = distill_student(
        DistillationConfig(
            foundation_model_id=args.foundation_model_id,
            teacher_layer_indices=teacher_layer_indices,
            teacher_feature_pooling=args.teacher_feature_pooling,
            cache_dir=args.cache_dir,
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
            teacher_representation_weight=args.teacher_representation_weight,
            teacher_kl_weight=args.teacher_kl_weight,
            teacher_probe_epochs=args.teacher_probe_epochs,
            teacher_probe_learning_rate=args.teacher_probe_learning_rate,
            teacher_probe_batch_size=args.teacher_probe_batch_size,
            teacher_temperature=args.teacher_temperature,
            teacher_max_length=args.teacher_max_length,
        )
    )
    print(format_training_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
