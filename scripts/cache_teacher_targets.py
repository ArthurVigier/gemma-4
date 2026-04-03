"""Command-line entrypoint for building reusable Gemma teacher caches."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if SRC_PATH.as_posix() not in sys.path:
    sys.path.insert(0, SRC_PATH.as_posix())

from arc_drone.student_distillation import (
    DistillationCacheConfig,
    build_teacher_target_cache,
    format_teacher_cache_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reusable Gemma teacher-target cache.")
    parser.add_argument("--foundation-model-id", default="google/gemma-4-e2b")
    parser.add_argument("--teacher-layer-indices", type=int, nargs="*", default=[17])
    parser.add_argument("--teacher-feature-pooling", choices=("mean", "concat"), default="mean")
    parser.add_argument("--task-count", type=int, default=32768)
    parser.add_argument("--eval-task-count", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="artifacts/distillation_cache/gemma_e2b_l17")
    parser.add_argument("--refinement-steps", type=int, default=6)
    parser.add_argument("--teacher-probe-epochs", type=int, default=5)
    parser.add_argument("--teacher-probe-learning-rate", type=float, default=1e-3)
    parser.add_argument("--teacher-probe-batch-size", type=int, default=64)
    parser.add_argument("--teacher-max-length", type=int, default=768)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = build_teacher_target_cache(
        DistillationCacheConfig(
            foundation_model_id=args.foundation_model_id,
            teacher_layer_indices=tuple(args.teacher_layer_indices),
            teacher_feature_pooling=args.teacher_feature_pooling,
            task_count=args.task_count,
            eval_task_count=args.eval_task_count,
            seed=args.seed,
            device=args.device,
            output_dir=args.output_dir,
            refinement_steps=args.refinement_steps,
            teacher_probe_epochs=args.teacher_probe_epochs,
            teacher_probe_learning_rate=args.teacher_probe_learning_rate,
            teacher_probe_batch_size=args.teacher_probe_batch_size,
            teacher_max_length=args.teacher_max_length,
        )
    )
    print(format_teacher_cache_summary(metadata))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
