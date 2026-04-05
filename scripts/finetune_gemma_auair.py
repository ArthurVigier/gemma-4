#!/usr/bin/env python3
"""
Fine-tune Gemma 4 E4B teacher on AU-AIR with GT telemetry labels.

No circular annotation — labels come directly from IMU/velocity telemetry
parsed by parse_auair.py. This is the clean training path.

Usage:
    python scripts/finetune_gemma_auair.py \
        --auair-path data/auair_sequences.jsonl \
        --epochs 3 \
        --learning-rate 2e-4
"""
import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from arc_drone.teacher_finetuning_auair import AuAirTeacherConfig, finetune_auair_teacher


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune Gemma 4 teacher on AU-AIR GT telemetry (clean, no circular annotation)."
    )
    parser.add_argument("--foundation-model-id", default="google/gemma-4-e4b-it")
    parser.add_argument("--auair-path", required=True, help="JSONL from parse_auair.py")
    parser.add_argument("--temporal-window", type=int, default=4)
    parser.add_argument("--action-chunk-size", type=int, default=4)
    parser.add_argument("--task-count", type=int, default=25000)
    parser.add_argument("--eval-task-count", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="artifacts/teacher_lora/gemma_e4b_auair")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    config = AuAirTeacherConfig(
        foundation_model_id=args.foundation_model_id,
        auair_path=args.auair_path,
        temporal_window=args.temporal_window,
        action_chunk_size=args.action_chunk_size,
        task_count=args.task_count,
        eval_task_count=args.eval_task_count,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        seed=args.seed,
        output_dir=args.output_dir,
        max_length=args.max_length,
    )

    logger.info(
        "Starting AU-AIR teacher fine-tuning | model=%s  data=%s  epochs=%d  lr=%.2e",
        args.foundation_model_id, args.auair_path, args.epochs, args.learning_rate,
    )
    summary = finetune_auair_teacher(config)
    logger.info("Fine-tuning complete — best eval loss: %.4f", summary["best_eval_loss"])
    print(f"\nDone. Best eval loss: {summary['best_eval_loss']:.4f}")
    print(f"Adapters saved to: {summary['output_dir']}")


if __name__ == "__main__":
    main()
