#!/usr/bin/env python3
"""CLI for QLoRA fine-tuning of the Gemma-4 MoE teacher model (26B-A4B) on ARC-Drone tasks."""

import argparse
from pathlib import Path
import sys

# Ensure the src directory is in the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arc_drone.teacher_finetuning_moe import TeacherMoEFinetuneConfig, finetune_teacher_moe


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Gemma-4 MoE teacher model using QLoRA.")
    parser.add_argument("--foundation-model-id", type=str, default="google/gemma-4-26B-A4B-it", help="HF model ID.")
    parser.add_argument("--task-count", type=int, default=25000, help="Number of training tasks.")
    parser.add_argument("--eval-task-count", type=int, default=1000, help="Number of evaluation tasks.")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size (keep low for MoE).")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, help="Steps to accumulate gradients before update.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default="artifacts/teacher_lora/gemma_26b_moe_arc_specialist", help="Output directory for LoRA adapters.")
    parser.add_argument("--real-data-path", type=str, default=None, help="Local JSON/JSONL manifest of real tasks to mix in.")
    parser.add_argument("--real-data-ratio", type=float, default=0.0, help="Fraction of the task pool sourced from real data.")
    parser.add_argument(
        "--real-dataset",
        type=str,
        default=None,
        help="Direct real dataset preset or HF dataset id. Built-in presets: visdrone_det, drone_detection.",
    )
    parser.add_argument("--real-dataset-split", type=str, default="train", help="Split to read from the direct real dataset source.")

    args = parser.parse_args()

    config = TeacherMoEFinetuneConfig(
        foundation_model_id=args.foundation_model_id,
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
        real_data_path=args.real_data_path,
        real_data_ratio=args.real_data_ratio,
        real_dataset=args.real_dataset,
        real_dataset_split=args.real_dataset_split,
    )

    print(f"Starting QLoRA fine-tuning for {config.foundation_model_id} (MoE)...")
    summary = finetune_teacher_moe(config)
    print("\nFine-tuning completed successfully!")
    print(f"Best Eval Loss: {summary['best_eval_loss']:.4f}")
    print(f"Adapters saved to: {summary['output_dir']}")


if __name__ == "__main__":
    main()
