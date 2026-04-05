#!/usr/bin/env python3
"""CLI for QLoRA fine-tuning of the Gemma teacher model on ARC-Drone tasks."""

import argparse
from pathlib import Path
import sys

# Ensure the src directory is in the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arc_drone.teacher_finetuning import TeacherFinetuneConfig, finetune_teacher


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Gemma-4 teacher model using QLoRA.")
    parser.add_argument("--foundation-model-id", type=str, default="google/gemma-4-e4b-it", help="HF model ID.")
    parser.add_argument("--task-count", type=int, default=25000, help="Number of training tasks.")
    parser.add_argument("--eval-task-count", type=int, default=1000, help="Number of evaluation tasks.")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs.")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default="artifacts/teacher_lora/gemma_e4b_arc_specialist", help="Output directory for LoRA adapters.")
    parser.add_argument("--real-data-path", type=str, default=None, help="Local JSON/JSONL manifest of real tasks.")
    parser.add_argument("--real-data-ratio", type=float, default=0.0, help="Fraction of tasks from real data.")
    parser.add_argument("--real-dataset", type=str, default=None, help="HF dataset preset (legacy).")
    parser.add_argument("--real-dataset-split", type=str, default="train", help="Dataset split (legacy).")
    parser.add_argument("--auair-path", type=str, default=None,
                        help="JSONL from annotate_auair.py (T-frame sequences with CoT + action chunk).")
    parser.add_argument("--temporal-window", type=int, default=4, help="Number of input frames per sequence (T).")
    parser.add_argument("--action-chunk-size", type=int, default=4, help="Number of future actions predicted (C).")

    args = parser.parse_args()

    config = TeacherFinetuneConfig(
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
        auair_path=args.auair_path,
        temporal_window=args.temporal_window,
        action_chunk_size=args.action_chunk_size,
    )

    print(f"Starting QLoRA fine-tuning for {config.foundation_model_id}...")
    summary = finetune_teacher(config)
    print("\nFine-tuning completed successfully!")
    print(f"Best Eval Loss: {summary['best_eval_loss']:.4f}")
    print(f"Adapters saved to: {summary['output_dir']}")


if __name__ == "__main__":
    main()
