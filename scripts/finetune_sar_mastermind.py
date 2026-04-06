"""
CLI entrypoint for fine-tuning the Gemma-4 26B MoE SAR Mastermind.

Usage:
  ./unsloth_env/bin/python scripts/finetune_sar_mastermind.py \
      --dataset data/sar_training_vlm.jsonl \
      --epochs 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if SRC_PATH.as_posix() not in sys.path:
    sys.path.insert(0, SRC_PATH.as_posix())

from arc_drone.sar_finetuning_unsloth import finetune_sar_mastermind

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma-4 26B MoE for SAR.")
    parser.add_argument("--model-id", default="unsloth/gemma-4-26b-a4b-it")
    parser.add_argument("--dataset", default="data/sar_training_vlm.jsonl")
    parser.add_argument("--output-dir", default="artifacts/mastermind_sar_lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    args = parser.parse_args()

    finetune_sar_mastermind(
        model_id = args.model_id,
        dataset_path = args.dataset,
        output_dir = args.output_dir,
        epochs = args.epochs,
        batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum
    )

if __name__ == "__main__":
    main()
