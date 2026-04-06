"""
CLI entrypoint for Unsloth-based Teacher fine-tuning.
"""

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

from arc_drone.teacher_finetuning_unsloth import AuAirTeacherConfig, train

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 4 teacher using Unsloth.")
    parser.add_argument("--foundation-model-id", default="unsloth/gemma-4-e4b-it")
    parser.add_argument("--auair-path", required=True)
    parser.add_argument("--auair-images-path", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--output-dir", default="artifacts/teacher_lora/gemma_e4b_auair_unsloth")
    args = parser.parse_args()

    config = AuAirTeacherConfig(
        foundation_model_id=args.foundation_model_id,
        auair_path=args.auair_path,
        auair_images_path=args.auair_images_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
    )

    logger.info("Starting Unsloth-optimized fine-tuning...")
    train(config)

if __name__ == "__main__":
    main()
