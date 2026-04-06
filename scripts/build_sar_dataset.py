"""
CLI to build the SAR Crisis Dataset.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Add src to path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if SRC_PATH.as_posix() not in sys.path:
    sys.path.insert(0, SRC_PATH.as_posix())

from arc_drone.sar_dataset_generator import CrisisVLMDatasetGenerator

def main():
    parser = argparse.ArgumentParser(description="Build SAR Training Dataset.")
    parser.add_argument("--count", type=int, default=2000, help="Number of samples.")
    parser.add_argument("--output", default="data/sar_training_vlm.jsonl")
    args = parser.parse_args()

    builder = CrisisVLMDatasetGenerator(output_path=args.output)
    path = builder.build_dataset(count=args.count)
    
    print(f"Successfully generated {args.count} SAR training samples at {path}")

if __name__ == "__main__":
    main()
