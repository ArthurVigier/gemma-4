#!/usr/bin/env python3
"""
Multi-model benchmark on AU-AIR sequences — rigorous comparison pipeline.

Evaluates up to 4 configurations against GT telemetry labels:
  1. Gemma 4 vanilla        (zero-shot baseline)
  2. Gemma 4 + LoRA         (fine-tuned on GT via finetune_gemma_auair.py)
  3. TRM student            (distilled, from train_trm_student.py checkpoint)
  4. Gemma 4 + LoRA + TTA  (test-time LoRA adaptation, NVARC OOD resilience)

Each step is optional — pass only the paths you have.

Output: JSON report + stdout comparison table.

Usage examples:
  # Vanilla baseline only
  python scripts/benchmark_auair.py \
    --sequences data/auair_sequences.jsonl \
    --model-id google/gemma-4-e4b-it

  # Full comparison
  python scripts/benchmark_auair.py \
    --sequences data/auair_sequences.jsonl \
    --model-id google/gemma-4-e4b-it \
    --lora-path artifacts/teacher_lora/gemma_e4b_auair \
    --trm-checkpoint artifacts/checkpoints/trm_student/best_student.pt \
    --tta-shots 16 \
    --n-eval 500
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from arc_drone.auair_eval import (
    ModelResult,
    _load_sequences,
    adapt_and_evaluate_gemma,
    evaluate_gemma,
    evaluate_trm,
)


def print_comparison(results: list[ModelResult], C: int) -> None:
    print("\n" + "=" * 70)
    print("AU-AIR BENCHMARK RESULTS")
    print("=" * 70)
    header = f"{'Model':<30} {'Act0':>6} {'Parse':>6} {'Halt':>6} {'ms/s':>7}"
    print(header)
    print("-" * 70)
    for r in results:
        print(
            f"{r.model_name:<30} {r.action_acc:>5.1f}% {r.parse_rate:>5.1f}% "
            f"{r.halt_acc:>5.1f}% {r.ms_per_sample:>6.0f}ms"
        )
    print("=" * 70)

    # Chunk accuracy table
    print(f"\nChunk accuracy per step (Action_0 .. Action_{C-1}):")
    header2 = f"{'Model':<30}" + "".join(f"  a{c:>2}" for c in range(C))
    print(header2)
    print("-" * (30 + C * 6))
    for r in results:
        row = f"{r.model_name:<30}" + "".join(f"{v:>5.1f}%" for v in r.chunk_acc)
        print(row)

    # Delta vs vanilla if multiple models
    if len(results) > 1:
        baseline = results[0]
        print(f"\nDelta vs {baseline.model_name}:")
        for r in results[1:]:
            delta = r.action_acc - baseline.action_acc
            sign = "+" if delta >= 0 else ""
            print(f"  {r.model_name:<28} {sign}{delta:.1f}% action0")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multi-model AU-AIR benchmark (vanilla → LoRA → TRM → TTA)."
    )
    parser.add_argument("--sequences", required=True,
                        help="JSONL from parse_auair.py or auair_sequences.jsonl")
    parser.add_argument("--images-path", default=None,
                        help="Root directory for AU-AIR images. Filenames from JSONL will be resolved relative to this.")
    parser.add_argument("--model-id", default="google/gemma-4-e4b-it",
                        help="HF model ID for Gemma 4")
    parser.add_argument("--lora-path", default=None,
                        help="Path to fine-tuned LoRA adapters (finetune_gemma_auair.py output)")
    parser.add_argument("--trm-checkpoint", default=None,
                        help="Path to student checkpoint (train_trm_student.py output)")
    parser.add_argument("--tta-shots", type=int, default=None,
                        help="If set, performs test-time LoRA adaptation with this many shots.")
    parser.add_argument("--n-eval", type=int, default=500,
                        help="Number of samples to evaluate on (randomly sampled from sequences file).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temporal-window", type=int, default=4)
    parser.add_argument("--action-chunk-size", type=int, default=4)
    parser.add_argument("--output-json", default="artifacts/auair_benchmark.json")
    parser.add_argument("--skip-vanilla", action="store_true", help="Skip the un-finetuned baseline.")
    args = parser.parse_args()

    # Create output dir if needed
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)

    # Load shared sequence list
    seq_all = _load_sequences(Path(args.sequences), args.n_eval, args.seed)
    results = []

    # 1. Vanilla Gemma
    if not args.skip_vanilla:
        res_vanilla = evaluate_gemma(
            sequences=seq_all,
            model_id=args.model_id,
            temporal_window=args.temporal_window,
            action_chunk_size=args.action_chunk_size,
            images_path=args.images_path,
        )
        results.append(res_vanilla)

    # 2. LoRA Gemma (if path provided)
    if args.lora_path:
        res_lora = evaluate_gemma(
            sequences=seq_all,
            model_id=args.model_id,
            lora_path=args.lora_path,
            temporal_window=args.temporal_window,
            action_chunk_size=args.action_chunk_size,
            images_path=args.images_path,
        )
        results.append(res_lora)

    # 3. TRM Reasoner (if checkpoint provided)
    if args.trm_checkpoint:
        res_trm = evaluate_trm(
            sequences=seq_all,
            checkpoint_path=args.trm_checkpoint,
            temporal_window=args.temporal_window,
            action_chunk_size=args.action_chunk_size,
            images_path=args.images_path,
        )
        results.append(res_trm)

    # 4. TTA adaptation (if shots requested)
    if args.tta_shots and args.lora_path:
        # Split: use first K for adaptation, rest for evaluation
        # Note: we use the full set for eval but hide ground truth for those not in adapt set.
        # For simplicity in this script, we take a disjoint evaluation set from the same file.
        rng = np.random.default_rng(args.seed + 100)
        all_recs = []
        with open(args.sequences, encoding="utf-8") as f:
            for line in f:
                all_recs.append(json.loads(line))
        
        perm = rng.permutation(len(all_recs))
        adapt_idx = perm[:args.tta_shots]
        eval_idx = perm[args.tta_shots: args.tta_shots + args.n_eval]
        
        adapt_seq = [all_recs[i] for i in adapt_idx]
        eval_seq = [all_recs[i] for i in eval_idx]

        res_tta = adapt_and_evaluate_gemma(
            adapt_sequences=adapt_seq,
            eval_sequences=eval_seq,
            model_id=args.model_id,
            base_lora_path=args.lora_path,
            temporal_window=args.temporal_window,
            action_chunk_size=args.action_chunk_size,
        )
        results.append(res_tta)

    # Print summary table
    print_comparison(results, args.action_chunk_size)

    # Save to JSON
    report = {
        "config": vars(args),
        "results": [
            {
                "model_name": r.model_name,
                "action_acc": r.action_acc,
                "chunk_acc": r.chunk_acc,
                "parse_rate": r.parse_rate,
                "halt_acc": r.halt_acc,
                "ms_per_sample": r.ms_per_sample,
                "n_samples": r.n_samples,
                "n_heuristic": r.n_heuristic,
                "extra": r.extra,
            }
            for r in results
        ]
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Full report saved to %s", args.output_json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
