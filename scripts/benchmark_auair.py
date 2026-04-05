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
  python scripts/benchmark_auair.py \\
    --sequences data/auair_sequences.jsonl \\
    --model-id google/gemma-4-e4b-it

  # Full comparison
  python scripts/benchmark_auair.py \\
    --sequences data/auair_sequences.jsonl \\
    --model-id google/gemma-4-e4b-it \\
    --lora-path artifacts/teacher_lora/gemma_e4b_auair \\
    --trm-checkpoint artifacts/checkpoints/trm_student/best_student.pt \\
    --tta-shots 16 \\
    --n-eval 500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-model AU-AIR benchmark (vanilla → LoRA → TRM → TTA)."
    )
    parser.add_argument("--sequences", required=True,
                        help="JSONL from parse_auair.py or auair_sequences.jsonl")
    parser.add_argument("--model-id", default="google/gemma-4-e4b-it",
                        help="HF model ID for Gemma 4")
    parser.add_argument("--lora-path", default=None,
                        help="Path to fine-tuned LoRA adapters (finetune_gemma_auair.py output)")
    parser.add_argument("--trm-checkpoint", default=None,
                        help="Path to TRM student .pt checkpoint")
    parser.add_argument("--tta-shots", type=int, default=0,
                        help="If >0, also run test-time adaptation with this many adapt shots")
    parser.add_argument("--tta-steps", type=int, default=20,
                        help="Gradient steps for test-time adaptation")
    parser.add_argument("--n-eval", type=int, default=300,
                        help="Number of test sequences to evaluate")
    parser.add_argument("--n-adapt", type=int, default=None,
                        help="Number of sequences reserved for TTA adaptation (default=tta-shots)")
    parser.add_argument("--temporal-window", type=int, default=4)
    parser.add_argument("--action-chunk-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/benchmark_auair.json",
                        help="JSON output path for results")
    parser.add_argument("--skip-vanilla", action="store_true",
                        help="Skip vanilla Gemma baseline (saves ~10min if you only care about LoRA/TRM)")
    args = parser.parse_args()

    T, C = args.temporal_window, args.action_chunk_size
    sequences_path = Path(args.sequences)

    # Load and split: adapt pool (for TTA) + eval pool
    n_adapt = args.n_adapt or args.tta_shots
    total_needed = args.n_eval + n_adapt
    all_seqs = _load_sequences(sequences_path, max_samples=total_needed, seed=args.seed)
    adapt_seqs = all_seqs[:n_adapt]
    eval_seqs = all_seqs[n_adapt: n_adapt + args.n_eval]
    print(f"Loaded {len(eval_seqs)} eval sequences, {len(adapt_seqs)} adapt sequences")
    print(f"GT heuristic rate in eval: {sum(s.get('used_heuristic', False) for s in eval_seqs)/max(len(eval_seqs),1)*100:.1f}%")

    results: list[ModelResult] = []

    # ── 1. Vanilla baseline ──────────────────────────────────────────────────
    if not args.skip_vanilla:
        print("\n[1/4] Vanilla Gemma 4 (zero-shot baseline)")
        r = evaluate_gemma(
            sequences=eval_seqs,
            model_id=args.model_id,
            lora_path=None,
            temporal_window=T,
            action_chunk_size=C,
            model_name="gemma4_vanilla",
        )
        results.append(r)
        print(r.summary())
    else:
        print("\n[1/4] Skipping vanilla baseline (--skip-vanilla)")

    # ── 2. Fine-tuned LoRA ───────────────────────────────────────────────────
    if args.lora_path:
        print("\n[2/4] Gemma 4 + LoRA (fine-tuned on GT telemetry)")
        r = evaluate_gemma(
            sequences=eval_seqs,
            model_id=args.model_id,
            lora_path=args.lora_path,
            temporal_window=T,
            action_chunk_size=C,
            model_name="gemma4_lora",
        )
        results.append(r)
        print(r.summary())
    else:
        print("\n[2/4] Skipping LoRA eval (--lora-path not set)")

    # ── 3. TRM student ───────────────────────────────────────────────────────
    if args.trm_checkpoint:
        print("\n[3/4] TRM student")
        r = evaluate_trm(
            sequences=eval_seqs,
            checkpoint_path=args.trm_checkpoint,
            temporal_window=T,
            action_chunk_size=C,
            model_name="trm_student",
        )
        results.append(r)
        print(r.summary())
    else:
        print("\n[3/4] Skipping TRM eval (--trm-checkpoint not set)")

    # ── 4. Test-time LoRA adaptation (OOD resilience) ────────────────────────
    if args.tta_shots > 0 and args.lora_path:
        print(f"\n[4/4] Gemma 4 + LoRA + TTA ({args.tta_shots} shots, {args.tta_steps} steps)")
        r = adapt_and_evaluate_gemma(
            adapt_sequences=adapt_seqs[:args.tta_shots],
            eval_sequences=eval_seqs,
            model_id=args.model_id,
            base_lora_path=args.lora_path,
            adapt_lr=5e-5,
            adapt_steps=args.tta_steps,
            temporal_window=T,
            action_chunk_size=C,
        )
        results.append(r)
        print(r.summary())
    else:
        print("\n[4/4] Skipping TTA (--tta-shots 0 or no --lora-path)")

    if not results:
        print("\nNo models evaluated. Pass at least --model-id for vanilla baseline.")
        return

    # ── Print comparison ─────────────────────────────────────────────────────
    print_comparison(results, C)

    # ── Save JSON ────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "sequences_path": str(sequences_path),
        "n_eval": len(eval_seqs),
        "n_adapt": len(adapt_seqs),
        "temporal_window": T,
        "action_chunk_size": C,
        "results": [
            {
                "model_name": r.model_name,
                "action_acc": round(r.action_acc, 2),
                "chunk_acc": [round(v, 2) for v in r.chunk_acc],
                "parse_rate": round(r.parse_rate, 2),
                "halt_acc": round(r.halt_acc, 2),
                "ms_per_sample": round(r.ms_per_sample, 1),
                "n_samples": r.n_samples,
                "n_heuristic": r.n_heuristic,
                **r.extra,
            }
            for r in results
        ],
    }
    output_path.write_text(json.dumps(report, indent=2))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
