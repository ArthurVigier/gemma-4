#!/usr/bin/env python3
"""
AU-AIR temporal annotation pipeline.

Passes T consecutive frames from AU-AIR sequences to Gemma 4 video encoder,
generates chain-of-thought reasoning traces that explicitly reason about
object motion across frames, then predicts an action chunk [a_t..a_t+C-1].

Output JSONL format (one record per sequence):
  sample_id       : str
  clip_id         : str
  frame_index     : int
  image_paths     : list[str]  (T paths, oldest → newest)
  reasoning_trace : str        (CoT including temporal motion reasoning)
  action_indices  : list[int]  (C predicted actions)
  halt_steps      : list[int]  (C halt steps)
  action_index    : int        (= action_indices[0], convenience)
  halt_step       : int        (= halt_steps[0], convenience)
  used_heuristic  : bool

Usage:
    python scripts/annotate_auair.py \
        --sequences data/auair_sequences.jsonl \
        --output data/auair_annotated.jsonl \
        --temporal-window 4 \
        --action-chunk-size 4 \
        --max-samples 5000 \
        --resume
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

TEMPORAL_USER_PROMPT = """\
You are an autonomous drone navigation assistant analyzing a sequence of {T} consecutive aerial frames.

Study the motion of objects across frames carefully before deciding.

Step 1 — Describe the scene in the FIRST frame: object types, count, spatial location.
Step 2 — Track motion: how do objects move from frame 1 to frame {T}? \
Estimate centroid drift (left/right/up/down) and whether the cluster is accelerating.
Step 3 — Predict the next {C} drone actions: given the observed motion trajectory, \
which actions (one per future timestep) should the drone take to follow or intercept the cluster?
Step 4 — Assess confidence for each action (1=very confident, 6=very uncertain).

Output EXACTLY in this format (one line per timestep):
Action_0: <0-7>  Halt_0: <1-6>
Action_1: <0-7>  Halt_1: <1-6>
Action_2: <0-7>  Halt_2: <1-6>
Action_3: <0-7>  Halt_3: <1-6>

Action index meanings:
0=north(fwd) 1=south(bwd) 2=east(right) 3=west(left) 4=up 5=down 6=yaw_right 7=yaw_left
"""

def _build_messages(image_paths: list[str], T: int, C: int) -> list[dict]:
    """Build multimodal message with T image tokens + temporal prompt."""
    content = []
    for _ in image_paths:
        content.append({"type": "image"})
    content.append({"type": "text", "text": TEMPORAL_USER_PROMPT.format(T=T, C=C)})
    return [{"role": "user", "content": content}]


def _parse_chunk(text: str, C: int) -> tuple[list[int] | None, list[int] | None]:
    """Parse Action_i / Halt_i lines from model output."""
    actions, halts = [], []
    for i in range(C):
        a_match = re.search(rf"Action_{i}:\s*(\d+)", text)
        h_match = re.search(rf"Halt_{i}:\s*(\d+)", text)
        if not a_match or not h_match:
            return None, None
        a = int(a_match.group(1))
        h = int(h_match.group(1))
        if not (0 <= a <= 7) or not (1 <= h <= 6):
            return None, None
        actions.append(a)
        halts.append(h)
    return actions, halts


def annotate(
    *,
    sequences_path: Path,
    output_path: Path,
    model_id: str,
    max_samples: int,
    temporal_window: int,
    action_chunk_size: int,
    resume: bool,
) -> None:
    import torch
    from PIL import Image as PILImage
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

    # Resume: load already-done IDs
    done_ids: set[str] = set()
    if resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["sample_id"])
                except Exception:
                    pass
        print(f"Resuming: {len(done_ids)} samples already annotated.")

    # Load sequences from parse_auair.py output
    sequences = []
    with open(sequences_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sequences.append(json.loads(line))
    print(f"Loaded {len(sequences)} sequences from {sequences_path}")

    print(f"Loading model {model_id} in 4-bit...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_file = open(output_path, "a", encoding="utf-8")

    annotated = len(done_ids)
    skipped = 0
    errors = 0
    T = temporal_window
    C = action_chunk_size

    print(f"Annotating up to {max_samples} sequences (T={T}, C={C})...")

    for seq in sequences:
        if annotated >= max_samples:
            break

        sample_id = seq["sample_id"]
        if sample_id in done_ids:
            continue

        image_paths = seq.get("image_paths", [])
        if len(image_paths) < T:
            skipped += 1
            continue

        # Load T images
        try:
            images = [PILImage.open(p).convert("RGB") for p in image_paths[-T:]]
        except Exception as e:
            print(f"[{sample_id}] image load error: {e}")
            errors += 1
            continue

        messages = _build_messages(image_paths[-T:], T=T, C=C)

        try:
            prompt_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=prompt_text,
                images=images,
                return_tensors="pt",
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=400,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            generated = output_ids[0][input_len:]
            response_text = processor.tokenizer.decode(generated, skip_special_tokens=True)

        except Exception as e:
            print(f"[{sample_id}] generation error: {e}")
            errors += 1
            continue

        action_indices, halt_steps = _parse_chunk(response_text, C)
        used_heuristic = False

        # Fallback to ground-truth telemetry actions from parse_auair output
        if action_indices is None:
            gt_actions = seq.get("action_indices")
            gt_halts = seq.get("halt_steps")
            if gt_actions and gt_halts and len(gt_actions) >= C:
                action_indices = gt_actions[:C]
                halt_steps = gt_halts[:C]
                used_heuristic = True
            else:
                skipped += 1
                continue

        record = {
            "sample_id": sample_id,
            "clip_id": seq.get("clip_id", ""),
            "frame_index": seq.get("frame_index", 0),
            "image_paths": image_paths[-T:],
            "reasoning_trace": response_text.strip(),
            "action_indices": action_indices,
            "halt_steps": halt_steps,
            "action_index": action_indices[0],
            "halt_step": halt_steps[0],
            "n_objects": seq.get("n_objects", 0),
            "telemetry": seq.get("telemetry", {}),
            "used_heuristic": used_heuristic,
        }
        out_file.write(json.dumps(record) + "\n")
        out_file.flush()
        annotated += 1

        if annotated % 50 == 0:
            print(f"  [{annotated}/{max_samples}] errors={errors} skipped={skipped}")

    out_file.close()
    print(f"\nDone. annotated={annotated}, skipped={skipped}, errors={errors}")
    print(f"Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate AU-AIR temporal sequences with Gemma 4 video encoder."
    )
    parser.add_argument("--sequences", type=str, required=True,
                        help="JSONL from parse_auair.py")
    parser.add_argument("--output", type=str, default="data/auair_annotated.jsonl")
    parser.add_argument("--model-id", type=str, default="google/gemma-4-e4b-it")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--temporal-window", type=int, default=4)
    parser.add_argument("--action-chunk-size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    annotate(
        sequences_path=Path(args.sequences),
        output_path=Path(args.output),
        model_id=args.model_id,
        max_samples=args.max_samples,
        temporal_window=args.temporal_window,
        action_chunk_size=args.action_chunk_size,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
