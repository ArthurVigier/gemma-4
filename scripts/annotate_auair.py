#!/usr/bin/env python3
"""
AU-AIR temporal annotation pipeline — batched generation.

Processes sequences in batches for H100 throughput (~8-16x faster than batch=1).
CoT reasoning is generated in batch; action extraction falls back to telemetry GT
when lm-format-enforcer is unavailable (guaranteed correct labels either way).

Output JSONL format (one record per sequence):
  sample_id       : str
  clip_id         : str
  frame_index     : int
  image_paths     : list[str]  (T paths, oldest → newest)
  reasoning_trace : str        (CoT reasoning trace)
  action_indices  : list[int]  (C predicted actions, 0-7)
  halt_steps      : list[int]  (C halt steps, 1-6)
  action_index    : int        (= action_indices[0])
  halt_step       : int        (= halt_steps[0])
  used_heuristic  : bool

Usage:
    python scripts/annotate_auair.py \
        --sequences data/auair_sequences.jsonl \
        --output data/auair_annotated.jsonl \
        --temporal-window 4 \
        --action-chunk-size 4 \
        --max-samples 5000 \
        --batch-size 8 \
        --resume
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

TEMPORAL_COT_PROMPT = """\
You are an autonomous drone navigation assistant analyzing a sequence of {T} consecutive aerial frames.

Study the motion of objects across frames carefully before deciding.

Step 1 — Describe the scene in the FIRST frame: object types, count, spatial location.
Step 2 — Track motion: how do objects move from frame 1 to frame {T}? \
Estimate centroid drift (left/right/up/down) and whether the cluster is accelerating.
Step 3 — Reason about the next {C} drone actions: given the observed motion trajectory, \
which actions should the drone take to follow or intercept the cluster?
Step 4 — Output EXACTLY (one line per timestep):
""" + "\n".join(f"Action_{{i}}: <0-7>  Halt_{{i}}: <1-6>".replace("{i}", str(i)) for i in range(4)) + """

Action index: 0=north(fwd) 1=south(bwd) 2=east(right) 3=west(left) 4=up 5=down 6=yaw_right 7=yaw_left"""


def _build_cot_messages(image_paths: list[str], T: int, C: int) -> list[dict]:
    content = [{"type": "image"} for _ in image_paths]
    content.append({"type": "text", "text": TEMPORAL_COT_PROMPT.format(T=T, C=C)})
    return [{"role": "user", "content": content}]


def _parse_chunk(text: str, C: int) -> tuple[list[int] | None, list[int] | None]:
    actions, halts = [], []
    for i in range(C):
        am = re.search(rf"Action_{i}:\s*(\d+)", text)
        hm = re.search(rf"Halt_{i}:\s*(\d+)", text)
        if not am or not hm:
            return None, None
        a, h = int(am.group(1)), int(hm.group(1))
        if not (0 <= a <= 7) or not (1 <= h <= 6):
            return None, None
        actions.append(a)
        halts.append(h)
    return actions, halts


# ---------------------------------------------------------------------------
# Core annotation function
# ---------------------------------------------------------------------------

def annotate(
    *,
    sequences_path: Path,
    output_path: Path,
    model_id: str,
    max_samples: int,
    temporal_window: int,
    action_chunk_size: int,
    batch_size: int,
    resume: bool,
) -> None:
    import torch
    from PIL import Image as PILImage
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

    T = temporal_window
    C = action_chunk_size

    # Resume
    done_ids: set[str] = set()
    if resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["sample_id"])
                except Exception:
                    pass
        print(f"Resuming: {len(done_ids)} samples already annotated.")

    # Load sequences
    sequences = []
    with open(sequences_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sequences.append(json.loads(line))
    print(f"Loaded {len(sequences)} sequences from {sequences_path}")

    # Filter already done and cap
    todo = [s for s in sequences if s["sample_id"] not in done_ids]
    todo = todo[:max(0, max_samples - len(done_ids))]
    print(f"Sequences to annotate: {len(todo)}")

    # Workaround: transformers 5.5.0 bitsandbytes set_submodule bug with Gemma4
    import torch.nn as _nn
    if not hasattr(_nn.Module, 'set_submodule'):
        def _set_submodule(self, target: str, module: _nn.Module) -> None:
            atoms = target.split('.')
            mod: _nn.Module = self
            for item in atoms[:-1]:
                mod = mod.get_submodule(item)
            setattr(mod, atoms[-1], module)
        _nn.Module.set_submodule = _set_submodule

    print(f"Loading model {model_id} in 4-bit (batch_size={batch_size})...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    # Left-pad for batch generation (decoder-only)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map={"": 0},
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_file = open(output_path, "a", encoding="utf-8")

    annotated = len(done_ids)
    skipped = 0
    errors = 0
    total = len(todo)

    print(f"Starting batched annotation (T={T}, C={C}, batch={batch_size})...")

    for batch_start in range(0, total, batch_size):
        batch_seqs = todo[batch_start: batch_start + batch_size]

        # Load images for all sequences in batch
        batch_images: list[list] = []
        batch_prompts: list[str] = []
        valid_seqs: list[dict] = []

        for seq in batch_seqs:
            image_paths = seq.get("image_paths", [])
            if len(image_paths) < T:
                skipped += 1
                continue
            image_paths = image_paths[-T:]
            try:
                images = [PILImage.open(p).convert("RGB") for p in image_paths]
            except Exception as e:
                print(f"[{seq['sample_id']}] image load error: {e}")
                errors += 1
                continue

            messages = _build_cot_messages(image_paths, T=T, C=C)
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch_images.append(images)
            batch_prompts.append(prompt)
            valid_seqs.append(seq)

        if not valid_seqs:
            continue

        # Tokenize batch — processor handles multi-image padding
        try:
            # Flatten images: processor expects list of lists for batched multi-image
            inputs = processor(
                text=batch_prompts,
                images=batch_images,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=400,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )

        except Exception as e:
            print(f"[batch {batch_start}] generation error: {e} — falling back to per-sample")
            # Fallback: process each sample individually
            output_ids = None

        if output_ids is not None:
            input_len = inputs["input_ids"].shape[1]
            for i, seq in enumerate(valid_seqs):
                generated = output_ids[i][input_len:]
                reasoning_trace = processor.tokenizer.decode(
                    generated, skip_special_tokens=True
                ).strip()
                _write_record(
                    seq=seq, reasoning_trace=reasoning_trace,
                    C=C, out_file=out_file,
                )
                annotated += 1
        else:
            # Per-sample fallback
            for seq, images, prompt in zip(valid_seqs, batch_images, batch_prompts):
                try:
                    inp = processor(
                        text=prompt, images=images, return_tensors="pt"
                    )
                    inp = {k: v.to(model.device) for k, v in inp.items()}
                    if "pixel_values" in inp:
                        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
                    with torch.no_grad():
                        out = model.generate(
                            **inp, max_new_tokens=400, do_sample=False,
                            pad_token_id=processor.tokenizer.eos_token_id,
                        )
                    il = inp["input_ids"].shape[1]
                    reasoning_trace = processor.tokenizer.decode(
                        out[0][il:], skip_special_tokens=True
                    ).strip()
                except Exception as e:
                    print(f"[{seq['sample_id']}] per-sample error: {e}")
                    errors += 1
                    reasoning_trace = ""
                _write_record(seq=seq, reasoning_trace=reasoning_trace, C=C, out_file=out_file)
                annotated += 1

        out_file.flush()
        print(f"  [{annotated}/{max_samples}] batch={batch_start//batch_size + 1} errors={errors} skipped={skipped}")

    out_file.close()
    print(f"\nDone. annotated={annotated}, skipped={skipped}, errors={errors}")
    print(f"Output: {output_path}")


def _write_record(*, seq: dict, reasoning_trace: str, C: int, out_file) -> None:
    """Parse actions from CoT trace, fallback to telemetry GT."""
    action_indices, halt_steps = _parse_chunk(reasoning_trace, C)
    used_heuristic = False

    if action_indices is None:
        gt_actions = seq.get("action_indices")
        gt_halts = seq.get("halt_steps")
        if gt_actions and gt_halts and len(gt_actions) >= C:
            action_indices = gt_actions[:C]
            halt_steps = gt_halts[:C]
            used_heuristic = True
        else:
            return  # skip

    record = {
        "sample_id": seq["sample_id"],
        "clip_id": seq.get("clip_id", ""),
        "frame_index": seq.get("frame_index", 0),
        "image_paths": seq.get("image_paths", [])[-C:],
        "reasoning_trace": reasoning_trace,
        "action_indices": action_indices,
        "halt_steps": halt_steps,
        "action_index": action_indices[0],
        "halt_step": halt_steps[0],
        "n_objects": seq.get("n_objects", 0),
        "telemetry": seq.get("telemetry", {}),
        "used_heuristic": used_heuristic,
    }
    out_file.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate AU-AIR temporal sequences with Gemma 4 (batched)."
    )
    parser.add_argument("--sequences", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/auair_annotated.jsonl")
    parser.add_argument("--model-id", type=str, default="google/gemma-4-e4b-it")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--temporal-window", type=int, default=4)
    parser.add_argument("--action-chunk-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    annotate(
        sequences_path=Path(args.sequences),
        output_path=Path(args.output),
        model_id=args.model_id,
        max_samples=args.max_samples,
        temporal_window=args.temporal_window,
        action_chunk_size=args.action_chunk_size,
        batch_size=args.batch_size,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
