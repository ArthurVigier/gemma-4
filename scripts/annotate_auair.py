#!/usr/bin/env python3
"""
AU-AIR temporal annotation pipeline.

Two-step generation:
  1. Free-form chain-of-thought reasoning over T consecutive frames.
  2. Constrained JSON action chunk via lm-format-enforcer — parse rate 100% by construction.

Output JSONL format (one record per sequence):
  sample_id       : str
  clip_id         : str
  frame_index     : int
  image_paths     : list[str]  (T paths, oldest → newest)
  reasoning_trace : str        (CoT only, no action lines)
  action_indices  : list[int]  (C predicted actions, 0-7)
  halt_steps      : list[int]  (C halt steps, 1-6)
  action_index    : int        (= action_indices[0], convenience)
  halt_step       : int        (= halt_steps[0], convenience)
  used_heuristic  : bool       (True if constrained gen failed and telemetry was used)

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
Step 4 — Assess confidence for each action (1=very confident, 6=very uncertain).

Action index: 0=north(fwd) 1=south(bwd) 2=east(right) 3=west(left) 4=up 5=down 6=yaw_right 7=yaw_left\
"""

ACTION_JSON_SUFFIX = "Based on your analysis, provide your {C} actions as a compact JSON object:"


# ---------------------------------------------------------------------------
# Dynamic Pydantic schema for the action chunk
# ---------------------------------------------------------------------------

def _make_chunk_model(C: int):
    """Return a Pydantic model with action_i (0-7) and halt_i (1-6) fields for i in 0..C-1."""
    try:
        from pydantic import Field, create_model
    except ImportError as exc:
        raise ImportError("pydantic is required: pip install pydantic") from exc

    fields: dict = {}
    for i in range(C):
        fields[f"action_{i}"] = (int, Field(ge=0, le=7))
        fields[f"halt_{i}"] = (int, Field(ge=1, le=6))
    return create_model("ActionChunk", **fields)


def _build_cot_messages(image_paths: list[str], T: int, C: int) -> list[dict]:
    """Build multimodal message with T image tokens + CoT prompt (no format constraints)."""
    content = [{"type": "image"} for _ in image_paths]
    content.append({"type": "text", "text": TEMPORAL_COT_PROMPT.format(T=T, C=C)})
    return [{"role": "user", "content": content}]


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
    resume: bool,
) -> None:
    import torch
    from PIL import Image as PILImage
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

    try:
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )
        _lmfe_available = True
    except ImportError:
        print(
            "WARNING: lm-format-enforcer not found. "
            "Falling back to regex parsing (lower parse rate).\n"
            "Install with: pip install lm-format-enforcer"
        )
        _lmfe_available = False

    T = temporal_window
    C = action_chunk_size

    # Build pydantic schema and lm-format-enforcer parser once
    ActionChunk = _make_chunk_model(C)
    if _lmfe_available:
        _json_parser = JsonSchemaParser(ActionChunk.model_json_schema())

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

    # Load sequences
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

    # Build prefix_allowed_tokens_fn from the tokenizer (built once, reused per sample)
    if _lmfe_available:
        _prefix_fn = build_transformers_prefix_allowed_tokens_fn(
            processor.tokenizer, _json_parser
        )
    else:
        _prefix_fn = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_file = open(output_path, "a", encoding="utf-8")

    annotated = len(done_ids)
    skipped = 0
    errors = 0

    print(f"Annotating up to {max_samples} sequences (T={T}, C={C})...")
    print(f"Constrained decoding: {'enabled (lm-format-enforcer)' if _lmfe_available else 'disabled (regex fallback)'}")

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

        image_paths = image_paths[-T:]

        try:
            images = [PILImage.open(p).convert("RGB") for p in image_paths]
        except Exception as e:
            print(f"[{sample_id}] image load error: {e}")
            errors += 1
            continue

        # ── Step 1: free-form CoT generation ─────────────────────────────────
        cot_messages = _build_cot_messages(image_paths, T=T, C=C)
        try:
            cot_prompt = processor.apply_chat_template(
                cot_messages, tokenize=False, add_generation_prompt=True
            )
            cot_inputs = processor(
                text=cot_prompt,
                images=images,
                return_tensors="pt",
            )
            cot_inputs = {k: v.to(model.device) for k, v in cot_inputs.items()}
            if "pixel_values" in cot_inputs:
                cot_inputs["pixel_values"] = cot_inputs["pixel_values"].to(torch.bfloat16)

            with torch.no_grad():
                cot_ids = model.generate(
                    **cot_inputs,
                    max_new_tokens=350,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )

            cot_input_len = cot_inputs["input_ids"].shape[1]
            reasoning_trace = processor.tokenizer.decode(
                cot_ids[0][cot_input_len:], skip_special_tokens=True
            ).strip()

        except Exception as e:
            print(f"[{sample_id}] CoT generation error: {e}")
            errors += 1
            continue

        # ── Step 2: constrained JSON action chunk ─────────────────────────────
        action_indices = None
        halt_steps = None
        used_heuristic = False

        if _lmfe_available:
            try:
                # Append CoT output + JSON request to the conversation
                json_suffix = ACTION_JSON_SUFFIX.format(C=C)
                json_prompt = cot_prompt + reasoning_trace + "\n" + json_suffix
                json_inputs = processor(
                    text=json_prompt,
                    images=images,
                    return_tensors="pt",
                )
                json_inputs = {k: v.to(model.device) for k, v in json_inputs.items()}
                if "pixel_values" in json_inputs:
                    json_inputs["pixel_values"] = json_inputs["pixel_values"].to(torch.bfloat16)

                with torch.no_grad():
                    json_ids = model.generate(
                        **json_inputs,
                        max_new_tokens=120,
                        do_sample=False,
                        prefix_allowed_tokens_fn=_prefix_fn,
                        pad_token_id=processor.tokenizer.eos_token_id,
                    )

                json_input_len = json_inputs["input_ids"].shape[1]
                json_text = processor.tokenizer.decode(
                    json_ids[0][json_input_len:], skip_special_tokens=True
                ).strip()

                chunk = ActionChunk.model_validate_json(json_text)
                action_indices = [getattr(chunk, f"action_{i}") for i in range(C)]
                halt_steps = [getattr(chunk, f"halt_{i}") for i in range(C)]

            except Exception as e:
                print(f"[{sample_id}] constrained JSON generation error: {e}")
                # fall through to telemetry fallback below

        else:
            # Regex fallback (legacy path, lower parse rate)
            import re
            actions_found, halts_found = [], []
            for i in range(C):
                am = re.search(rf"Action_{i}:\s*(\d+)", reasoning_trace)
                hm = re.search(rf"Halt_{i}:\s*(\d+)", reasoning_trace)
                if am and hm:
                    a, h = int(am.group(1)), int(hm.group(1))
                    if (0 <= a <= 7) and (1 <= h <= 6):
                        actions_found.append(a)
                        halts_found.append(h)
            if len(actions_found) == C:
                action_indices = actions_found
                halt_steps = halts_found

        # ── Telemetry fallback if all else fails ──────────────────────────────
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
            "image_paths": image_paths,
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
        out_file.flush()
        annotated += 1

        if annotated % 50 == 0:
            print(f"  [{annotated}/{max_samples}] errors={errors} skipped={skipped}")

    out_file.close()
    print(f"\nDone. annotated={annotated}, skipped={skipped}, errors={errors}")
    print(f"Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate AU-AIR temporal sequences with Gemma 4 (two-step: CoT + constrained JSON)."
    )
    parser.add_argument("--sequences", type=str, required=True, help="JSONL from parse_auair.py")
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
