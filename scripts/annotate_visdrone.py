#!/usr/bin/env python3
"""
Auto-annotation pipeline: runs Gemma E4B (zero-shot) over VisDrone images to generate
(image_path, reasoning_trace, action_index, halt_step) JSONL for teacher fine-tuning.

This replaces the synthetic ARC proxy with real drone scene understanding.
The model sees a drone image and must reason about where the primary object cluster
is, then predict the correct drone action to move toward it.

Usage:
    python scripts/annotate_visdrone.py \
        --output data/visdrone_annotated.jsonl \
        --max-samples 5000 \
        --model-id google/gemma-4-e4b-it
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ---------------------------------------------------------------------------
# Action vocabulary (mirrors student_training.ACTION_VOCABULARY indices)
# 0: north (+x), 1: south (-x), 2: east (+y), 3: west (-y)
# 4: up (+z), 5: down (-z), 6: yaw+, 7: yaw-
# ---------------------------------------------------------------------------
ACTION_LABELS = {
    0: "north (move forward / +x)",
    1: "south (move backward / -x)",
    2: "east (move right / +y)",
    3: "west (move left / -y)",
    4: "up (ascend / +z)",
    5: "down (descend / -z)",
    6: "yaw_right (rotate clockwise)",
    7: "yaw_left (rotate counter-clockwise)",
}

SYSTEM_PROMPT = """\
You are an autonomous drone navigation assistant.
You receive a drone camera image and must reason step-by-step about where the \
primary object cluster (vehicles, pedestrians, targets) is located, then decide \
the best single drone action to move toward or center on that cluster.

Reason explicitly. Then output EXACTLY this format on the last two lines:
Action: <integer 0-7>
Halt: <integer 1-6>

Action index meanings:
0 = north (forward, +x)
1 = south (backward, -x)
2 = east (right, +y)
3 = west (left, -y)
4 = up (ascend, +z)
5 = down (descend, -z)
6 = yaw_right (rotate clockwise)
7 = yaw_left (rotate counter-clockwise)

Halt step (1-6): how many refinement steps until confident enough to stop.
1 = very confident (clear single cluster), 6 = very uncertain (sparse or ambiguous scene).
"""

ANNOTATION_PROMPT = """\
Analyze this drone aerial image.

Step 1 — Describe what you see: object types, approximate count, spatial distribution.
Step 2 — Identify the dominant cluster: where is it in the frame (left/right/center, top/bottom/center)?
Step 3 — Decide the action: which of the 8 actions best moves the drone toward the dominant cluster?
Step 4 — Assess confidence: how clear is the target? (1=very clear, 6=very ambiguous)

Then output:
Action: <0-7>
Halt: <1-6>
"""


def _pil_to_base64(image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _parse_action_halt(text: str) -> tuple[int | None, int | None]:
    import re
    action_match = re.search(r"Action:\s*(\d+)", text)
    halt_match = re.search(r"Halt:\s*(\d+)", text)
    action = int(action_match.group(1)) if action_match else None
    halt = int(halt_match.group(1)) if halt_match else None
    if action is not None and not (0 <= action <= 7):
        action = None
    if halt is not None and not (1 <= halt <= 6):
        halt = None
    return action, halt


def _heuristic_action_from_bboxes(
    bboxes: list, categories: list, width: float, height: float
) -> tuple[int, int]:
    """Fallback: compute action from bbox centroid when model output is unparseable."""
    centers_x, centers_y = [], []
    for bbox in bboxes:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x, y, w, h = [float(v) for v in bbox]
        centers_x.append(x + w / 2.0)
        centers_y.append(y + h / 2.0)
    if not centers_x:
        return 6, 6  # yaw_right, low confidence

    import numpy as np
    x_bias = (float(np.mean(centers_x)) / max(width, 1.0)) - 0.5
    y_bias = 0.5 - (float(np.mean(centers_y)) / max(height, 1.0))
    n = len(centers_x)
    halt = max(1, min(6, 6 - min(n, 5)))

    if abs(x_bias) >= abs(y_bias):
        return (2 if x_bias >= 0 else 3), halt   # east / west
    return (0 if y_bias >= 0 else 1), halt        # north / south


def annotate(
    *,
    model_id: str,
    output_path: Path,
    max_samples: int,
    dataset_id: str,
    split: str,
    batch_size: int,
    resume: bool,
) -> None:
    import torch
    from datasets import load_dataset
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

    # Load already-done IDs if resuming
    done_ids: set[str] = set()
    if resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["sample_id"])
                except Exception:
                    pass
        print(f"Resuming: {len(done_ids)} samples already annotated.")

    print(f"Loading dataset {dataset_id} ({split})...")
    dataset = load_dataset(dataset_id, split=split, streaming=True)
    dataset = dataset.shuffle(buffer_size=5000, seed=42)

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

    print(f"Annotating up to {max_samples} samples...")

    for idx, row in enumerate(dataset):
        if annotated >= max_samples:
            break

        sample_id = f"visdrone-{idx:06d}"
        if sample_id in done_ids:
            continue

        # Extract image
        try:
            image = row.get("image")
            if image is None:
                skipped += 1
                continue
            if not hasattr(image, "convert"):
                from PIL import Image as PILImage
                import numpy as np
                image = PILImage.fromarray(image).convert("RGB")
            image = image.convert("RGB")
        except Exception:
            skipped += 1
            continue

        width = int(row.get("width", image.width))
        height = int(row.get("height", image.height))
        objects = row.get("objects", {})
        bboxes = list(objects.get("bbox", [])) if isinstance(objects, dict) else []
        categories = list(objects.get("category", [])) if isinstance(objects, dict) else []

        # Build multimodal prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": ANNOTATION_PROMPT},
                ],
            }
        ]

        try:
            prompt_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=prompt_text,
                images=image,
                return_tensors="pt",
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            generated = output_ids[0][input_len:]
            response_text = processor.tokenizer.decode(generated, skip_special_tokens=True)

        except Exception as e:
            print(f"[{idx}] Generation error: {e}")
            errors += 1
            continue

        action, halt = _parse_action_halt(response_text)

        # Fallback to heuristic if model output is unparseable
        used_heuristic = False
        if action is None or halt is None:
            if bboxes:
                action, halt = _heuristic_action_from_bboxes(bboxes, categories, width, height)
                used_heuristic = True
            else:
                skipped += 1
                continue

        record = {
            "sample_id": sample_id,
            "dataset": dataset_id,
            "width": width,
            "height": height,
            "n_objects": len(bboxes),
            "reasoning_trace": response_text.strip(),
            "action_index": action,
            "halt_step": halt,
            "used_heuristic": used_heuristic,
        }
        out_file.write(json.dumps(record) + "\n")
        out_file.flush()
        annotated += 1

        if annotated % 50 == 0:
            print(f"  [{annotated}/{max_samples}] errors={errors} skipped={skipped}")

    out_file.close()
    print(f"\nDone. Annotated={annotated}, skipped={skipped}, errors={errors}")
    print(f"Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-annotate VisDrone with Gemma E4B reasoning traces.")
    parser.add_argument("--output", type=str, default="data/visdrone_annotated.jsonl")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--model-id", type=str, default="google/gemma-4-e4b-it")
    parser.add_argument("--dataset-id", type=str, default="Voxel51/VisDrone2019-DET")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch-size", type=int, default=1, help="Currently unused, reserved for future batching.")
    parser.add_argument("--resume", action="store_true", help="Skip already-annotated sample IDs.")
    args = parser.parse_args()

    annotate(
        model_id=args.model_id,
        output_path=Path(args.output),
        max_samples=args.max_samples,
        dataset_id=args.dataset_id,
        split=args.split,
        batch_size=args.batch_size,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
