"""
AU-AIR evaluation module — multi-model benchmark on real drone sequences.

Supports three model types:
  - Gemma 4 vanilla      (no fine-tuning, zero-shot)
  - Gemma 4 + LoRA       (fine-tuned on GT telemetry via finetune_gemma_auair.py)
  - TRMReasoner          (student checkpoint from train_trm_student.py)

Each is evaluated against GT telemetry action labels from parse_auair.py.

Also supports test-time LoRA adaptation (NVARC-inspired OOD resilience):
  given K examples from a new environment, rapidly adapt LoRA weights before eval.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    model_name: str
    action_acc: float          # action_0 accuracy vs GT
    chunk_acc: list[float]     # per-step accuracy [a0..aC-1]
    parse_rate: float          # fraction of outputs with parseable action chunk
    halt_acc: float            # halt_0 accuracy vs GT
    ms_per_sample: float       # inference latency
    n_samples: int
    n_heuristic: int           # samples where GT was used (from used_heuristic field)
    extra: dict = field(default_factory=dict)

    def summary(self) -> str:
        chunk_str = "  ".join(f"a{i}={v:.1f}%" for i, v in enumerate(self.chunk_acc))
        return (
            f"[{self.model_name}]\n"
            f"  action0_acc={self.action_acc:.1f}%  halt_acc={self.halt_acc:.1f}%\n"
            f"  chunk: {chunk_str}\n"
            f"  parse_rate={self.parse_rate:.1f}%  latency={self.ms_per_sample:.0f}ms/sample\n"
            f"  n={self.n_samples}"
        )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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


def _load_sequences(jsonl_path: Path, max_samples: int, seed: int = 42) -> list[dict]:
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(records))[:max_samples]
    selected = [records[i] for i in idx]
    logger.info("Loaded %d sequences from %s (total in file: %d)", len(selected), jsonl_path, len(records))
    return selected


def _load_images(image_paths: list[str], T: int, images_path: str | None = None) -> list[Image.Image]:
    if len(image_paths) < T:
        image_paths = [image_paths[0]] * (T - len(image_paths)) + image_paths
    images = []
    base = Path(images_path) if images_path else None
    for p in image_paths[-T:]:
        try:
            resolved = base / Path(p).name if base else Path(p)
            images.append(Image.open(resolved).convert("RGB"))
        except Exception:
            images.append(Image.new("RGB", (640, 480), color=(80, 80, 80)))
    return images


def _user_prompt(T: int, C: int) -> str:
    return (
        f"You are an autonomous drone navigation assistant. "
        f"You are given {T} consecutive aerial frames.\n\n"
        "Analyze object positions and motion across frames, then predict the next "
        f"{C} drone actions.\n\nOutput EXACTLY:\n"
        + "\n".join(f"Action_{i}: <0-7>  Halt_{i}: <1-6>" for i in range(C))
        + "\n\nAction index: 0=north 1=south 2=east 3=west 4=up 5=down 6=yaw_right 7=yaw_left"
    )


# ---------------------------------------------------------------------------
# Gemma evaluation (vanilla or LoRA)
# ---------------------------------------------------------------------------

def evaluate_gemma(
    *,
    sequences: list[dict],
    model_id: str,
    lora_path: str | None = None,
    temporal_window: int = 4,
    action_chunk_size: int = 4,
    device: str = "cuda",
    model_name: str | None = None,
) -> ModelResult:
    """
    Evaluate Gemma 4 on AU-AIR sequences.
    If lora_path is set, loads LoRA adapters (fine-tuned teacher).
    Otherwise evaluates vanilla model (zero-shot baseline).
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

    name = model_name or ("gemma4_lora" if lora_path else "gemma4_vanilla")
    T, C = temporal_window, action_chunk_size

    logger.info("[%s] Loading model %s (lora=%s)...", name, model_id, lora_path or "none")
    t_load = time.perf_counter()

    # Workaround: transformers 5.5.0 bitsandbytes set_submodule bug
    import torch.nn as _nn
    if not hasattr(_nn.Module, "set_submodule"):
        def _ssm(self, target, module):
            atoms = target.split(".")
            mod = self
            for a in atoms[:-1]:
                mod = mod.get_submodule(a)
            setattr(mod, atoms[-1], module)
        _nn.Module.set_submodule = _ssm

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, quantization_config=bnb,
        device_map={"": 0}, dtype=torch.bfloat16,
        attn_implementation="sdpa", trust_remote_code=True,
    )

    if lora_path is not None:
        from peft import PeftModel
        logger.info("[%s] Loading LoRA adapters from %s...", name, lora_path)
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()
    logger.info("[%s] Model ready in %.1fs | evaluating %d sequences", name, time.perf_counter() - t_load, len(sequences))

    correct_actions = [0] * C
    correct_halts = 0
    parseable = 0
    total = 0
    latencies: list[float] = []
    prompt_tmpl = _user_prompt(T, C)

    for seq in sequences:
        images = _load_images(seq.get("image_paths", []), T)
        gt_actions = seq.get("action_indices", [seq.get("action_index", 0)] * C)
        gt_halts = seq.get("halt_steps", [seq.get("halt_step", 3)] * C)
        gt_actions = (list(gt_actions) * C)[:C]
        gt_halts = (list(gt_halts) * C)[:C]

        content = [{"type": "image"} for _ in range(T)]
        content.append({"type": "text", "text": prompt_tmpl})
        messages = [{"role": "user", "content": content}]

        try:
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=prompt, images=images, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

            t0 = time.perf_counter()
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs, max_new_tokens=120, do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )
            latencies.append((time.perf_counter() - t0) * 1000)

            il = inputs["input_ids"].shape[1]
            pred_text = processor.tokenizer.decode(out_ids[0][il:], skip_special_tokens=True)
            pa, ph = _parse_chunk(pred_text, C)

            if pa is not None:
                parseable += 1
                for c in range(C):
                    if pa[c] == gt_actions[c]:
                        correct_actions[c] += 1
                if ph is not None and ph[0] == gt_halts[0]:
                    correct_halts += 1

        except Exception as e:
            logger.error("[%s] sample %s failed: %s", name, seq.get("sample_id"), e)

        total += 1
        if total % 50 == 0:
            logger.debug(
                "[%s] progress %d/%d | parse=%.1f%%  acc=%.1f%%",
                name, total, len(sequences),
                parseable / total * 100,
                correct_actions[0] / total * 100,
            )

    n_heuristic = sum(1 for s in sequences if s.get("used_heuristic", False))
    result = ModelResult(
        model_name=name,
        action_acc=correct_actions[0] / max(total, 1) * 100,
        chunk_acc=[correct_actions[c] / max(total, 1) * 100 for c in range(C)],
        parse_rate=parseable / max(total, 1) * 100,
        halt_acc=correct_halts / max(total, 1) * 100,
        ms_per_sample=float(np.mean(latencies)) if latencies else 0.0,
        n_samples=total,
        n_heuristic=n_heuristic,
    )
    logger.info(
        "[%s] eval done | action0=%.1f%%  parse=%.1f%%  halt=%.1f%%  latency=%.0fms  n=%d",
        name, result.action_acc, result.parse_rate, result.halt_acc, result.ms_per_sample, total,
    )
    return result


# ---------------------------------------------------------------------------
# TRM student evaluation
# ---------------------------------------------------------------------------

def evaluate_trm(
    *,
    sequences: list[dict],
    checkpoint_path: str,
    temporal_window: int = 4,
    action_chunk_size: int = 4,
    device: str = "cuda",
    model_name: str | None = None,
    images_path: str | None = None,
) -> ModelResult:
    """Evaluate TRM student checkpoint on AU-AIR sequences."""
    from .model import TRMReasoner
    from .config import ReasonerConfig
    from .student_training import AuAirStudentDataset

    name = model_name or f"trm_{Path(checkpoint_path).stem}"
    T, C = temporal_window, action_chunk_size
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    logger.info("[%s] Loading checkpoint from %s...", name, checkpoint_path)
    t_load = time.perf_counter()
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    reasoner_cfg = ckpt.get("reasoner_config", {})
    rc = ReasonerConfig(
        hidden_size=reasoner_cfg.get("hidden_size", 96),
        refinement_steps=reasoner_cfg.get("refinement_steps", 6),
        halting_threshold=reasoner_cfg.get("halting_threshold", 0.82),
        action_chunk_size=C,
        temporal_window=T,
    )
    model = TRMReasoner(rc).to(dev)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    logger.info("[%s] Checkpoint loaded in %.1fs | device=%s  evaluating %d sequences",
                name, time.perf_counter() - t_load, dev, len(sequences))

    correct_actions = [0] * C
    correct_halts = 0
    total = 0
    latencies: list[float] = []

    # Reuse AuAirStudentDataset for image→grid conversion
    import tempfile, json as _json
    tmp = Path(tempfile.mktemp(suffix=".jsonl"))
    tmp.write_text("\n".join(_json.dumps(s) for s in sequences))
    ds = AuAirStudentDataset(jsonl_path=tmp, reasoner_config=rc, images_path=images_path)
    tmp.unlink(missing_ok=True)

    for i, seq in enumerate(sequences):
        gt_actions = (list(seq.get("action_indices", [seq.get("action_index", 0)] * C)) * C)[:C]
        gt_halts = (list(seq.get("halt_steps", [seq.get("halt_step", 3)] * C)) * C)[:C]

        item = ds[i]
        grids = item["grids"].unsqueeze(0).to(dev)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(grids)
        latencies.append((time.perf_counter() - t0) * 1000)

        # action_chunk_logits: (1, C, action_dim)
        pred_actions = out.action_chunk_logits[0].argmax(dim=-1).tolist()  # [C]
        halt_pred = out.halted_at_step[0].item()

        for c in range(C):
            if pred_actions[c] == gt_actions[c]:
                correct_actions[c] += 1
        if round(halt_pred) == gt_halts[0]:
            correct_halts += 1

        total += 1
        if total % 50 == 0:
            logger.debug(
                "[%s] progress %d/%d | acc=%.1f%%",
                name, total, len(sequences),
                correct_actions[0] / total * 100,
            )

    n_heuristic = sum(1 for s in sequences if s.get("used_heuristic", False))
    result = ModelResult(
        model_name=name,
        action_acc=correct_actions[0] / max(total, 1) * 100,
        chunk_acc=[correct_actions[c] / max(total, 1) * 100 for c in range(C)],
        parse_rate=100.0,   # TRM always produces valid output
        halt_acc=correct_halts / max(total, 1) * 100,
        ms_per_sample=float(np.mean(latencies)) if latencies else 0.0,
        n_samples=total,
        n_heuristic=n_heuristic,
    )
    logger.info(
        "[%s] eval done | action0=%.1f%%  halt=%.1f%%  latency=%.0fms  n=%d",
        name, result.action_acc, result.halt_acc, result.ms_per_sample, total,
    )
    return result


# ---------------------------------------------------------------------------
# Test-time LoRA adaptation (NVARC-inspired OOD resilience)
# ---------------------------------------------------------------------------

def adapt_and_evaluate_gemma(
    *,
    adapt_sequences: list[dict],   # K examples from new environment
    eval_sequences: list[dict],
    model_id: str,
    base_lora_path: str,
    adapt_lr: float = 5e-5,
    adapt_steps: int = 20,
    temporal_window: int = 4,
    action_chunk_size: int = 4,
    device: str = "cuda",
) -> ModelResult:
    """
    NVARC-inspired test-time LoRA adaptation.
    Given K examples from a new environment (adapt_sequences), briefly
    fine-tunes the LoRA weights, then evaluates on eval_sequences.
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    from peft import PeftModel
    from .teacher_finetuning_auair import AuAirTeacherDataset, _user_prompt as _up

    T, C = temporal_window, action_chunk_size
    name = f"gemma4_lora_tta_{len(adapt_sequences)}shot"

    logger.info("[%s] Loading model for TTA | adapt_shots=%d  adapt_steps=%d  lora=%s",
                name, len(adapt_sequences), adapt_steps, base_lora_path)
    t_load = time.perf_counter()

    import torch.nn as _nn
    if not hasattr(_nn.Module, "set_submodule"):
        def _ssm(self, target, module):
            atoms = target.split(".")
            mod = self
            for a in atoms[:-1]:
                mod = mod.get_submodule(a)
            setattr(mod, atoms[-1], module)
        _nn.Module.set_submodule = _ssm

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    base_model = AutoModelForImageTextToText.from_pretrained(
        model_id, quantization_config=bnb,
        device_map={"": 0}, dtype=torch.bfloat16,
        attn_implementation="sdpa", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, base_lora_path, is_trainable=True)

    # Only train LoRA params
    for name_p, param in model.named_parameters():
        param.requires_grad = "lora" in name_p
        if param.requires_grad and param.dtype != torch.bfloat16:
            param.data = param.data.to(torch.bfloat16)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=adapt_lr
    )

    import tempfile
    tmp = Path(tempfile.mkdtemp())
    adapt_jsonl = tmp / "adapt.jsonl"
    adapt_jsonl.write_text("\n".join(json.dumps(s) for s in adapt_sequences))

    from .teacher_finetuning_auair import AuAirTeacherDataset
    adapt_ds = AuAirTeacherDataset(
        adapt_jsonl, processor, max_length=512,
        temporal_window=T, action_chunk_size=C,
    )
    from torch.utils.data import DataLoader
    adapt_loader = DataLoader(adapt_ds, batch_size=min(4, len(adapt_sequences)), shuffle=True)

    logger.info("[%s] Model loaded in %.1fs — starting adaptation...", name, time.perf_counter() - t_load)
    model.train()
    step = 0
    for batch in adapt_loader:
        if step >= adapt_steps:
            break
        batch = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if "pixel_values" in batch:
            batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
        optimizer.zero_grad()
        loss = model(**batch, return_dict=True).loss
        loss.backward()
        optimizer.step()
        step += 1

    model.eval()
    logger.info("[%s] Adaptation done (%d steps) — evaluating %d sequences...", name, step, len(eval_sequences))

    # Now evaluate using the adapted model
    correct_actions = [0] * C
    correct_halts = 0
    parseable = 0
    total = 0
    latencies: list[float] = []
    prompt_tmpl = _user_prompt(T, C)

    for seq in eval_sequences:
        images = _load_images(seq.get("image_paths", []), T)
        gt_actions = (list(seq.get("action_indices", [seq.get("action_index", 0)] * C)) * C)[:C]
        gt_halts = (list(seq.get("halt_steps", [seq.get("halt_step", 3)] * C)) * C)[:C]

        content = [{"type": "image"} for _ in range(T)]
        content.append({"type": "text", "text": prompt_tmpl})
        messages = [{"role": "user", "content": content}]

        try:
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=prompt, images=images, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

            t0 = time.perf_counter()
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs, max_new_tokens=120, do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )
            latencies.append((time.perf_counter() - t0) * 1000)

            il = inputs["input_ids"].shape[1]
            pred_text = processor.tokenizer.decode(out_ids[0][il:], skip_special_tokens=True)
            pa, ph = _parse_chunk(pred_text, C)
            if pa is not None:
                parseable += 1
                for c in range(C):
                    if pa[c] == gt_actions[c]:
                        correct_actions[c] += 1
                if ph is not None and ph[0] == gt_halts[0]:
                    correct_halts += 1
        except Exception as e:
            logger.error("[%s] sample %s failed: %s", name, seq.get("sample_id"), e)

        total += 1
        if total % 50 == 0:
            logger.debug(
                "[%s] progress %d/%d | parse=%.1f%%  acc=%.1f%%",
                name, total, len(eval_sequences),
                parseable / total * 100,
                correct_actions[0] / total * 100,
            )

    n_heuristic = sum(1 for s in eval_sequences if s.get("used_heuristic", False))
    result = ModelResult(
        model_name=name,
        action_acc=correct_actions[0] / max(total, 1) * 100,
        chunk_acc=[correct_actions[c] / max(total, 1) * 100 for c in range(C)],
        parse_rate=parseable / max(total, 1) * 100,
        halt_acc=correct_halts / max(total, 1) * 100,
        ms_per_sample=float(np.mean(latencies)) if latencies else 0.0,
        n_samples=total,
        n_heuristic=n_heuristic,
        extra={"adapt_steps": step, "adapt_shots": len(adapt_sequences)},
    )
    logger.info(
        "[%s] TTA eval done | action0=%.1f%%  parse=%.1f%%  halt=%.1f%%  latency=%.0fms  n=%d",
        name, result.action_acc, result.parse_rate, result.halt_acc, result.ms_per_sample, total,
    )
    return result
