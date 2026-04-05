"""
QLoRA fine-tuning of Gemma 4 teacher on AU-AIR GT telemetry labels.

Pipeline:
  auair_sequences.jsonl  (parse_auair.py output, GT actions from IMU)
        ↓
  Gemma 4 E4B + LoRA
        ↓
  artifacts/teacher_lora/gemma_e4b_auair/

No circular annotation — labels come directly from drone telemetry.
The model learns: [T frames] → action chunk (GT) with optional CoT prefix.
"""

from __future__ import annotations

import json
import logging
import platform
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable

from .gemma_layer_sweep import _lazy_import_transformers
from .student_training import select_device, set_seed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
# ... rest of code ...

def _user_prompt(T: int, C: int) -> str:
    return (
        f"You are an autonomous drone navigation assistant. "
        f"You are given {T} consecutive aerial frames.\n\n"
        "Analyze object positions and motion across frames, then predict the next "
        f"{C} drone actions.\n\n"
        "Output EXACTLY:\n"
        + "\n".join(f"Action_{i}: <0-7>  Halt_{i}: <1-6>" for i in range(C))
        + "\n\nAction index: 0=north 1=south 2=east 3=west 4=up 5=down 6=yaw_right 7=yaw_left"
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AuAirTeacherConfig:
    foundation_model_id: str = "google/gemma-4-e4b-it"
    auair_path: str = "data/auair_sequences.jsonl"
    temporal_window: int = 4
    action_chunk_size: int = 4
    eval_ratio: float = 0.1
    task_count: int = 25000        # max train samples (capped to dataset size)
    eval_task_count: int = 1000
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    epochs: int = 3
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    seed: int = 7
    output_dir: str = "artifacts/teacher_lora/gemma_e4b_auair"
    max_length: int = 512          # text tokens (image tokens added on top)
    log_sample_every: int = 100


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AuAirTeacherDataset(Dataset[dict[str, torch.Tensor]]):
    """
    Loads AU-AIR sequences from parse_auair.py output.
    Labels = GT telemetry action_indices + halt_steps (no model annotation).

    Supervised target: structured action chunk only (no CoT).
    Short target → fits more samples in context, cleaner gradient signal.
    """

    def __init__(
        self,
        jsonl_path: Path,
        processor: Any,
        max_length: int,
        temporal_window: int = 4,
        action_chunk_size: int = 4,
    ) -> None:
        self.processor = processor
        self.T = temporal_window
        self.C = action_chunk_size

        image_seq_length = getattr(processor, "image_seq_length", 280)
        try:
            self.image_seq_length = int(image_seq_length)
        except (TypeError, ValueError):
            self.image_seq_length = 280
        self.max_length = max_length + self.T * self.image_seq_length

        self.records: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        logger.info("AuAirTeacherDataset loaded %d sequences from %s", len(self.records), jsonl_path)

    def __len__(self) -> int:
        return len(self.records)

    def _supports_chat_template(self) -> bool:
        return bool(
            getattr(self.processor, "chat_template", None)
            or getattr(getattr(self.processor, "tokenizer", None), "chat_template", None)
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]

        # Load T frames
        image_paths = record.get("image_paths", [])
        if len(image_paths) < self.T:
            image_paths = [image_paths[0]] * (self.T - len(image_paths)) + image_paths
        image_paths = image_paths[-self.T:]

        images = []
        for p in image_paths:
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception:
                images.append(Image.new("RGB", (640, 480), color=(80, 80, 80)))

        # GT labels from telemetry
        action_indices = record.get("action_indices", [record.get("action_index", 0)] * self.C)
        halt_steps = record.get("halt_steps", [record.get("halt_step", 3)] * self.C)
        # Ensure length C
        action_indices = (list(action_indices) * self.C)[: self.C]
        halt_steps = (list(halt_steps) * self.C)[: self.C]

        answer = "\n".join(
            f"Action_{i}: {action_indices[i]}  Halt_{i}: {halt_steps[i]}"
            for i in range(self.C)
        )

        user_prompt = _user_prompt(self.T, self.C)

        if self._supports_chat_template():
            content: list[dict] = [{"type": "image"} for _ in range(self.T)]
            content.append({"type": "text", "text": user_prompt})
            messages = [{"role": "user", "content": content}]
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = "".join(f"<image_{i}>\n" for i in range(self.T)) + user_prompt + "\n"

        full_text = prompt + answer + self.processor.tokenizer.eos_token

        inputs = self.processor(
            text=full_text,
            images=images,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        payload = {k: v.squeeze(0) for k, v in inputs.items()}

        # Mask prompt — supervise answer only
        prompt_tokens = self.processor(
            text=prompt,
            images=images,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).input_ids.squeeze(0)

        labels = payload["input_ids"].clone()
        labels[: len(prompt_tokens)] = -100
        labels[payload["attention_mask"] == 0] = -100
        payload["labels"] = labels

        return payload


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _parse_chunk(text: str, C: int) -> tuple[list[int] | None, list[int] | None]:
    actions, halts = [], []
    for i in range(C):
        am = re.search(rf"Action_{i}:\s*(\d+)", text)
        hm = re.search(rf"Halt_{i}:\s*(\d+)", text)
        if not am or not hm:
            return None, None
        actions.append(int(am.group(1)))
        halts.append(int(hm.group(1)))
    return actions, halts


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def finetune_auair_teacher(config: AuAirTeacherConfig) -> dict[str, Any]:
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device("cuda")

    AutoModelForImageTextToText, AutoProcessor, _ = _lazy_import_transformers()
    from peft import LoraConfig, get_peft_model
    from transformers import BitsAndBytesConfig, get_cosine_schedule_with_warmup

    logger.info("--- AU-AIR Teacher Fine-tuning (GT telemetry labels) ---")
    logger.info("Model:  %s", config.foundation_model_id)
    logger.info("Data:   %s", config.auair_path)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s (%s)", device, gpu_name)

    # ── Load all records and split ──────────────────────────────────────────
    all_lines: list[str] = []
    with open(config.auair_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_lines.append(line)

    rng = np.random.default_rng(config.seed)
    idx = rng.permutation(len(all_lines))
    eval_n = min(config.eval_task_count, max(1, int(len(all_lines) * config.eval_ratio)))
    train_n = min(config.task_count, len(all_lines) - eval_n)
    logger.info("Dataset split — train: %d  eval: %d  (total available: %d)", train_n, eval_n, len(all_lines))

    import tempfile
    tmp = Path(tempfile.mkdtemp())
    (tmp / "train.jsonl").write_text("\n".join(all_lines[i] for i in idx[:train_n]))
    (tmp / "eval.jsonl").write_text("\n".join(all_lines[i] for i in idx[train_n: train_n + eval_n]))

    # ── Model ───────────────────────────────────────────────────────────────
    logger.info("Loading processor for %s...", config.foundation_model_id)
    t_load = time.time()
    processor = AutoProcessor.from_pretrained(config.foundation_model_id, trust_remote_code=True)

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

    logger.info("Loading model in 4-bit QLoRA...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        config.foundation_model_id,
        quantization_config=bnb,
        device_map={"": 0},
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()

    for param in model.parameters():
        param.requires_grad = False
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    for name, param in model.named_parameters():
        if "lora" in name:
            param.data = param.data.to(torch.bfloat16)
    model.print_trainable_parameters()
    logger.info("Model loaded in %.1fs", time.time() - t_load)

    # ── Datasets ────────────────────────────────────────────────────────────
    train_ds = AuAirTeacherDataset(
        tmp / "train.jsonl", processor, config.max_length,
        temporal_window=config.temporal_window,
        action_chunk_size=config.action_chunk_size,
    )
    eval_ds = AuAirTeacherDataset(
        tmp / "eval.jsonl", processor, config.max_length,
        temporal_window=config.temporal_window,
        action_chunk_size=config.action_chunk_size,
    )
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=config.batch_size, shuffle=False)

    # ── Optimizer ───────────────────────────────────────────────────────────
    try:
        import bitsandbytes as bnb_opt
        optimizer = bnb_opt.optim.AdamW8bit(model.parameters(), lr=config.learning_rate)
        logger.info("Optimizer: AdamW 8-bit  lr=%.2e", config.learning_rate)
    except ImportError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        logger.warning("bitsandbytes not available — falling back to standard AdamW  lr=%.2e", config.learning_rate)

    total_steps = (len(train_loader) // config.gradient_accumulation_steps) * config.epochs
    warmup_steps = max(1, int(total_steps * 0.05))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ── Training ────────────────────────────────────────────────────────────
    logger.info("Starting training — %d epochs, %d steps/epoch, warmup=%d", config.epochs, len(train_loader) // config.gradient_accumulation_steps, warmup_steps)
    best_eval_loss = float("inf")
    epoch_history: list[dict] = []
    C = config.action_chunk_size

    for epoch in range(1, config.epochs + 1):
        logger.info("Epoch %d/%d starting...", epoch, config.epochs)
        t_epoch = time.time()
        model.train()
        total_train_loss = 0.0
        correct_actions = 0
        total_processed = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs} [Train]")
        for batch_idx, batch in enumerate(pbar):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if "pixel_values" in batch:
                batch["pixel_values"] = batch["pixel_values"].to(dtype=torch.bfloat16)

            outputs = model(**batch, return_dict=True)
            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_train_loss += loss.item() * config.gradient_accumulation_steps

            # Sample metrics (first item in batch)
            with torch.no_grad():
                logits = outputs.logits[0]
                labels_i = batch["labels"][0]
                sl, sl2 = logits[:-1], labels_i[1:]
                vmask = sl2 != -100
                if vmask.any():
                    pred_ids = torch.argmax(sl, dim=-1)[vmask]
                    true_ids = sl2[vmask]
                    pred_text = processor.tokenizer.decode(pred_ids, skip_special_tokens=True)
                    true_text = processor.tokenizer.decode(true_ids, skip_special_tokens=True)
                    pa, _ = _parse_chunk(pred_text, C)
                    ta, _ = _parse_chunk(true_text, C)
                    if pa is not None and ta is not None and pa[0] == ta[0]:
                        correct_actions += 1
                    total_processed += 1

                    if batch_idx % config.log_sample_every == 0:
                        logger.debug(
                            "[B%d] loss=%.4f  lr=%.2e",
                            batch_idx,
                            loss.item() * config.gradient_accumulation_steps,
                            scheduler.get_last_lr()[0],
                        )
                        logger.debug("  GT:   %s", true_text.strip()[:120])
                        logger.debug("  Pred: %s", pred_text.strip()[:120])

            pbar.set_postfix({"loss": f"{loss.item()*config.gradient_accumulation_steps:.4f}",
                              "act0_acc": f"{correct_actions/max(total_processed,1)*100:.1f}%"})

        avg_train_loss = total_train_loss / len(train_loader)
        train_act_acc = correct_actions / max(total_processed, 1) * 100

        # ── Eval ────────────────────────────────────────────────────────────
        model.eval()
        total_eval_loss = 0.0
        eval_correct = [0] * C
        eval_parseable = 0
        eval_total = 0

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc=f"Epoch {epoch}/{config.epochs} [Eval]"):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                if "pixel_values" in batch:
                    batch["pixel_values"] = batch["pixel_values"].to(dtype=torch.bfloat16)
                outputs = model(**batch, return_dict=True)
                total_eval_loss += outputs.loss.item()

                for i in range(batch["labels"].shape[0]):
                    sl = outputs.logits[i][:-1]
                    sl2 = batch["labels"][i][1:]
                    vmask = sl2 != -100
                    if not vmask.any():
                        continue
                    pred_text = processor.tokenizer.decode(torch.argmax(sl, dim=-1)[vmask], skip_special_tokens=True)
                    true_text = processor.tokenizer.decode(sl2[vmask], skip_special_tokens=True)
                    pa, _ = _parse_chunk(pred_text, C)
                    ta, _ = _parse_chunk(true_text, C)
                    if pa is not None:
                        eval_parseable += 1
                    if pa is not None and ta is not None:
                        for c in range(C):
                            if pa[c] == ta[c]:
                                eval_correct[c] += 1
                    eval_total += 1

        avg_eval_loss = total_eval_loss / max(len(eval_loader), 1)
        eval_parse_rate = eval_parseable / max(eval_total, 1) * 100
        chunk_acc = [eval_correct[c] / max(eval_parseable, 1) * 100 for c in range(C)]

        t_epoch_elapsed = time.time() - t_epoch
        chunk_acc_str = "  ".join(f"a{c}={chunk_acc[c]:.1f}%" for c in range(C))
        logger.info(
            "Epoch %d/%d done in %.1fs | train_loss=%.4f  eval_loss=%.4f  "
            "train_act0=%.1f%%  eval_parse=%.1f%%  eval_chunk=[%s]",
            epoch, config.epochs, t_epoch_elapsed,
            avg_train_loss, avg_eval_loss,
            train_act_acc, eval_parse_rate,
            chunk_acc_str,
        )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "eval_loss": round(avg_eval_loss, 4),
            "train_action0_acc": round(train_act_acc, 2),
            "eval_parse_rate": round(eval_parse_rate, 2),
            "eval_samples": eval_total,
            "eval_chunk_acc": [round(v, 2) for v in chunk_acc],
        }
        epoch_history.append(epoch_metrics)

        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            logger.info("New best eval loss %.4f — saving adapters to %s", best_eval_loss, output_dir)
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)

    summary = {
        "foundation_model_id": config.foundation_model_id,
        "output_dir": output_dir.as_posix(),
        "auair_path": config.auair_path,
        "train_count": train_n,
        "eval_count": eval_n,
        "temporal_window": config.temporal_window,
        "action_chunk_size": config.action_chunk_size,
        "epochs": config.epochs,
        "best_eval_loss": best_eval_loss,
        "epoch_history": epoch_history,
    }
    (output_dir / "finetune_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
