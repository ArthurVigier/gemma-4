"""
Fine-tuning of Gemma 4 teacher on AU-AIR using Unsloth for 2x speedup.

Unsloth FastVisionModel preserves the vision encoder's precision while
quantizing the LLM to 4-bit, avoiding the 'blind model' bug.
"""

from __future__ import annotations

import json
import logging
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

from .student_training import select_device, set_seed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def _user_prompt(T: int, C: int) -> str:
    return (
        f"You are an autonomous drone navigation assistant. "
        f"Analyze this mosaic containing {T} consecutive aerial frames. "
        f"Predict the next {C} actions.\n\n"
        "Output EXACTLY in this format:\n"
        + "\n".join(f"Action_{i}: <0-7>  Halt_{i}: <1-6>" for i in range(C))
    )

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AuAirTeacherConfig:
    foundation_model_id: str = "google/gemma-4-e4b-it"
    auair_path: str = "data/auair_sequences.jsonl"
    auair_images_path: str | None = None
    temporal_window: int = 4
    action_chunk_size: int = 4
    eval_ratio: float = 0.1
    task_count: int = 25000
    eval_task_count: int = 1000
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    epochs: int = 3
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    seed: int = 7
    output_dir: str = "artifacts/teacher_lora/gemma_e4b_auair"
    max_length: int = 1024
    log_sample_every: int = 100

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AuAirTeacherDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        jsonl_path: Path,
        processor: Any,
        max_length: int,
        temporal_window: int = 4,
        action_chunk_size: int = 4,
        images_path: str | Path | None = None,
    ) -> None:
        self.processor = processor
        self.T = temporal_window
        self.C = action_chunk_size
        self.images_path = Path(images_path) if images_path else None
        self.max_length = max_length

        self.records: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        logger.info("AuAirTeacherDataset loaded %d sequences from %s", len(self.records), jsonl_path)

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _make_mosaic(images: list[Image.Image]) -> Image.Image:
        T = len(images)
        res = 224
        imgs = [img.resize((res, res), Image.Resampling.LANCZOS) for img in images]
        if T == 1: return imgs[0]
        mosaic = Image.new("RGB", (res * 2, res * 2))
        for i in range(min(4, T)):
            mosaic.paste(imgs[i], ((i % 2) * res, (i // 2) * res))
        if T < 4:
            for i in range(T, 4):
                mosaic.paste(imgs[-1], ((i % 2) * res, (i // 2) * res))
        return mosaic

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        image_paths = record.get("image_paths", [])
        if len(image_paths) < self.T:
            image_paths = [image_paths[0]] * (self.T - len(image_paths)) + image_paths
        image_paths = image_paths[-self.T:]

        images = []
        for p in image_paths:
            resolved_path = self.images_path / Path(p).name if self.images_path else Path(p)
            if not resolved_path.exists():
                raise FileNotFoundError(f"Image not found: {resolved_path}")
            images.append(Image.open(resolved_path).convert("RGB"))

        mosaic = self._make_mosaic(images)

        action_indices_raw = record.get("action_indices", [record.get("action_index", 0)] * self.C)
        halt_steps_raw = record.get("halt_steps", [record.get("halt_step", 3)] * self.C)
        action_indices = (list(action_indices_raw) * self.C)[: self.C]
        halt_steps = (list(halt_steps_raw) * self.C)[: self.C]

        answer = "\n".join(f"Action_{i}: {action_indices[i]}  Halt_{i}: {halt_steps[i]}" for i in range(self.C))
        user_prompt = _user_prompt(self.T, self.C)

        # Standard conversation format for Unsloth
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        ]
        
        prompt = self.processor.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
        full_text = self.processor.apply_chat_template(messages, tokenize=False)

        inputs = self.processor(text=full_text, images=mosaic, return_tensors="pt", padding="max_length", max_length=self.max_length, truncation=True)
        payload = {k: v.squeeze(0) for k, v in inputs.items()}

        prompt_inputs = self.processor(text=prompt, images=mosaic, return_tensors="pt", truncation=True, max_length=self.max_length)
        prompt_len = prompt_inputs.input_ids.shape[1]

        labels = payload["input_ids"].clone()
        labels[:prompt_len] = -100
        labels[payload["attention_mask"] == 0] = -100
        payload["labels"] = labels

        return payload

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _parse_chunk(text: str, C: int) -> tuple[list[int] | None, list[int] | None]:
    actions, halts = [], []
    for i in range(C):
        am = re.search(rf"Action[_\s:]*{i}\s*[:\s-]*\s*(\d+)", text, re.IGNORECASE)
        hm = re.search(rf"Halt[_\s:]*{i}\s*[:\s-]*\s*(\d+)", text, re.IGNORECASE)
        if not am or not hm: return None, None
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

    import torch
    # Workaround for PyTorch 2.6.0 + Unsloth bug where torch._inductor.config is missing
    try:
        import torch._dynamo
        import torch._inductor.config
    except ImportError:
        pass
        
    # Workaround for torchao / transformers crash on PyTorch 2.6 where sub-byte dtypes are missing
    class _FakeDtype:
        pass
    for i in range(1, 8):
        if not hasattr(torch, f"int{i}"):
            setattr(torch, f"int{i}", _FakeDtype())
            logger.debug("Applied torch.int%d monkey-patch for torchao compatibility.", i)

    from unsloth import FastVisionModel

    logger.info("--- AU-AIR Teacher Fine-tuning (Unsloth Speedup) ---")
    
    # 1. Load Model
    model, processor = FastVisionModel.from_pretrained(
        model_name = config.foundation_model_id,
        load_in_4bit = True,
        use_gradient_checkpointing = "unsloth",
    )

    # 2. Add LoRA
    model = FastVisionModel.get_peft_model(
        model,
        r = config.lora_r,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = config.lora_alpha,
        lora_dropout = 0,
        bias = "none",
        random_state = config.seed,
        finetune_vision_layers = False,
    )

    # ── Split Data ──────────────────────────────────────────────────────────
    all_lines = []
    with open(config.auair_path, encoding="utf-8") as f:
        for line in f:
            if line.strip(): all_lines.append(line)

    rng = np.random.default_rng(config.seed)
    idx = rng.permutation(len(all_lines))
    eval_n = min(config.eval_task_count, max(1, int(len(all_lines) * config.eval_ratio)))
    train_n = min(config.task_count, len(all_lines) - eval_n)
    
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    (tmp / "train.jsonl").write_text("\n".join(all_lines[i] for i in idx[:train_n]))
    (tmp / "eval.jsonl").write_text("\n".join(all_lines[i] for i in idx[train_n: train_n + eval_n]))

    train_ds = AuAirTeacherDataset(tmp / "train.jsonl", processor, config.max_length, config.temporal_window, config.action_chunk_size, config.auair_images_path)
    eval_ds = AuAirTeacherDataset(tmp / "eval.jsonl", processor, config.max_length, config.temporal_window, config.action_chunk_size, config.auair_images_path)
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=config.batch_size, shuffle=False)

    # 3. Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    from transformers import get_cosine_schedule_with_warmup
    total_steps = (len(train_loader) // config.gradient_accumulation_steps) * config.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps)

    logger.info("Starting training...")
    best_eval_loss = float("inf")
    epoch_history = []
    C = config.action_chunk_size

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_train_loss = 0.0
        correct_actions = 0
        total_processed = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}")
        for batch_idx, batch in enumerate(pbar):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_train_loss += outputs.loss.item()
            
            # Sample accuracy on Action_0
            with torch.no_grad():
                logits = outputs.logits[0]
                labels = batch["labels"][0]
                vmask = labels != -100
                if vmask.any():
                    pred_text = processor.tokenizer.decode(torch.argmax(logits[:-1], dim=-1)[vmask[1:]], skip_special_tokens=True)
                    true_text = processor.tokenizer.decode(labels[vmask], skip_special_tokens=True)
                    pa, _ = _parse_chunk(pred_text, C)
                    ta, _ = _parse_chunk(true_text, C)
                    if pa and ta and pa[0] == ta[0]: correct_actions += 1
                    total_processed += 1

            pbar.set_postfix({"loss": f"{outputs.loss.item():.4f}", "acc": f"{correct_actions/max(total_processed,1)*100:.1f}%"})

        # Eval
        model.eval()
        total_eval_loss = 0.0
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = model(**batch)
                total_eval_loss += outputs.loss.item()
        
        avg_eval_loss = total_eval_loss / len(eval_loader)
        logger.info("Epoch %d: eval_loss=%.4f", epoch, avg_eval_loss)
        
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)

    return {"best_eval_loss": best_eval_loss, "output_dir": str(output_dir)}
