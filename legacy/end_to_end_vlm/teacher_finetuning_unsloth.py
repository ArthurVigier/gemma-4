"""
High-speed Unsloth-based fine-tuning for Gemma 4 Teacher.
Designed to run in a dedicated virtual environment with Torch 2.5.1.
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
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
# Prompt & Config
# ---------------------------------------------------------------------------

def _user_prompt(T: int, C: int) -> str:
    return (
        f"You are an autonomous drone navigation assistant. "
        f"Analyze this mosaic containing {T} consecutive aerial frames. "
        f"Predict the next {C} actions.\n\nOutput EXACTLY in this format:\n"
        + "\n".join(f"Action_{i}: <0-7>  Halt_{i}: <1-6>" for i in range(C))
    )

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
    output_dir: str = "artifacts/teacher_lora/gemma_e4b_auair_unsloth"
    max_length: int = 1024

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AuAirTeacherDataset(Dataset[dict[str, Any]]):
    def __init__(self, jsonl_path: Path, processor: Any, max_length: int, temporal_window: int, action_chunk_size: int, images_path: str | Path | None = None) -> None:
        self.processor = processor
        self.T = temporal_window
        self.C = action_chunk_size
        self.images_path = Path(images_path) if images_path else None
        self.max_length = max_length
        self.records = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]

    def __len__(self) -> int: return len(self.records)

    @staticmethod
    def _make_mosaic(images: list[Image.Image]) -> Image.Image:
        T = len(images)
        res_size = 224
        imgs = [img.resize((res_size, res_size), Image.Resampling.LANCZOS) for img in images]
        if T == 1: return imgs[0]
        mosaic = Image.new("RGB", (res_size*2, res_size*2))
        for i in range(4):
            mosaic.paste(imgs[min(i, len(imgs)-1)], ((i%2)*res_size, (i//2)*res_size))
        return mosaic

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.records[index]
        image_paths = rec.get("image_paths", [])
        if not image_paths:
            image_paths = [""] * self.T
        elif len(image_paths) < self.T:
            image_paths = [image_paths[0]] * (self.T - len(image_paths)) + image_paths
        image_paths = image_paths[-self.T:]

        images = []
        for p in image_paths:
            res = self.images_path / Path(p).name if self.images_path and p else Path(p)
            try:
                img = Image.open(res).convert("RGB")
                images.append(img)
            except Exception:
                images.append(Image.new("RGB", (224, 224), color=(80, 80, 80)))

        mosaic = self._make_mosaic(images)

        ai_raw = rec.get("action_indices", [rec.get("action_index", 0)] * self.C)
        hs_raw = rec.get("halt_steps", [rec.get("halt_step", 3)] * self.C)
        ai = (list(ai_raw) * self.C)[:self.C]
        hs = (list(hs_raw) * self.C)[:self.C]
        
        answer = "\n".join(f"Action_{i}: {ai[i]}  Halt_{i}: {hs[i]}" for i in range(self.C))
        
        # Format for UnslothVisionDataCollator
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": _user_prompt(self.T, self.C)}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        ]
        
        return {
            "messages": messages,
            "images": [mosaic],
        }

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config: AuAirTeacherConfig):
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    model, processor = FastVisionModel.from_pretrained(model_name=config.foundation_model_id, load_in_4bit=True)
    model = FastVisionModel.get_peft_model(model, r=config.lora_r, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    # Load & Split
    lines = Path(config.auair_path).read_text().splitlines()
    rng = np.random.default_rng(config.seed)
    idx = rng.permutation(len(lines))
    eval_n = int(len(lines) * config.eval_ratio)
    
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        (td/"train.jsonl").write_text("\n".join(lines[i] for i in idx[eval_n:eval_n+config.task_count]))
        (td/"eval.jsonl").write_text("\n".join(lines[i] for i in idx[:eval_n]))
        
        train_ds = AuAirTeacherDataset(td/"train.jsonl", processor, config.max_length, config.temporal_window, config.action_chunk_size, config.auair_images_path)
        
        trainer = SFTTrainer(
            model=model, tokenizer=processor, train_dataset=train_ds,
            data_collator=UnslothVisionDataCollator(model, processor),
            args=SFTConfig(
                output_dir=config.output_dir, fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
                per_device_train_batch_size=config.batch_size, gradient_accumulation_steps=config.gradient_accumulation_steps,
                learning_rate=config.learning_rate, num_train_epochs=config.epochs, logging_steps=10,
                save_strategy="epoch", lr_scheduler_type="cosine", weight_decay=0.01,
            )
        )
        trainer.train()
        model.save_pretrained(config.output_dir)
        processor.save_pretrained(config.output_dir)
