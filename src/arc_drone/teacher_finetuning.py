"""Multimodal Hybrid SOTA QLoRA fine-tuning logic for the Gemma-4 teacher model with enhanced logging."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable

from .arc_drone_bench import ARCDroneBench
from .config import BenchmarkConfig
from .gemma_layer_sweep import _lazy_import_transformers, serialize_task_for_teacher
from .student_training import action_to_index, halt_probability_to_step, select_device, set_seed

# Official ARC color palette
ARC_COLORS = [
    (0, 0, 0),        # 0: black
    (0, 114, 227),    # 1: blue
    (255, 65, 54),    # 2: red
    (46, 204, 64),    # 3: green
    (255, 220, 0),    # 4: yellow
    (170, 170, 170),  # 5: gray
    (240, 18, 190),   # 6: magenta
    (255, 133, 27),   # 7: orange
    (127, 219, 255),  # 8: azure
    (135, 12, 37),    # 9: maroon
]


def grid_to_image(grid_values: np.ndarray, upscale: int = 10) -> Image.Image:
    h, w = grid_values.shape
    img = Image.new("RGB", (w, h))
    for y in range(h):
        for x in range(w):
            img.putpixel((x, y), ARC_COLORS[int(grid_values[y, x]) % 10])
    if upscale > 1:
        img = img.resize((w * upscale, h * upscale), resample=Image.NEAREST)
    return img


@dataclass(frozen=True, slots=True)
class TeacherFinetuneConfig:
    foundation_model_id: str = "google/gemma-4-e2b-it"
    task_count: int = 25000
    eval_task_count: int = 1000
    batch_size: int = 4
    epochs: int = 2
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    seed: int = 7
    output_dir: str = "artifacts/teacher_lora/gemma_e2b_arc_specialist"
    max_length: int = 1024
    log_sample_every: int = 250


class TeacherHybridDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, tasks: list[Any], processor: Any, max_length: int) -> None:
        self.tasks = tasks
        self.processor = processor
        image_seq_length = getattr(processor, "image_seq_length", 280)
        try:
            self.image_seq_length = int(image_seq_length)
        except (TypeError, ValueError):
            self.image_seq_length = 280
        self.max_length = max_length + self.image_seq_length

    def __len__(self) -> int:
        return len(self.tasks)

    def _supports_chat_template(self) -> bool:
        processor_template = getattr(self.processor, "chat_template", None)
        tokenizer_template = getattr(getattr(self.processor, "tokenizer", None), "chat_template", None)
        return bool(processor_template or tokenizer_template)

    def _image_placeholder_token(self) -> str:
        for owner in (self.processor, getattr(self.processor, "tokenizer", None)):
            if owner is None:
                continue
            token = getattr(owner, "image_token", None)
            if token:
                return str(token)
        return "<|image|>"

    def _build_prompt(self, text_grid: str) -> str:
        task_prompt = f"{text_grid}\nPredict the best drone action family and halting step."
        if self._supports_chat_template():
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": task_prompt},
                    ],
                }
            ]
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Fallback for base/non-chat checkpoints whose processors do not ship a chat template.
        return f"{self._image_placeholder_token()}\n{task_prompt}\n"

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        task = self.tasks[index]
        image = grid_to_image(task.input_grid.values)
        text_grid = serialize_task_for_teacher(task)

        action_idx = action_to_index(task.target_action)
        halt_step = halt_probability_to_step(halt_probability=task.target_action.halt_probability, refinement_steps=6)
        answer = f"Action: {action_idx}, Halt: {halt_step}"

        prompt = self._build_prompt(text_grid)

        full_text = prompt + answer + self.processor.tokenizer.eos_token

        inputs = self.processor(
            text=full_text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )

        payload = {k: v.squeeze(0) for k, v in inputs.items()}

        prompt_tokens = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).input_ids.squeeze(0)
        
        prompt_len = len(prompt_tokens)

        labels = payload["input_ids"].clone()
        labels[:prompt_len] = -100
        labels[payload["attention_mask"] == 0] = -100
        payload["labels"] = labels

        return payload


def _parse_metrics_from_text(text: str) -> tuple[int | None, int | None]:
    try:
        action_match = re.search(r"Action:\s*(\d+)", text)
        halt_match = re.search(r"Halt:\s*(\d+)", text)
        action = int(action_match.group(1)) if action_match else None
        halt = int(halt_match.group(1)) if halt_match else None
        return action, halt
    except Exception:
        return None, None


def finetune_teacher(config: TeacherFinetuneConfig) -> dict[str, Any]:
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device("cuda")

    AutoModelForImageTextToText, AutoProcessor, AutoTokenizer = _lazy_import_transformers()
    from peft import LoraConfig, get_peft_model
    from transformers import BitsAndBytesConfig

    print(f"--- Fine-tuning Initialization ---")
    print(f"Model: {config.foundation_model_id}")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    
    print("\nLoading Multimodal Processor...")
    processor = AutoProcessor.from_pretrained(config.foundation_model_id, trust_remote_code=True)
    
    print("Loading Gemma in 4-bit precision (QLoRA, SDPA attention)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        config.foundation_model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    print("Preparing model for training (BFloat16 mode)...")
    model.gradient_checkpointing_enable()

    for param in model.parameters():
        param.requires_grad = False

    for param in model.parameters():
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)
    
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        if "lora" in name:
            param.data = param.data.to(torch.bfloat16)
    model.print_trainable_parameters()

    print("\nGenerating augmented ARC-Drone tasks for fine-tuning...")
    bench = ARCDroneBench(BenchmarkConfig(task_count=config.task_count, seed=config.seed))
    train_tasks = bench.generate_tasks(augment=True)
    eval_bench = ARCDroneBench(BenchmarkConfig(task_count=config.eval_task_count, seed=config.seed + 1))
    eval_tasks = eval_bench.generate_tasks(augment=True)

    train_dataset = TeacherHybridDataset(train_tasks, processor, config.max_length)
    eval_dataset = TeacherHybridDataset(eval_tasks, processor, config.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    print(f"\n--- Starting Hybrid Multimodal QLoRA Fine-tuning ---")
    best_eval_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_train_loss = 0.0
        correct_actions = 0
        correct_halts = 0
        total_processed = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs} [Train]")
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if "pixel_values" in batch:
                batch["pixel_values"] = batch["pixel_values"].to(dtype=torch.bfloat16)
            
            outputs = model(**batch, return_dict=True)
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()

            with torch.no_grad():
                idx = 0
                logits = outputs.logits[idx]
                target_ids = batch["labels"][idx]
                valid_mask = target_ids != -100
                if valid_mask.any():
                    pred_ids = torch.argmax(logits, dim=-1)[valid_mask]
                    true_ids = target_ids[valid_mask]
                    
                    pred_text = processor.tokenizer.decode(pred_ids, skip_special_tokens=True)
                    true_text = processor.tokenizer.decode(true_ids, skip_special_tokens=True)
                    
                    p_action, p_halt = _parse_metrics_from_text(pred_text)
                    t_action, t_halt = _parse_metrics_from_text(true_text)
                    
                    if p_action == t_action: correct_actions += 1
                    if p_halt == t_halt: correct_halts += 1
                    total_processed += 1

                    if batch_idx % config.log_sample_every == 0:
                        tqdm.write(f"\n[Batch {batch_idx}] Sample Progress:")
                        tqdm.write(f"  Ground Truth: {true_text.strip()}")
                        tqdm.write(f"  Prediction:   {pred_text.strip()}")
                        tqdm.write(f"  Current Loss: {loss.item():.4f}")

            act_acc = (correct_actions / total_processed * 100) if total_processed > 0 else 0
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "act_acc": f"{act_acc:.1f}%"})

        avg_train_loss = total_train_loss / len(train_loader)
        
        model.eval()
        total_eval_loss = 0.0
        eval_correct_actions = 0
        eval_total = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc=f"Epoch {epoch}/{config.epochs} [Eval]"):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                if "pixel_values" in batch:
                    batch["pixel_values"] = batch["pixel_values"].to(dtype=torch.bfloat16)
                
                outputs = model(**batch, return_dict=True)
                total_eval_loss += outputs.loss.item()
                
                pred_ids = torch.argmax(outputs.logits[0], dim=-1)[batch["labels"][0] != -100]
                true_ids = batch["labels"][0][batch["labels"][0] != -100]
                pred_text = processor.tokenizer.decode(pred_ids, skip_special_tokens=True)
                true_text = processor.tokenizer.decode(true_ids, skip_special_tokens=True)
                pa, _ = _parse_metrics_from_text(pred_text)
                ta, _ = _parse_metrics_from_text(true_text)
                if pa == ta: eval_correct_actions += 1
                eval_total += 1
                
        avg_eval_loss = total_eval_loss / len(eval_loader)
        eval_acc = (eval_correct_actions / eval_total * 100) if eval_total > 0 else 0
        
        print(f"\n--- Epoch {epoch} Results ---")
        print(f"Train Loss: {avg_train_loss:.4f} | Eval Loss: {avg_eval_loss:.4f}")
        print(f"Eval Action Accuracy: {eval_acc:.1f}%")
        
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            print(f"New best model! Saving hybrid specialist adapter to {output_dir}")
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)

    summary = {
        "foundation_model_id": config.foundation_model_id,
        "output_dir": output_dir.as_posix(),
        "train_task_count": config.task_count,
        "eval_task_count": config.eval_task_count,
        "epochs": config.epochs,
        "best_eval_loss": best_eval_loss,
        "hybrid_multimodal": True
    }
    
    (output_dir / "finetune_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
