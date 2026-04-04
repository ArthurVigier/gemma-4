"""Multimodal Hybrid SOTA QLoRA fine-tuning logic specifically for Gemma-4 MoE (26B-A4B)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

# Optimization: avoid memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable

from .arc_drone_bench import ARCDroneBench
from .config import BenchmarkConfig
from .gemma_layer_sweep import _lazy_import_transformers, serialize_task_for_teacher
from .student_training import action_to_index, halt_probability_to_step, select_device, set_seed
from .teacher_finetuning import grid_to_image


@dataclass(frozen=True, slots=True)
class TeacherMoEFinetuneConfig:
    """Configuration optimized for the 26B-A4B Mixture-of-Experts model."""
    foundation_model_id: str = "google/gemma-4-26B-A4B-it"
    task_count: int = 25000
    eval_task_count: int = 1000
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    epochs: int = 2
    learning_rate: float = 5e-5
    lora_r: int = 8
    lora_alpha: int = 16
    seed: int = 7
    output_dir: str = "artifacts/teacher_lora/gemma_26b_moe_arc_specialist"
    max_length: int = 1024
    log_sample_every: int = 50


class TeacherMoEHybridDataset(torch.utils.data.Dataset[dict[str, torch.Tensor]]):
    """Dataset providing ALL keys required by the Gemma-4 multimodal processor."""

    def __init__(self, tasks: list[Any], processor: Any, max_length: int) -> None:
        self.tasks = tasks
        self.processor = processor
        image_seq_length = getattr(processor, "image_seq_length", 280)
        try:
            self.image_seq_length = int(image_seq_length)
        except (TypeError, ValueError):
            self.image_seq_length = 280
        # Reserve space for the processor's configured image token expansion.
        self.max_length = max_length + self.image_seq_length

    def __len__(self) -> int:
        return len(self.tasks)

    def _format_isaac_world_state(self, isaac_scene: dict[str, Any]) -> str:
        """Converts the Isaac Sim scene descriptor into compact text for the LLM."""
        entities = isaac_scene.get("entities", [])
        if not entities:
            return "Isaac world state unavailable."

        lines = ["Isaac Sim World State (NED Coordinates):"]
        for entity in entities[:30]:
            pos = entity.get("position", (0.0, 0.0, 0.0))
            semantics = entity.get("semantics", {})
            entity_id = semantics.get("id", "unknown")
            material = entity.get("material", "unknown")
            prim_type = entity.get("prim_type", "object")
            color_id = entity.get("color_id", "?")
            lines.append(
                f"- {entity_id}: color={color_id} material={material} shape={prim_type} "
                f"at N:{float(pos[0]):.2f}m E:{float(pos[1]):.2f}m D:{float(pos[2]):.2f}m"
            )

        if len(entities) > 30:
            lines.append(f"... {len(entities) - 30} additional entities omitted")

        randomization = isaac_scene.get("replicator_randomization")
        if isinstance(randomization, dict) and randomization:
            lines.append(f"Replicator randomization: {json.dumps(randomization, sort_keys=True)}")

        return "\n".join(lines)

    def __getitem__(self, index: int) -> dict[str, Any]:
        task = self.tasks[index]
        image = grid_to_image(task.input_grid.values)
        text_grid = serialize_task_for_teacher(task)
        world_state = self._format_isaac_world_state(task.metadata.get("isaac_scene", {}))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": (
                            f"--- PHYSICAL SENSORS ---\n{world_state}\n\n"
                            f"--- LOGICAL GRID ---\n{text_grid}\n\n"
                            "Predict the best drone action family and halting step."
                        ),
                    },
                ],
            }
        ]

        action_idx = action_to_index(task.target_action)
        halt_step = halt_probability_to_step(halt_probability=task.target_action.halt_probability, refinement_steps=6)
        reasoning_trace = str(task.metadata.get("reasoning_trace", "Reasoning: Analyze the pattern.")).strip()
        answer = f"{reasoning_trace}\nAction: {action_idx}, Halt: {halt_step}"

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        full_text = prompt + answer + self.processor.tokenizer.eos_token

        try:
            inputs = self.processor(
                text=full_text,
                images=image,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True
            )
        except TypeError:
            inputs = self.processor(
                text=full_text,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True
            )
            raise RuntimeError("AutoProcessor returned a basic tokenizer instead of a multimodal processor. Ensure transformers is up to date.")

        payload = {k: v.squeeze(0) for k, v in inputs.items()}

        try:
            prompt_inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )
            prompt_len = prompt_inputs.input_ids.shape[1]
        except TypeError:
            prompt_inputs = self.processor(text=prompt, return_tensors="pt")
            prompt_len = prompt_inputs.input_ids.shape[1]

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


def finetune_teacher_moe(config: TeacherMoEFinetuneConfig) -> dict[str, Any]:
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device("cuda")

    AutoModelForImageTextToText, AutoProcessor, AutoTokenizer = _lazy_import_transformers()
    from peft import LoraConfig, get_peft_model
    from transformers import BitsAndBytesConfig

    print(f"--- MoE Ultra-Surgical Fine-tuning ---")
    print(f"Model: {config.foundation_model_id}")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    
    print("\nLoading Multimodal Processor...")
    processor = AutoProcessor.from_pretrained(config.foundation_model_id, trust_remote_code=True)
    
    print("Loading Gemma MoE in 4-bit with SDPA attention...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        config.foundation_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
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

    print("\nGenerating ARC-Drone tasks for fine-tuning...")
    train_tasks = ARCDroneBench(BenchmarkConfig(task_count=config.task_count, seed=config.seed)).generate_tasks(augment=True)
    eval_tasks = ARCDroneBench(BenchmarkConfig(task_count=config.eval_task_count, seed=config.seed + 1)).generate_tasks(augment=True)

    train_dataset = TeacherMoEHybridDataset(train_tasks, processor, config.max_length)
    eval_dataset = TeacherMoEHybridDataset(eval_tasks, processor, config.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    print(f"\n--- Starting Hybrid Multimodal Fine-tuning (Surgical MoE) ---")
    best_eval_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
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
            
            if ((batch_idx + 1) % config.gradient_accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
            
            total_train_loss += (loss.item() * config.gradient_accumulation_steps)

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
                    p_action, _ = _parse_metrics_from_text(pred_text)
                    t_action, _ = _parse_metrics_from_text(true_text)
                    if p_action == t_action: correct_actions += 1
                    total_processed += 1

                    if batch_idx % config.log_sample_every == 0:
                        tqdm.write(f"\n[Batch {batch_idx}] Loss: {(loss.item() * config.gradient_accumulation_steps):.4f} | Acc: {(correct_actions/total_processed*100):.1f}%")

            act_acc = (correct_actions / total_processed * 100) if total_processed > 0 else 0
            pbar.set_postfix({"loss": f"{(loss.item()*config.gradient_accumulation_steps):.2f}", "acc": f"{act_acc:.1f}%"})

        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- Evaluation ---
        model.eval()
        total_eval_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc=f"Eval"):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                if "pixel_values" in batch:
                    batch["pixel_values"] = batch["pixel_values"].to(dtype=torch.bfloat16)
                outputs = model(**batch, return_dict=True)
                total_eval_loss += outputs.loss.item()
                
        avg_eval_loss = total_eval_loss / len(eval_loader)
        print(f"Epoch {epoch} | Train: {avg_train_loss:.4f} | Eval: {avg_eval_loss:.4f}")
        
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            print(f"Saving model to {output_dir}")
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)

    return {"best_eval_loss": best_eval_loss, "output_dir": output_dir.as_posix()}
