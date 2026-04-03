"""Multimodal Hybrid SOTA QLoRA fine-tuning logic specifically for Gemma-4 MoE (26B-A4B)."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable

from .arc_drone_bench import ARCDroneBench
from .config import BenchmarkConfig
from .gemma_layer_sweep import _lazy_import_transformers
from .student_training import select_device, set_seed
from .teacher_finetuning import TeacherHybridDataset


@dataclass(frozen=True, slots=True)
class TeacherMoEFinetuneConfig:
    """Configuration optimized for the 26B-A4B Mixture-of-Experts model."""
    foundation_model_id: str = "google/gemma-4-26b-a4b"
    task_count: int = 25000
    eval_task_count: int = 1000
    # Batch size is strictly 1 or 2 by default to fit the 25B+ params in 4-bit on 48GB-80GB GPUs
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    epochs: int = 2
    learning_rate: float = 1e-4  # Slightly lower LR for larger model
    lora_r: int = 16
    lora_alpha: int = 32
    seed: int = 7
    output_dir: str = "artifacts/teacher_lora/gemma_26b_moe_arc_specialist"
    max_length: int = 1024
    log_sample_every: int = 100


def _parse_metrics_from_text(text: str) -> tuple[int | None, int | None]:
    """Extracts Action and Halt from Gemma's response text."""
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
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig

    print(f"--- MoE Fine-tuning Initialization ---")
    print(f"Model: {config.foundation_model_id}")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"WARNING: This is a 25B+ parameter model. Ensure you have 40GB+ VRAM.")
    
    print("\nLoading Multimodal Processor...")
    processor = AutoProcessor.from_pretrained(config.foundation_model_id, trust_remote_code=True)
    
    print("Loading Gemma MoE in 4-bit precision (QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,  # Extra memory saving for huge models
    )

    # By setting torch_dtype to bfloat16, we prevent transformers from defaulting
    # to float32 for certain unquantized layers (like LM head), saving gigabytes of VRAM.
    model = AutoModelForImageTextToText.from_pretrained(
        config.foundation_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Enable gradient checkpointing to save VRAM during training of massive models
    model.gradient_checkpointing_enable()

    # We manually cast layers that PEFT normally casts to fp32 (causing OOM on 80GB)
    # into bfloat16 instead, which is native for A100 and halves the memory spike.
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    for name, param in model.named_parameters():
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)
    
    # For MoE models, target modules often include expert routing layers depending on the exact architecture.
    # The standard attention/MLP projections are targeted here.
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    # Ensure LoRA adapters are also bfloat16, not float32
    for name, param in model.named_parameters():
        if "lora" in name:
            param.data = param.data.to(torch.bfloat16)

    model.print_trainable_parameters()

    print("\nGenerating ARC-Drone tasks for fine-tuning...")
    train_tasks = ARCDroneBench(BenchmarkConfig(task_count=config.task_count, seed=config.seed)).generate_tasks()
    eval_tasks = ARCDroneBench(BenchmarkConfig(task_count=config.eval_task_count, seed=config.seed + 1)).generate_tasks()

    # Reuse the Hybrid SOTA dataset from the standard pipeline
    train_dataset = TeacherHybridDataset(train_tasks, processor, config.max_length)
    eval_dataset = TeacherHybridDataset(eval_tasks, processor, config.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    print(f"\n--- Starting Hybrid Multimodal QLoRA Fine-tuning (MoE Optimized) ---")
    best_eval_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_train_loss = 0.0
        correct_actions = 0
        correct_halts = 0
        total_processed = 0
        
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs} [Train]")
        for batch_idx, batch in enumerate(pbar):
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                return_dict=True
            )
            
            # Gradient accumulation logic
            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()
            
            if ((batch_idx + 1) % config.gradient_accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
            
            # Track unscaled loss for logging
            total_train_loss += (loss.item() * config.gradient_accumulation_steps)

            # Live Accuracy Calculation (on the last batch sample)
            with torch.no_grad():
                idx = 0
                logits = outputs.logits[idx]
                target_ids = labels[idx]
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
                        tqdm.write(f"\n[Batch {batch_idx}] MoE Sample Progress:")
                        tqdm.write(f"  Ground Truth: {true_text.strip()}")
                        tqdm.write(f"  Prediction:   {pred_text.strip()}")
                        tqdm.write(f"  Current Loss: {(loss.item() * config.gradient_accumulation_steps):.4f}")

            act_acc = (correct_actions / total_processed * 100) if total_processed > 0 else 0
            pbar.set_postfix({
                "loss": f"{(loss.item() * config.gradient_accumulation_steps):.3f}",
                "act_acc": f"{act_acc:.1f}%",
            })

        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- Evaluation Split ---
        model.eval()
        total_eval_loss = 0.0
        eval_correct_actions = 0
        eval_total = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc=f"Epoch {epoch}/{config.epochs} [Eval]"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
                labels = batch["labels"].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                    return_dict=True
                )
                total_eval_loss += outputs.loss.item()
                
                # Check accuracy
                pred_ids = torch.argmax(outputs.logits[0], dim=-1)[labels[0] != -100]
                true_ids = labels[0][labels[0] != -100]
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
            print(f"New best model! Saving MoE hybrid specialist adapter to {output_dir}")
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)

    summary = {
        "foundation_model_id": config.foundation_model_id,
        "output_dir": output_dir.as_posix(),
        "train_task_count": config.task_count,
        "eval_task_count": config.eval_task_count,
        "epochs": config.epochs,
        "best_eval_loss": best_eval_loss,
        "hybrid_multimodal": True,
        "moe_optimized": True
    }
    
    (output_dir / "finetune_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
