"""QLoRA fine-tuning logic for the Gemma teacher model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
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


@dataclass(frozen=True, slots=True)
class TeacherFinetuneConfig:
    foundation_model_id: str = "google/gemma-4-e2b"
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


class TeacherInstructDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, tasks: list[Any], tokenizer: Any, max_length: int) -> None:
        self.tasks = tasks
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        task = self.tasks[index]
        prompt = serialize_task_for_teacher(task)
        
        action_idx = action_to_index(task.target_action)
        halt_step = halt_probability_to_step(halt_probability=task.target_action.halt_probability, refinement_steps=6)
        
        answer = f"\nAction Index: {action_idx}\nHalt Step: {halt_step}"
        full_text = prompt + answer + self.tokenizer.eos_token

        # Tokenize prompt only to know where to mask labels
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=True).input_ids
        prompt_len = len(prompt_tokens)

        # Tokenize full text
        encoded = self.tokenizer(
            full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        
        # Mask out the prompt so loss is only computed on the answer
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def finetune_teacher(config: TeacherFinetuneConfig) -> dict[str, Any]:
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device("cuda")  # QLoRA requires CUDA

    AutoModelForImageTextToText, AutoProcessor, AutoTokenizer = _lazy_import_transformers()
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig

    print("Loading Tokenizer...")
    try:
        processor = AutoProcessor.from_pretrained(config.foundation_model_id, trust_remote_code=True)
        tokenizer = getattr(processor, "tokenizer", None)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(config.foundation_model_id, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading Gemma in 4-bit precision (QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        config.foundation_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Generating ARC-Drone tasks for fine-tuning...")
    train_tasks = ARCDroneBench(BenchmarkConfig(task_count=config.task_count, seed=config.seed)).generate_tasks()
    eval_tasks = ARCDroneBench(BenchmarkConfig(task_count=config.eval_task_count, seed=config.seed + 1)).generate_tasks()

    train_dataset = TeacherInstructDataset(train_tasks, tokenizer, config.max_length)
    eval_dataset = TeacherInstructDataset(eval_tasks, tokenizer, config.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    print("Starting QLoRA Fine-tuning...")
    best_eval_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs} [Train]"):
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        
        model.eval()
        total_eval_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc=f"Epoch {epoch}/{config.epochs} [Eval]"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True
                )
                total_eval_loss += outputs.loss.item()
                
        avg_eval_loss = total_eval_loss / len(eval_loader)
        
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Eval Loss: {avg_eval_loss:.4f}")
        
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            print(f"New best model! Saving adapter to {output_dir}")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    summary = {
        "foundation_model_id": config.foundation_model_id,
        "output_dir": output_dir.as_posix(),
        "train_task_count": config.task_count,
        "eval_task_count": config.eval_task_count,
        "epochs": config.epochs,
        "best_eval_loss": best_eval_loss
    }
    
    (output_dir / "finetune_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
