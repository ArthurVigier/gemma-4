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

def _temporal_user_prompt(T: int, C: int) -> str:
    return (
        f"You are an autonomous drone navigation assistant analyzing a sequence of {T} consecutive aerial frames.\n\n"
        "Study the motion of objects across frames carefully before deciding.\n\n"
        "Step 1 — Describe the scene in the FIRST frame: object types, count, spatial location.\n"
        f"Step 2 — Track motion: how do objects move from frame 1 to frame {T}? "
        "Estimate centroid drift (left/right/up/down) and whether the cluster is accelerating.\n"
        f"Step 3 — Predict the next {C} drone actions: given the observed motion trajectory, "
        "which actions should the drone take to follow or intercept the cluster?\n"
        "Step 4 — Assess confidence for each action (1=very confident, 6=very uncertain).\n\n"
        "Output EXACTLY in this format:\n"
        + "\n".join(f"Action_{i}: <0-7>  Halt_{i}: <1-6>" for i in range(C))
        + "\n\nAction index: 0=north 1=south 2=east 3=west 4=up 5=down 6=yaw_right 7=yaw_left"
    )

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
    foundation_model_id: str = "google/gemma-4-e4b-it"
    task_count: int = 25000
    eval_task_count: int = 1000
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    epochs: int = 2
    learning_rate: float = 5e-5
    lora_r: int = 16
    lora_alpha: int = 32
    seed: int = 7
    output_dir: str = "artifacts/teacher_lora/gemma_e4b_arc_specialist"
    max_length: int = 2048
    log_sample_every: int = 250
    real_data_path: str | None = None
    real_data_ratio: float = 0.0
    real_dataset: str | None = None
    real_dataset_split: str = "train"
    # Path to JSONL produced by scripts/annotate_auair.py.
    # When set, the model is fine-tuned on real AU-AIR temporal sequences
    # (T frames → CoT + action chunk) instead of synthetic ARC tasks.
    auair_path: str | None = None
    temporal_window: int = 4
    action_chunk_size: int = 4


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


class TeacherVideoDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset built from AU-AIR annotated sequences (annotate_auair.py output).

    Each record contains T consecutive frames + a CoT reasoning trace that
    explicitly discusses object motion across frames + an action chunk of C steps.

    The model learns to:
      - receive T image tokens (Gemma video encoder handles temporal ordering)
      - generate the CoT reasoning trace as supervised tokens
      - output Action_0..Action_{C-1} and Halt_0..Halt_{C-1}
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
        # Each image token sequence adds to effective max length
        image_seq_length = getattr(processor, "image_seq_length", 280)
        try:
            self.image_seq_length = int(image_seq_length)
        except (TypeError, ValueError):
            self.image_seq_length = 280
        # T images × image_seq_length tokens each
        self.max_length = max_length + self.T * self.image_seq_length

        self.records: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        print(f"  TeacherVideoDataset: {len(self.records)} sequences loaded from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.records)

    def _supports_chat_template(self) -> bool:
        processor_template = getattr(self.processor, "chat_template", None)
        tokenizer_template = getattr(getattr(self.processor, "tokenizer", None), "chat_template", None)
        return bool(processor_template or tokenizer_template)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]

        # Load T consecutive frames
        image_paths = record["image_paths"]
        if len(image_paths) < self.T:
            image_paths = [image_paths[0]] * (self.T - len(image_paths)) + image_paths
        image_paths = image_paths[-self.T:]

        images = []
        for p in image_paths:
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception:
                images.append(Image.new("RGB", (640, 480), color=(80, 80, 80)))

        reasoning_trace = record.get("reasoning_trace", "")
        action_indices = record.get("action_indices", [record.get("action_index", 0)] * self.C)
        halt_steps = record.get("halt_steps", [record.get("halt_step", 3)] * self.C)
        action_indices = (action_indices * self.C)[: self.C]
        halt_steps = (halt_steps * self.C)[: self.C]

        # Answer: structured chunk output matching annotate_auair.py format
        answer_lines = "\n".join(
            f"Action_{i}: {action_indices[i]}  Halt_{i}: {halt_steps[i]}"
            for i in range(self.C)
        )

        user_prompt = _temporal_user_prompt(self.T, self.C)

        # Build prompt with T image tokens
        if self._supports_chat_template():
            content: list[dict] = [{"type": "image"} for _ in range(self.T)]
            content.append({"type": "text", "text": user_prompt})
            messages = [{"role": "user", "content": content}]
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            image_placeholders = "".join(
                f"<image_{i}>\n" for i in range(self.T)
            )
            prompt = f"{image_placeholders}{user_prompt}\n"

        # Full supervised sequence: CoT reasoning + structured answer
        full_text = (
            prompt
            + reasoning_trace.strip()
            + "\n"
            + answer_lines
            + self.processor.tokenizer.eos_token
        )

        inputs = self.processor(
            text=full_text,
            images=images,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        payload = {k: v.squeeze(0) for k, v in inputs.items()}

        # Mask prompt tokens — supervise only CoT + answer
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


def _parse_metrics_from_text(text: str) -> tuple[int | None, int | None]:
    """Parses current-timestep action and halt from model output.

    Supports both legacy format ('Action: X') and chunk format ('Action_0: X').
    """
    try:
        # Chunk format (primary): Action_0: X  Halt_0: Y
        action_match = re.search(r"Action_0:\s*(\d+)", text)
        halt_match = re.search(r"Halt_0:\s*(\d+)", text)
        # Legacy fallback
        if not action_match:
            action_match = re.search(r"Action:\s*(\d+)", text)
        if not halt_match:
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
    from transformers import BitsAndBytesConfig, get_cosine_schedule_with_warmup

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

    if config.auair_path is not None:
        # --- Real-data path: AU-AIR temporal sequences annotated by Gemma video encoder ---
        auair_path = Path(config.auair_path)
        print(f"\nLoading AU-AIR annotated sequences from {auair_path}...")
        all_records_raw: list[str] = []
        with open(auair_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_records_raw.append(line)

        rng = np.random.default_rng(config.seed)
        indices = rng.permutation(len(all_records_raw))
        eval_n = min(config.eval_task_count, max(1, len(all_records_raw) // 10))
        train_n = min(config.task_count, len(all_records_raw) - eval_n)

        import tempfile
        tmp_dir = Path(tempfile.mkdtemp())
        train_jsonl = tmp_dir / "train.jsonl"
        eval_jsonl = tmp_dir / "eval.jsonl"
        train_jsonl.write_text(
            "\n".join(all_records_raw[i] for i in indices[:train_n]), encoding="utf-8"
        )
        eval_jsonl.write_text(
            "\n".join(all_records_raw[i] for i in indices[train_n: train_n + eval_n]),
            encoding="utf-8",
        )
        print(f"  Train sequences: {train_n} | Eval sequences: {eval_n}")

        train_dataset = TeacherVideoDataset(
            train_jsonl, processor, config.max_length,
            temporal_window=config.temporal_window,
            action_chunk_size=config.action_chunk_size,
        )
        eval_dataset = TeacherVideoDataset(
            eval_jsonl, processor, config.max_length,
            temporal_window=config.temporal_window,
            action_chunk_size=config.action_chunk_size,
        )
    else:
        # --- Synthetic ARC proxy path (legacy) ---
        print("\nGenerating augmented ARC-Drone tasks for fine-tuning...")
        train_bench_config = BenchmarkConfig(
            task_count=config.task_count,
            seed=config.seed,
            real_data_path=config.real_data_path,
            real_data_ratio=config.real_data_ratio,
            real_dataset=config.real_dataset,
            real_dataset_split=config.real_dataset_split,
        )
        eval_bench_config = BenchmarkConfig(
            task_count=config.eval_task_count,
            seed=config.seed + 1,
            real_data_path=config.real_data_path,
            real_data_ratio=config.real_data_ratio,
            real_dataset=config.real_dataset,
            real_dataset_split=config.real_dataset_split,
        )
        train_tasks = ARCDroneBench(train_bench_config).generate_tasks(augment=True)
        eval_tasks = ARCDroneBench(eval_bench_config).generate_tasks(augment=True)
        train_dataset = TeacherHybridDataset(train_tasks, processor, config.max_length)
        eval_dataset = TeacherHybridDataset(eval_tasks, processor, config.max_length)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)

    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=config.learning_rate)
        print("Optimizer: AdamW 8-bit (paged)")
    except ImportError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        print("Optimizer: AdamW")

    total_steps = (len(train_loader) // config.gradient_accumulation_steps) * config.epochs
    warmup_steps = max(1, int(total_steps * 0.05))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    print(f"LR scheduler: cosine with {warmup_steps} warmup steps over {total_steps} total steps")

    print(f"\n--- Starting Hybrid Multimodal QLoRA Fine-tuning ---")
    best_eval_loss = float("inf")
    epoch_history: list[dict] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_train_loss = 0.0
        correct_actions = 0
        correct_halts = 0
        total_processed = 0
        
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs} [Train]")
        supervised_count = 0
        for batch_idx, batch in enumerate(pbar):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if "pixel_values" in batch:
                batch["pixel_values"] = batch["pixel_values"].to(dtype=torch.bfloat16)

            outputs = model(**batch, return_dict=True)

            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()

            if ((batch_idx + 1) % config.gradient_accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_train_loss += loss.item() * config.gradient_accumulation_steps

            with torch.no_grad():
                idx = 0
                logits = outputs.logits[idx]
                target_ids = batch["labels"][idx]
                # Causal LM: logits[i] predicts token[i+1] — apply shift before masking
                shifted_logits = logits[:-1]
                shifted_labels = target_ids[1:]
                valid_mask = shifted_labels != -100
                supervised_count = int(valid_mask.sum().item())
                if valid_mask.any():
                    pred_ids = torch.argmax(shifted_logits, dim=-1)[valid_mask]
                    true_ids = shifted_labels[valid_mask]

                    pred_text = processor.tokenizer.decode(pred_ids, skip_special_tokens=True)
                    true_text = processor.tokenizer.decode(true_ids, skip_special_tokens=True)

                    p_action, p_halt = _parse_metrics_from_text(pred_text)
                    t_action, t_halt = _parse_metrics_from_text(true_text)

                    # Guard: None == None would be a false positive when "Action:" was truncated
                    if p_action is not None and p_action == t_action:
                        correct_actions += 1
                    if p_halt is not None and p_halt == t_halt:
                        correct_halts += 1
                    total_processed += 1

                    if batch_idx % config.log_sample_every == 0:
                        lr_now = scheduler.get_last_lr()[0]
                        tqdm.write(f"\n[Batch {batch_idx}] Loss: {(loss.item()*config.gradient_accumulation_steps):.4f}"
                                   f" | sup_tok: {supervised_count} | lr: {lr_now:.2e}")
                        tqdm.write(f"  Ground Truth: {true_text.strip()}")
                        tqdm.write(f"  Prediction:   {pred_text.strip()}")

            act_acc = (correct_actions / total_processed * 100) if total_processed > 0 else 0
            pbar.set_postfix({
                "loss": f"{(loss.item()*config.gradient_accumulation_steps):.4f}",
                "act_acc": f"{act_acc:.1f}%",
                "sup_tok": supervised_count,
            })

        avg_train_loss = total_train_loss / len(train_loader)
        train_act_acc = (correct_actions / total_processed * 100) if total_processed > 0 else 0.0
        train_halt_acc = (correct_halts / total_processed * 100) if total_processed > 0 else 0.0

        # ── Eval loop — all samples per batch, full metric suite ──────────────
        model.eval()
        total_eval_loss = 0.0
        eval_correct_actions = 0
        eval_correct_halts = 0
        eval_parseable = 0      # model output contained valid Action_X / Halt_X tokens
        eval_total = 0
        C = config.action_chunk_size
        chunk_correct = [0] * C  # per-step accuracy across chunk
        chunk_parseable = [0] * C

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc=f"Epoch {epoch}/{config.epochs} [Eval]"):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                if "pixel_values" in batch:
                    batch["pixel_values"] = batch["pixel_values"].to(dtype=torch.bfloat16)

                outputs = model(**batch, return_dict=True)
                total_eval_loss += outputs.loss.item()

                # Iterate over all samples in the batch
                batch_size = batch["labels"].shape[0]
                for i in range(batch_size):
                    logits_e = outputs.logits[i]
                    labels_e = batch["labels"][i]
                    shifted_logits_e = logits_e[:-1]
                    shifted_labels_e = labels_e[1:]
                    vmask_e = shifted_labels_e != -100
                    if not vmask_e.any():
                        continue

                    pred_ids = torch.argmax(shifted_logits_e, dim=-1)[vmask_e]
                    true_ids = shifted_labels_e[vmask_e]
                    pred_text = processor.tokenizer.decode(pred_ids, skip_special_tokens=True)
                    true_text = processor.tokenizer.decode(true_ids, skip_special_tokens=True)

                    pa0, ph0 = _parse_metrics_from_text(pred_text)
                    ta0, th0 = _parse_metrics_from_text(true_text)

                    if pa0 is not None:
                        eval_parseable += 1
                    if pa0 is not None and pa0 == ta0:
                        eval_correct_actions += 1
                    if ph0 is not None and ph0 == th0:
                        eval_correct_halts += 1

                    # Per-chunk-step accuracy (Action_0..Action_{C-1})
                    for c in range(C):
                        pa_c = None
                        ta_c = None
                        try:
                            import re as _re
                            pm = _re.search(rf"Action_{c}:\s*(\d+)", pred_text)
                            tm = _re.search(rf"Action_{c}:\s*(\d+)", true_text)
                            if pm:
                                pa_c = int(pm.group(1))
                            if tm:
                                ta_c = int(tm.group(1))
                        except Exception:
                            pass
                        if pa_c is not None:
                            chunk_parseable[c] += 1
                        if pa_c is not None and pa_c == ta_c:
                            chunk_correct[c] += 1

                    eval_total += 1

        avg_eval_loss = total_eval_loss / max(len(eval_loader), 1)
        eval_act_acc = (eval_correct_actions / eval_total * 100) if eval_total > 0 else 0.0
        eval_halt_acc = (eval_correct_halts / eval_total * 100) if eval_total > 0 else 0.0
        eval_parse_rate = (eval_parseable / eval_total * 100) if eval_total > 0 else 0.0
        chunk_acc_pct = [
            (chunk_correct[c] / max(chunk_parseable[c], 1) * 100) for c in range(C)
        ]

        print(f"\n--- Epoch {epoch} Results ---")
        print(f"  Train Loss:       {avg_train_loss:.4f}  |  Eval Loss:       {avg_eval_loss:.4f}")
        print(f"  Train Act Acc:    {train_act_acc:.1f}%   |  Eval Act Acc:    {eval_act_acc:.1f}%")
        print(f"  Train Halt Acc:   {train_halt_acc:.1f}%   |  Eval Halt Acc:   {eval_halt_acc:.1f}%")
        print(f"  Eval Parse Rate:  {eval_parse_rate:.1f}%  ({eval_parseable}/{eval_total} samples parseable)")
        print(f"  Chunk accuracy per step: " + "  ".join(
            f"a{c}={chunk_acc_pct[c]:.1f}%" for c in range(C)
        ))

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "eval_loss": avg_eval_loss,
            "train_action_acc": round(train_act_acc, 2),
            "train_halt_acc": round(train_halt_acc, 2),
            "eval_action_acc": round(eval_act_acc, 2),
            "eval_halt_acc": round(eval_halt_acc, 2),
            "eval_parse_rate": round(eval_parse_rate, 2),
            "eval_samples": eval_total,
            "chunk_acc_per_step": [round(v, 2) for v in chunk_acc_pct],
        }
        epoch_history.append(epoch_metrics)

        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            print(f"  → New best! Saving adapter to {output_dir}")
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)

    summary = {
        "foundation_model_id": config.foundation_model_id,
        "output_dir": output_dir.as_posix(),
        "train_task_count": config.task_count,
        "eval_task_count": config.eval_task_count,
        "epochs": config.epochs,
        "temporal_window": config.temporal_window,
        "action_chunk_size": config.action_chunk_size,
        "best_eval_loss": best_eval_loss,
        "epoch_history": epoch_history,
    }

    (output_dir / "finetune_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
