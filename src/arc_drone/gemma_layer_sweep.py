"""Lightweight Gemma hidden-layer sweep for teacher signal selection."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .arc_drone_bench import ARCDroneBench
from .config import BenchmarkConfig
from .student_training import ACTION_VOCABULARY, halt_probability_to_step


def _lazy_import_transformers() -> tuple[Any, Any, Any]:
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
    except Exception as exc:  # pragma: no cover - exercised only in training envs
        raise RuntimeError(
            "transformers is required for Gemma layer sweep. Install with `python3 -m pip install -e '.[training]'`."
        ) from exc
    return AutoModelForImageTextToText, AutoProcessor, AutoTokenizer


def _load_teacher_components(*, foundation_model_id: str, device: torch.device) -> tuple[Any, Any]:
    AutoModelForImageTextToText, AutoProcessor, AutoTokenizer = _lazy_import_transformers()

    processor = None
    tokenizer = None
    try:
        processor = AutoProcessor.from_pretrained(foundation_model_id, trust_remote_code=True)
        tokenizer = getattr(processor, "tokenizer", None)
    except Exception:
        processor = None

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(foundation_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    teacher_model = AutoModelForImageTextToText.from_pretrained(
        foundation_model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    teacher_model.eval()

    return tokenizer, teacher_model


def _extract_hidden_states(outputs: Any) -> tuple[Any, ...]:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is not None:
        return hidden_states

    nested = getattr(outputs, "language_model_outputs", None)
    hidden_states = getattr(nested, "hidden_states", None)
    if hidden_states is not None:
        return hidden_states

    raise RuntimeError("Unable to extract hidden states from Gemma language backbone outputs.")


@dataclass(frozen=True, slots=True)
class LayerSweepConfig:
    foundation_model_id: str = "google/gemma-4-e2b"
    task_count: int = 1024
    eval_task_count: int = 256
    probe_epochs: int = 5
    probe_batch_size: int = 32
    probe_learning_rate: float = 1e-3
    seed: int = 7
    device: str = "cuda"
    output_dir: str = "artifacts/layer_sweeps/gemma_layers"
    layers: tuple[int, ...] = ()
    layer_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 0.9)
    max_length: int = 768


@dataclass(frozen=True, slots=True)
class LayerSweepResult:
    layer_index: int
    train_action_accuracy: float
    eval_action_accuracy: float
    train_halt_step_mae: float
    eval_halt_step_mae: float


@dataclass(frozen=True, slots=True)
class LayerSweepSummary:
    foundation_model_id: str
    output_dir: str
    hidden_layer_count: int
    selected_layers: list[int]
    best_layer_index: int
    best_eval_action_accuracy: float
    results: list[LayerSweepResult]


class TeacherFeatureDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        *,
        features_by_layer: dict[int, torch.Tensor],
        action_indices: torch.Tensor,
        halt_steps: torch.Tensor,
    ) -> None:
        self.features_by_layer = features_by_layer
        self.action_indices = action_indices
        self.halt_steps = halt_steps

    def __len__(self) -> int:
        return int(self.action_indices.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item: dict[str, torch.Tensor] = {
            "action_index": self.action_indices[index],
            "halt_step": self.halt_steps[index],
        }
        for layer_index, features in self.features_by_layer.items():
            item[f"layer_{layer_index}"] = features[index]
        return item


class LayerProbe(nn.Module):
    def __init__(self, hidden_size: int, refinement_steps: int) -> None:
        super().__init__()
        self.action_head = nn.Linear(hidden_size, len(ACTION_VOCABULARY))
        self.halt_head = nn.Linear(hidden_size, refinement_steps)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.action_head(features), self.halt_head(features)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def serialize_task_for_teacher(task) -> str:
    grid_lines = ["".join(str(int(value)) for value in row) for row in task.input_grid.values]
    return "\n".join(
        [
            "You are a teacher model for ARC-Drone-Bench.",
            f"Family: {task.family}",
            "Input grid:",
            *grid_lines,
            "Predict the best drone action family and halting step.",
        ]
    )


def choose_layer_indices(*, total_layers: int, explicit_layers: tuple[int, ...], fractions: tuple[float, ...]) -> list[int]:
    if explicit_layers:
        return sorted({max(0, min(total_layers - 1, int(layer))) for layer in explicit_layers})

    candidates = {
        max(0, min(total_layers - 1, int(round((total_layers - 1) * fraction))))
        for fraction in fractions
    }
    return sorted(candidates)


def build_teacher_features(
    *,
    tasks: list[Any],
    foundation_model_id: str,
    layers: list[int],
    max_length: int,
    device: torch.device,
) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor, int]:
    tokenizer, teacher_model = _load_teacher_components(
        foundation_model_id=foundation_model_id,
        device=device,
    )

    prompts = [serialize_task_for_teacher(task) for task in tasks]
    action_indices = torch.tensor(
        [action_to_index(task.target_action) for task in tasks],
        dtype=torch.long,
    )
    halt_steps = torch.tensor(
        [
            halt_probability_to_step(halt_probability=task.target_action.halt_probability, refinement_steps=6) - 1
            for task in tasks
        ],
        dtype=torch.long,
    )

    feature_batches: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    batch_size = 8
    hidden_layer_count = -1
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            encoded = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            outputs = teacher_model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = _extract_hidden_states(outputs)
            hidden_layer_count = len(hidden_states) - 1
            attention_mask = encoded["attention_mask"]
            token_positions = attention_mask.sum(dim=1) - 1
            for layer_index in layers:
                layer_hidden = hidden_states[layer_index + 1]
                pooled = layer_hidden[torch.arange(layer_hidden.size(0), device=device), token_positions]
                feature_batches[layer_index].append(pooled.detach().cpu())

    features_by_layer = {
        layer_index: torch.cat(chunks, dim=0)
        for layer_index, chunks in feature_batches.items()
    }
    return features_by_layer, action_indices, halt_steps, hidden_layer_count


def action_to_index(action, atol: float = 1e-6) -> int:
    target = np.array([*action.velocity_xyz, action.yaw_rate], dtype=float)
    distances = [
        float(
            np.linalg.norm(
                target - np.array([*candidate.velocity_xyz, candidate.yaw_rate], dtype=float)
            )
        )
        for candidate in ACTION_VOCABULARY
    ]
    best = int(np.argmin(distances))
    if distances[best] <= atol:
        return best
    return best


def _run_probe_epoch(
    *,
    probe: LayerProbe,
    loader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    layer_index: int,
    device: torch.device,
    training: bool,
) -> dict[str, float]:
    probe.train(mode=training)
    total_action_accuracy = 0.0
    total_halt_step_mae = 0.0
    batch_count = 0

    for batch in loader:
        batch_count += 1
        features = batch[f"layer_{layer_index}"].to(device=device, dtype=torch.float32)
        action_index = batch["action_index"].to(device)
        halt_step = batch["halt_step"].to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        action_logits, halt_logits = probe(features)
        loss = F.cross_entropy(action_logits, action_index) + 0.5 * F.cross_entropy(halt_logits, halt_step)

        if training:
            loss.backward()
            optimizer.step()

        total_action_accuracy += float((action_logits.argmax(dim=-1) == action_index).float().mean().item())
        total_halt_step_mae += float(torch.mean(torch.abs(halt_logits.argmax(dim=-1).float() - halt_step.float())).item())

    return {
        "action_accuracy": total_action_accuracy / max(batch_count, 1),
        "halt_step_mae": total_halt_step_mae / max(batch_count, 1),
    }


def run_layer_sweep(config: LayerSweepConfig) -> LayerSweepSummary:
    set_seed(config.seed)
    device = torch.device(config.device if config.device != "cuda" or torch.cuda.is_available() else "cpu")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_tasks = ARCDroneBench(BenchmarkConfig(task_count=config.task_count, seed=config.seed)).generate_tasks()
    eval_tasks = ARCDroneBench(BenchmarkConfig(task_count=config.eval_task_count, seed=config.seed + 1)).generate_tasks()

    probe_layers = list(config.layers)
    train_features, train_actions, train_halts, hidden_layer_count = build_teacher_features(
        tasks=train_tasks,
        foundation_model_id=config.foundation_model_id,
        layers=probe_layers or [0],
        max_length=config.max_length,
        device=device,
    )
    if not probe_layers:
        probe_layers = choose_layer_indices(
            total_layers=hidden_layer_count,
            explicit_layers=config.layers,
            fractions=config.layer_fractions,
        )
        train_features, train_actions, train_halts, hidden_layer_count = build_teacher_features(
            tasks=train_tasks,
            foundation_model_id=config.foundation_model_id,
            layers=probe_layers,
            max_length=config.max_length,
            device=device,
        )
    eval_features, eval_actions, eval_halts, _ = build_teacher_features(
        tasks=eval_tasks,
        foundation_model_id=config.foundation_model_id,
        layers=probe_layers,
        max_length=config.max_length,
        device=device,
    )

    train_dataset = TeacherFeatureDataset(
        features_by_layer=train_features,
        action_indices=train_actions,
        halt_steps=train_halts,
    )
    eval_dataset = TeacherFeatureDataset(
        features_by_layer=eval_features,
        action_indices=eval_actions,
        halt_steps=eval_halts,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.probe_batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.probe_batch_size, shuffle=False)

    hidden_size = next(iter(train_features.values())).shape[-1]
    results: list[LayerSweepResult] = []
    best_layer_index = -1
    best_eval_action_accuracy = -math.inf

    for layer_index in probe_layers:
        probe = LayerProbe(hidden_size=hidden_size, refinement_steps=6).to(device)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=config.probe_learning_rate)
        train_metrics = {"action_accuracy": 0.0, "halt_step_mae": 0.0}
        eval_metrics = {"action_accuracy": 0.0, "halt_step_mae": 0.0}
        for _ in range(config.probe_epochs):
            train_metrics = _run_probe_epoch(
                probe=probe,
                loader=train_loader,
                optimizer=optimizer,
                layer_index=layer_index,
                device=device,
                training=True,
            )
            eval_metrics = _run_probe_epoch(
                probe=probe,
                loader=eval_loader,
                optimizer=optimizer,
                layer_index=layer_index,
                device=device,
                training=False,
            )

        result = LayerSweepResult(
            layer_index=layer_index,
            train_action_accuracy=train_metrics["action_accuracy"],
            eval_action_accuracy=eval_metrics["action_accuracy"],
            train_halt_step_mae=train_metrics["halt_step_mae"],
            eval_halt_step_mae=eval_metrics["halt_step_mae"],
        )
        results.append(result)
        if result.eval_action_accuracy > best_eval_action_accuracy:
            best_eval_action_accuracy = result.eval_action_accuracy
            best_layer_index = layer_index

    summary = LayerSweepSummary(
        foundation_model_id=config.foundation_model_id,
        output_dir=output_dir.as_posix(),
        hidden_layer_count=hidden_layer_count,
        selected_layers=probe_layers,
        best_layer_index=best_layer_index,
        best_eval_action_accuracy=best_eval_action_accuracy,
        results=results,
    )
    (output_dir / "layer_sweep_summary.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def format_layer_sweep_summary(summary: LayerSweepSummary) -> str:
    lines = [
        f"layer_sweep_complete output_dir={summary.output_dir}",
        f"foundation_model_id={summary.foundation_model_id}",
        f"hidden_layer_count={summary.hidden_layer_count}",
        f"selected_layers={','.join(str(layer) for layer in summary.selected_layers)}",
        f"best_layer_index={summary.best_layer_index}",
        f"best_eval_action_accuracy={summary.best_eval_action_accuracy:.4f}",
    ]
    for result in summary.results:
        lines.append(
            "layer_result "
            f"layer={result.layer_index} "
            f"train_action_accuracy={result.train_action_accuracy:.4f} "
            f"eval_action_accuracy={result.eval_action_accuracy:.4f} "
            f"train_halt_step_mae={result.train_halt_step_mae:.4f} "
            f"eval_halt_step_mae={result.eval_halt_step_mae:.4f}"
        )
    return "\n".join(lines)
