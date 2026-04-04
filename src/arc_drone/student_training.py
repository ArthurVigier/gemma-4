"""Student training CLI helpers for ARC-Drone-Bench."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .arc_drone_bench import ARCDroneBench
from .arc_types import BenchmarkTask, DroneAction
from .config import BenchmarkConfig, DeploymentConfig, ReasonerConfig
from .export_tensorrt import build_trtexec_command, export_reasoner_model_to_onnx
from .model import TRMReasoner
from .stack_profiles import CURRENT_STACK_2026

ACTION_VOCABULARY: tuple[DroneAction, ...] = (
    DroneAction((0.3, 0.0, 0.0), 0.0, 0.0),
    DroneAction((-0.3, 0.0, 0.0), 0.0, 0.0),
    DroneAction((0.0, 0.3, 0.0), 0.0, 0.0),
    DroneAction((0.0, -0.3, 0.0), 0.0, 0.0),
    DroneAction((0.0, 0.0, 0.3), 0.0, 0.0),
    DroneAction((0.0, 0.0, -0.3), 0.0, 0.0),
    DroneAction((0.0, 0.0, 0.0), 0.25, 0.0),
    DroneAction((0.0, 0.0, 0.0), -0.25, 0.0),
)


@dataclass(frozen=True, slots=True)
class StudentTrainingConfig:
    """Configuration for student training on ARC-Drone-Bench."""

    foundation_model_id: str = "google/gemma-4-e4b"
    task_count: int = 4096
    eval_task_count: int = 512
    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    seed: int = 7
    device: str = "cuda"
    output_dir: str = "artifacts/checkpoints/trm_student"
    export_onnx: bool = False
    onnx_output_path: str | None = None
    trt_engine_output_path: str | None = None
    log_every_steps: int = 20
    num_workers: int = 0
    hidden_size: int = 96
    refinement_steps: int = 6
    halting_threshold: float = 0.82
    action_loss_weight: float = 2.0
    action_regression_weight: float = 0.5
    halt_loss_weight: float = 0.5
    temporal_window: int = 4
    action_chunk_size: int = 4


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """One epoch of train/eval metrics."""

    epoch: int
    train_loss: float
    train_action_accuracy: float
    train_halt_step_mae: float
    eval_loss: float
    eval_action_accuracy: float
    eval_halt_step_mae: float


@dataclass(frozen=True, slots=True)
class StudentTrainingSummary:
    """Serializable summary written by the training CLI."""

    foundation_model_id: str
    output_dir: str
    device: str
    parameter_count_millions: float
    cloud_gpu_recommendation: str
    best_eval_action_accuracy: float
    best_eval_halt_step_mae: float
    epochs: list[EpochMetrics]
    onnx_output_path: str | None
    trtexec_command: list[str] | None


class ArcStudentDataset(Dataset[dict[str, torch.Tensor]]):
    """Torch dataset built from synthetic ARC benchmark tasks.

    Returns:
        grids: (T, H, W) — temporal window of frames.
               For synthetic tasks the same grid is repeated T times (no motion).
               For real video sequences, consecutive frames are stacked here.
        action_indices: (chunk_size,) — action chunk targets.
               For synthetic tasks the same action is repeated chunk_size times.
        action_target_vectors: (chunk_size, 4) — dense control vectors per chunk step.
        halt_targets: (refinement_steps,) — unchanged, for the current step only.
        halt_step: scalar.
    """

    def __init__(self, tasks: list[BenchmarkTask], reasoner_config: ReasonerConfig) -> None:
        self.tasks = tasks
        self.reasoner_config = reasoner_config

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        task = self.tasks[index]
        halt_step = halt_probability_to_step(
            halt_probability=task.target_action.halt_probability,
            refinement_steps=self.reasoner_config.refinement_steps,
        )
        T = self.reasoner_config.temporal_window
        C = self.reasoner_config.action_chunk_size

        grid = torch.tensor(task.input_grid.values, dtype=torch.long)
        # Repeat single frame T times — architecture-ready, motion=0 for synthetic data.
        grids = grid.unsqueeze(0).expand(T, -1, -1)  # (T, H, W)

        action_idx = action_to_index(task.target_action)
        action_vec = action_to_vector(task.target_action)
        # Repeat same action C times — for synthetic data all chunk steps are identical.
        action_indices = torch.full((C,), action_idx, dtype=torch.long)
        action_target_vectors = action_vec.unsqueeze(0).expand(C, -1)  # (C, 4)

        return {
            "grids": grids,
            "action_indices": action_indices,
            "action_target_vectors": action_target_vectors,
            "halt_targets": build_halt_targets(
                halt_step=halt_step,
                refinement_steps=self.reasoner_config.refinement_steps,
            ),
            "halt_step": torch.tensor(halt_step, dtype=torch.long),
        }


def action_to_index(action: DroneAction, atol: float = 1e-6) -> int:
    """Maps a benchmark action to the nearest student action vocabulary entry."""

    for index, candidate in enumerate(ACTION_VOCABULARY):
        velocity_match = np.allclose(action.velocity_xyz, candidate.velocity_xyz, atol=atol)
        yaw_match = abs(float(action.yaw_rate) - float(candidate.yaw_rate)) <= atol
        if velocity_match and yaw_match:
            return index

    target = np.array([*action.velocity_xyz, action.yaw_rate], dtype=float)
    distances = [
        float(
            np.linalg.norm(
                target - np.array([*candidate.velocity_xyz, candidate.yaw_rate], dtype=float)
            )
        )
        for candidate in ACTION_VOCABULARY
    ]
    return int(np.argmin(distances))


def action_to_vector(action: DroneAction) -> torch.Tensor:
    """Converts an action envelope to a dense control vector."""

    return torch.tensor([*action.velocity_xyz, action.yaw_rate], dtype=torch.float32)


def action_vocabulary_tensor(*, device: torch.device) -> torch.Tensor:
    """Returns the action vocabulary as a dense tensor on the requested device."""

    return torch.tensor(
        [[*candidate.velocity_xyz, candidate.yaw_rate] for candidate in ACTION_VOCABULARY],
        dtype=torch.float32,
        device=device,
    )


def halt_probability_to_step(*, halt_probability: float, refinement_steps: int) -> int:
    """Converts a target halt probability into a target recursive step."""

    clipped = float(np.clip(halt_probability, 0.0, 1.0))
    step = int(round((1.0 - clipped) * max(refinement_steps - 1, 1))) + 1
    return int(max(1, min(refinement_steps, step)))


def build_halt_targets(*, halt_step: int, refinement_steps: int) -> torch.Tensor:
    """Builds one-hot halt-step targets for direct step classification."""

    targets = torch.zeros(refinement_steps, dtype=torch.float32)
    targets[max(0, halt_step - 1)] = 1.0
    return targets


def build_reasoner_config(config: StudentTrainingConfig) -> ReasonerConfig:
    """Builds the reasoner config used by training."""

    return ReasonerConfig(
        hidden_size=config.hidden_size,
        refinement_steps=config.refinement_steps,
        halting_threshold=config.halting_threshold,
        temporal_window=config.temporal_window,
        action_chunk_size=config.action_chunk_size,
    )


def parameter_count_millions(model: nn.Module) -> float:
    """Returns the model parameter count in millions."""

    total = sum(parameter.numel() for parameter in model.parameters())
    return total / 1_000_000.0


def build_dataloaders(
    config: StudentTrainingConfig,
    reasoner_config: ReasonerConfig,
) -> tuple[DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]]]:
    """Builds train/eval loaders from ARC-Drone-Bench tasks."""

    train_bench = ARCDroneBench(BenchmarkConfig(task_count=config.task_count, seed=config.seed))
    eval_bench = ARCDroneBench(BenchmarkConfig(task_count=config.eval_task_count, seed=config.seed + 1))
    train_dataset = ArcStudentDataset(train_bench.generate_tasks(), reasoner_config)
    eval_dataset = ArcStudentDataset(eval_bench.generate_tasks(), reasoner_config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, eval_loader


def select_device(device_name: str) -> torch.device:
    """Resolves the requested torch device."""

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA is not available. {CURRENT_STACK_2026.training.cloud_gpu_recommendation()}"
        )
    return torch.device(device_name)


def set_seed(seed: int) -> None:
    """Seeds Python, NumPy, and torch RNGs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def training_stack_status() -> dict[str, bool]:
    """Returns availability of the intended Gemma/Unsloth training stack."""

    status: dict[str, bool] = {}
    for module_name in ("unsloth", "transformers", "peft", "bitsandbytes"):
        try:
            __import__(module_name)
        except Exception:
            status[module_name] = False
        else:
            status[module_name] = True
    return status


def compute_loss(
    *,
    model: TRMReasoner,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    config: StudentTrainingConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Computes the combined action-chunk and halting loss.

    Action loss is averaged over the chunk dimension so that all future
    timesteps contribute equally to the gradient.
    """

    grids = batch["grids"].to(device)                           # (B, T, H, W)
    action_indices = batch["action_indices"].to(device)         # (B, C)
    action_target_vectors = batch["action_target_vectors"].to(device)  # (B, C, 4)
    halt_targets = batch["halt_targets"].to(device)
    halt_step = batch["halt_step"].to(device)

    output = model(grids)
    # output.action_chunk_logits: (B, C, action_dim)

    B, C, A = output.action_chunk_logits.shape
    vocab = action_vocabulary_tensor(device=device)             # (A, 4)

    # Cross-entropy over all chunk steps: reshape to (B*C, A) vs (B*C,)
    action_loss = F.cross_entropy(
        output.action_chunk_logits.reshape(B * C, A),
        action_indices.reshape(B * C),
    )

    # Regression: expected continuous action per chunk step
    action_probs = output.action_chunk_logits.softmax(dim=-1)  # (B, C, A)
    expected_actions = action_probs @ vocab                     # (B, C, 4)
    action_regression_loss = F.smooth_l1_loss(expected_actions, action_target_vectors)

    halt_supervision = halt_targets.argmax(dim=-1)
    halt_loss = F.cross_entropy(output.halt_logits, halt_supervision)

    total_loss = (
        config.action_loss_weight * action_loss
        + config.action_regression_weight * action_regression_loss
        + config.halt_loss_weight * halt_loss
    )

    # Accuracy measured on the first (current) chunk step only
    current_action_logits = output.action_chunk_logits[:, 0, :]  # (B, A)
    action_accuracy = float(
        (current_action_logits.argmax(dim=-1) == action_indices[:, 0]).float().mean().item()
    )
    halt_step_mae = float(
        torch.mean(torch.abs(output.halted_at_step.float() - halt_step.float())).item()
    )
    return total_loss, {
        "action_loss": float(action_loss.item()),
        "action_regression_loss": float(action_regression_loss.item()),
        "halt_loss": float(halt_loss.item()),
        "action_accuracy": action_accuracy,
        "halt_step_mae": halt_step_mae,
    }


def train_student(config: StudentTrainingConfig) -> StudentTrainingSummary:
    """Runs student training and writes checkpoints/summary artifacts."""

    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(config.device)
    reasoner_config = build_reasoner_config(config)
    model = TRMReasoner(reasoner_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_loader, eval_loader = build_dataloaders(config, reasoner_config)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    history: list[EpochMetrics] = []
    best_eval_action_accuracy = -1.0
    best_eval_halt_step_mae = float("inf")
    best_checkpoint_path = output_dir / "best_student.pt"

    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            config=config,
            training=True,
        )
        eval_metrics = _run_epoch(
            model=model,
            loader=eval_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            config=config,
            training=False,
        )
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_metrics["loss"],
            train_action_accuracy=train_metrics["action_accuracy"],
            train_halt_step_mae=train_metrics["halt_step_mae"],
            eval_loss=eval_metrics["loss"],
            eval_action_accuracy=eval_metrics["action_accuracy"],
            eval_halt_step_mae=eval_metrics["halt_step_mae"],
        )
        history.append(epoch_metrics)

        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_config": asdict(config),
            "reasoner_config": asdict(reasoner_config),
            "epoch_metrics": asdict(epoch_metrics),
            "training_stack_status": training_stack_status(),
        }
        torch.save(checkpoint_payload, output_dir / f"epoch_{epoch:03d}.pt")
        torch.save(checkpoint_payload, output_dir / "latest.pt")

        if eval_metrics["action_accuracy"] >= best_eval_action_accuracy:
            best_eval_action_accuracy = eval_metrics["action_accuracy"]
            best_eval_halt_step_mae = eval_metrics["halt_step_mae"]
            torch.save(checkpoint_payload, best_checkpoint_path)

    onnx_output_path: str | None = None
    trtexec_command: list[str] | None = None
    if config.export_onnx:
        resolved_onnx_path = Path(config.onnx_output_path or output_dir / "student_reasoner.onnx")
        export_reasoner_model_to_onnx(
            model=model.cpu().eval(),
            output_path=resolved_onnx_path,
            deployment=DeploymentConfig(),
        )
        onnx_output_path = resolved_onnx_path.as_posix()
        engine_path = Path(config.trt_engine_output_path or output_dir / "student_reasoner.plan")
        trtexec_command = build_trtexec_command(
            onnx_path=resolved_onnx_path,
            engine_path=engine_path,
            precision=DeploymentConfig().trt_precision,
        )

    summary = StudentTrainingSummary(
        foundation_model_id=config.foundation_model_id,
        output_dir=output_dir.as_posix(),
        device=device.type,
        parameter_count_millions=parameter_count_millions(model),
        cloud_gpu_recommendation=CURRENT_STACK_2026.training.cloud_gpu_recommendation(),
        best_eval_action_accuracy=best_eval_action_accuracy,
        best_eval_halt_step_mae=best_eval_halt_step_mae,
        epochs=history,
        onnx_output_path=onnx_output_path,
        trtexec_command=trtexec_command,
    )
    (output_dir / "training_summary.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def format_training_summary(summary: StudentTrainingSummary) -> str:
    """Formats the final training summary for command-line output."""

    lines = [
        f"student_training_complete output_dir={summary.output_dir}",
        f"foundation_model_id={summary.foundation_model_id}",
        f"device={summary.device}",
        f"parameter_count_millions={summary.parameter_count_millions:.3f}",
        f"best_eval_action_accuracy={summary.best_eval_action_accuracy:.4f}",
        f"best_eval_halt_step_mae={summary.best_eval_halt_step_mae:.4f}",
    ]
    if summary.onnx_output_path is not None:
        lines.append(f"onnx_output_path={summary.onnx_output_path}")
    if summary.trtexec_command is not None:
        lines.append(f"trtexec_command={' '.join(summary.trtexec_command)}")
    return "\n".join(lines)


def _run_epoch(
    *,
    model: TRMReasoner,
    loader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    config: StudentTrainingConfig,
    training: bool,
) -> dict[str, float]:
    """Runs one train or eval epoch."""

    model.train(mode=training)
    total_loss = 0.0
    total_action_accuracy = 0.0
    total_halt_step_mae = 0.0
    batch_count = 0

    for batch in loader:
        batch_count += 1
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            loss, metrics = compute_loss(model=model, batch=batch, device=device, config=config)

        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += float(loss.item())
        total_action_accuracy += metrics["action_accuracy"]
        total_halt_step_mae += metrics["halt_step_mae"]

    if batch_count == 0:
        raise ValueError("Empty dataloader encountered during training.")

    return {
        "loss": total_loss / batch_count,
        "action_accuracy": total_action_accuracy / batch_count,
        "halt_step_mae": total_halt_step_mae / batch_count,
    }
