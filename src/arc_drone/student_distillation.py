"""Teacher-guided distillation from Gemma hidden layers into the TRM student."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .arc_drone_bench import ARCDroneBench
from .config import BenchmarkConfig, DeploymentConfig, ReasonerConfig
from .export_tensorrt import build_trtexec_command, export_reasoner_model_to_onnx
from .gemma_layer_sweep import LayerProbe, build_teacher_features
from .model import TRMReasoner
from .student_training import (
    ArcStudentDataset,
    StudentTrainingSummary,
    action_vocabulary_tensor,
    build_reasoner_config,
    parameter_count_millions,
    select_device,
    set_seed,
    training_stack_status,
)
from .stack_profiles import CURRENT_STACK_2026


@dataclass(frozen=True, slots=True)
class DistillationConfig:
    foundation_model_id: str = "google/gemma-4-e2b"
    teacher_layer_index: int = 17
    task_count: int = 4096
    eval_task_count: int = 512
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    seed: int = 7
    device: str = "cuda"
    output_dir: str = "artifacts/checkpoints/trm_student_distilled"
    export_onnx: bool = False
    onnx_output_path: str | None = None
    trt_engine_output_path: str | None = None
    hidden_size: int = 96
    refinement_steps: int = 6
    halting_threshold: float = 0.82
    action_loss_weight: float = 2.0
    action_regression_weight: float = 0.5
    halt_loss_weight: float = 0.5
    teacher_representation_weight: float = 1.0
    teacher_kl_weight: float = 1.0
    teacher_probe_epochs: int = 5
    teacher_probe_learning_rate: float = 1e-3
    teacher_temperature: float = 2.0
    teacher_max_length: int = 768


class DistillationDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        *,
        base_dataset: ArcStudentDataset,
        teacher_features: torch.Tensor,
        teacher_action_logits: torch.Tensor,
        teacher_halt_logits: torch.Tensor,
    ) -> None:
        self.base_dataset = base_dataset
        self.teacher_features = teacher_features
        self.teacher_action_logits = teacher_action_logits
        self.teacher_halt_logits = teacher_halt_logits

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = dict(self.base_dataset[index])
        item["teacher_features"] = self.teacher_features[index]
        item["teacher_action_logits"] = self.teacher_action_logits[index]
        item["teacher_halt_logits"] = self.teacher_halt_logits[index]
        return item


class DistilledReasoner(nn.Module):
    def __init__(self, *, reasoner_config: ReasonerConfig, teacher_hidden_size: int) -> None:
        super().__init__()
        self.reasoner = TRMReasoner(reasoner_config)
        self.teacher_projection = nn.Linear(reasoner_config.hidden_size, teacher_hidden_size)

    def forward(self, grid: torch.Tensor):
        output = self.reasoner(grid)
        student_repr = self.teacher_projection(output.hidden_states[-1])
        return output, student_repr


def _teacher_kl_divergence(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)


def _fit_teacher_probe(
    *,
    teacher_features: torch.Tensor,
    action_indices: torch.Tensor,
    halt_steps: torch.Tensor,
    device: torch.device,
    config: DistillationConfig,
) -> LayerProbe:
    probe = LayerProbe(hidden_size=teacher_features.shape[-1], refinement_steps=config.refinement_steps).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=config.teacher_probe_learning_rate)
    dataset = torch.utils.data.TensorDataset(teacher_features, action_indices, halt_steps)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    for _ in range(config.teacher_probe_epochs):
        probe.train()
        for features, actions, halts in loader:
            optimizer.zero_grad(set_to_none=True)
            features = features.to(device=device, dtype=torch.float32)
            actions = actions.to(device)
            halts = (halts.to(device) - 1).clamp(min=0)
            action_logits, halt_logits = probe(features)
            loss = F.cross_entropy(action_logits, actions) + 0.5 * F.cross_entropy(halt_logits, halts)
            loss.backward()
            optimizer.step()

    probe.eval()
    return probe


def _project_teacher_logits(
    *,
    probe: LayerProbe,
    teacher_features: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(teacher_features, batch_size=256, shuffle=False)
    action_logits_batches: list[torch.Tensor] = []
    halt_logits_batches: list[torch.Tensor] = []
    with torch.no_grad():
        for features in loader:
            features = features.to(device=device, dtype=torch.float32)
            action_logits, halt_logits = probe(features)
            action_logits_batches.append(action_logits.cpu())
            halt_logits_batches.append(halt_logits.cpu())
    return torch.cat(action_logits_batches, dim=0), torch.cat(halt_logits_batches, dim=0)


def _compute_distillation_loss(
    *,
    model: DistilledReasoner,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    config: DistillationConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    grid = batch["grid"].to(device)
    action_index = batch["action_index"].to(device)
    action_target_vector = batch["action_target_vector"].to(device)
    halt_step = batch["halt_step"].to(device) - 1
    teacher_features = batch["teacher_features"].to(device=device, dtype=torch.float32)
    teacher_action_logits = batch["teacher_action_logits"].to(device=device, dtype=torch.float32)
    teacher_halt_logits = batch["teacher_halt_logits"].to(device=device, dtype=torch.float32)

    output, student_repr = model(grid)
    action_loss = F.cross_entropy(output.action_logits, action_index)
    action_probabilities = output.action_logits.softmax(dim=-1)
    expected_action = action_probabilities @ action_vocabulary_tensor(device=device)
    action_regression_loss = F.smooth_l1_loss(expected_action, action_target_vector)
    halt_loss = F.cross_entropy(output.halt_logits, halt_step)
    teacher_representation_loss = F.smooth_l1_loss(student_repr, teacher_features)
    teacher_action_kl = _teacher_kl_divergence(
        student_logits=output.action_logits,
        teacher_logits=teacher_action_logits,
        temperature=config.teacher_temperature,
    )
    teacher_halt_kl = _teacher_kl_divergence(
        student_logits=output.halt_logits,
        teacher_logits=teacher_halt_logits,
        temperature=config.teacher_temperature,
    )
    total_loss = (
        config.action_loss_weight * action_loss
        + config.action_regression_weight * action_regression_loss
        + config.halt_loss_weight * halt_loss
        + config.teacher_representation_weight * teacher_representation_loss
        + config.teacher_kl_weight * (teacher_action_kl + teacher_halt_kl)
    )

    action_accuracy = float((output.action_logits.argmax(dim=-1) == action_index).float().mean().item())
    halt_step_mae = float(torch.mean(torch.abs(output.halted_at_step.float() - (halt_step.float() + 1.0))).item())
    return total_loss, {
        "action_accuracy": action_accuracy,
        "halt_step_mae": halt_step_mae,
    }


def _run_epoch(
    *,
    model: DistilledReasoner,
    loader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: DistillationConfig,
    training: bool,
) -> dict[str, float]:
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
            loss, metrics = _compute_distillation_loss(model=model, batch=batch, device=device, config=config)
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += float(loss.item())
        total_action_accuracy += metrics["action_accuracy"]
        total_halt_step_mae += metrics["halt_step_mae"]

    return {
        "loss": total_loss / max(batch_count, 1),
        "action_accuracy": total_action_accuracy / max(batch_count, 1),
        "halt_step_mae": total_halt_step_mae / max(batch_count, 1),
    }


def distill_student(config: DistillationConfig) -> StudentTrainingSummary:
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(config.device)
    reasoner_config = build_reasoner_config(config)

    train_tasks = ARCDroneBench(BenchmarkConfig(task_count=config.task_count, seed=config.seed)).generate_tasks()
    eval_tasks = ARCDroneBench(BenchmarkConfig(task_count=config.eval_task_count, seed=config.seed + 1)).generate_tasks()

    teacher_layer = [config.teacher_layer_index]
    train_teacher_features, _, _, _ = build_teacher_features(
        tasks=train_tasks,
        foundation_model_id=config.foundation_model_id,
        layers=teacher_layer,
        max_length=config.teacher_max_length,
        device=device,
    )
    eval_teacher_features, _, _, _ = build_teacher_features(
        tasks=eval_tasks,
        foundation_model_id=config.foundation_model_id,
        layers=teacher_layer,
        max_length=config.teacher_max_length,
        device=device,
    )

    train_base = ArcStudentDataset(train_tasks, reasoner_config)
    eval_base = ArcStudentDataset(eval_tasks, reasoner_config)
    teacher_probe = _fit_teacher_probe(
        teacher_features=train_teacher_features[config.teacher_layer_index],
        action_indices=torch.stack([train_base[index]["action_index"] for index in range(len(train_base))]),
        halt_steps=torch.stack([train_base[index]["halt_step"] for index in range(len(train_base))]),
        device=device,
        config=config,
    )
    train_teacher_action_logits, train_teacher_halt_logits = _project_teacher_logits(
        probe=teacher_probe,
        teacher_features=train_teacher_features[config.teacher_layer_index],
        device=device,
    )
    eval_teacher_action_logits, eval_teacher_halt_logits = _project_teacher_logits(
        probe=teacher_probe,
        teacher_features=eval_teacher_features[config.teacher_layer_index],
        device=device,
    )
    train_dataset = DistillationDataset(
        base_dataset=train_base,
        teacher_features=train_teacher_features[config.teacher_layer_index],
        teacher_action_logits=train_teacher_action_logits,
        teacher_halt_logits=train_teacher_halt_logits,
    )
    eval_dataset = DistillationDataset(
        base_dataset=eval_base,
        teacher_features=eval_teacher_features[config.teacher_layer_index],
        teacher_action_logits=eval_teacher_action_logits,
        teacher_halt_logits=eval_teacher_halt_logits,
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)

    teacher_hidden_size = train_teacher_features[config.teacher_layer_index].shape[-1]
    model = DistilledReasoner(reasoner_config=reasoner_config, teacher_hidden_size=teacher_hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    history = []
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
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_action_accuracy": train_metrics["action_accuracy"],
            "train_halt_step_mae": train_metrics["halt_step_mae"],
            "eval_loss": eval_metrics["loss"],
            "eval_action_accuracy": eval_metrics["action_accuracy"],
            "eval_halt_step_mae": eval_metrics["halt_step_mae"],
        }
        history.append(epoch_metrics)

        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.reasoner.state_dict(),
            "projection_state_dict": model.teacher_projection.state_dict(),
            "distillation_config": asdict(config),
            "reasoner_config": asdict(reasoner_config),
            "epoch_metrics": epoch_metrics,
            "training_stack_status": training_stack_status(),
        }
        torch.save(checkpoint_payload, output_dir / f"epoch_{epoch:03d}.pt")
        torch.save(checkpoint_payload, output_dir / "latest.pt")
        if eval_metrics["action_accuracy"] >= best_eval_action_accuracy:
            best_eval_action_accuracy = eval_metrics["action_accuracy"]
            best_eval_halt_step_mae = eval_metrics["halt_step_mae"]
            torch.save(checkpoint_payload, best_checkpoint_path)

    onnx_output_path = None
    trtexec_command = None
    if config.export_onnx:
        resolved_onnx_path = Path(config.onnx_output_path or output_dir / "student_reasoner.onnx")
        export_reasoner_model_to_onnx(
            model=model.reasoner.cpu().eval(),
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
        parameter_count_millions=parameter_count_millions(model.reasoner),
        cloud_gpu_recommendation=CURRENT_STACK_2026.training.cloud_gpu_recommendation(),
        best_eval_action_accuracy=best_eval_action_accuracy,
        best_eval_halt_step_mae=best_eval_halt_step_mae,
        epochs=history,
        onnx_output_path=onnx_output_path,
        trtexec_command=trtexec_command,
    )
    (output_dir / "distillation_summary.json").write_text(json.dumps(asdict(summary), indent=2, sort_keys=True), encoding="utf-8")
    return summary
