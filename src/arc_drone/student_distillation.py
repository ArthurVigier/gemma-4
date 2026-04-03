"""Teacher-target caching and heavy student distillation helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .arc_drone_bench import ARCDroneBench
from .config import BenchmarkConfig, DeploymentConfig, ReasonerConfig
from .export_tensorrt import build_trtexec_command, export_reasoner_model_to_onnx
from .gemma_layer_sweep import LayerProbe, build_teacher_features
from .model import TRMReasoner
from .stack_profiles import CURRENT_STACK_2026
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


@dataclass(frozen=True, slots=True)
class DistillationCacheConfig:
    foundation_model_id: str = "google/gemma-4-e2b"
    teacher_layer_indices: tuple[int, ...] = (17,)
    teacher_feature_pooling: str = "mean"
    task_count: int = 32768
    eval_task_count: int = 4096
    seed: int = 7
    device: str = "cuda"
    output_dir: str = "artifacts/distillation_cache/gemma_e2b_l17"
    refinement_steps: int = 6
    teacher_probe_epochs: int = 5
    teacher_probe_learning_rate: float = 1e-3
    teacher_probe_batch_size: int = 64
    teacher_max_length: int = 768


@dataclass(frozen=True, slots=True)
class DistillationConfig:
    foundation_model_id: str = "google/gemma-4-e2b"
    teacher_layer_indices: tuple[int, ...] = (17,)
    teacher_feature_pooling: str = "mean"
    cache_dir: str | None = None
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
    teacher_representation_weight: float = 0.0
    teacher_kl_weight: float = 0.25
    teacher_probe_epochs: int = 5
    teacher_probe_learning_rate: float = 1e-3
    teacher_probe_batch_size: int = 64
    teacher_temperature: float = 3.0
    teacher_max_length: int = 768


class CachedDistillationDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset backed by cached teacher/student supervision tensors."""

    def __init__(self, cache_payload: dict[str, torch.Tensor | list[str]]) -> None:
        self.cache_payload = cache_payload
        self.tensor_keys = [
            "grid",
            "action_index",
            "action_target_vector",
            "halt_step",
            "teacher_features",
            "teacher_action_logits",
            "teacher_halt_logits",
        ]

    def __len__(self) -> int:
        return int(self.cache_payload["action_index"].shape[0])  # type: ignore[index]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            key: self.cache_payload[key][index]  # type: ignore[index]
            for key in self.tensor_keys
        }


class DistilledReasoner(nn.Module):
    """Student reasoner plus a projection head for teacher feature matching."""

    def __init__(self, *, reasoner_config: ReasonerConfig, teacher_hidden_size: int) -> None:
        super().__init__()
        self.reasoner = TRMReasoner(reasoner_config)
        self.teacher_projection = nn.Linear(reasoner_config.hidden_size, teacher_hidden_size)

    def forward(self, grid: torch.Tensor) -> tuple:
        output = self.reasoner(grid)
        student_repr = self.teacher_projection(output.hidden_states[-1])
        return output, student_repr


def _normalize_teacher_layer_indices(layer_indices: tuple[int, ...]) -> tuple[int, ...]:
    if not layer_indices:
        return (17,)
    return tuple(sorted({int(layer) for layer in layer_indices}))


def _combine_teacher_features(
    *,
    features_by_layer: dict[int, torch.Tensor],
    layer_indices: tuple[int, ...],
    pooling: str,
) -> torch.Tensor:
    ordered = [features_by_layer[layer_index] for layer_index in layer_indices]
    if pooling == "mean":
        return torch.stack(ordered, dim=0).mean(dim=0)
    if pooling == "concat":
        return torch.cat(ordered, dim=-1)
    raise ValueError(f"Unsupported teacher feature pooling: {pooling}")


def _fit_teacher_probe(
    *,
    teacher_features: torch.Tensor,
    action_indices: torch.Tensor,
    halt_steps: torch.Tensor,
    device: torch.device,
    probe_batch_size: int,
    probe_epochs: int,
    probe_learning_rate: float,
    refinement_steps: int,
) -> LayerProbe:
    probe = LayerProbe(hidden_size=teacher_features.shape[-1], refinement_steps=refinement_steps).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=probe_learning_rate)
    dataset = TensorDataset(teacher_features, action_indices, halt_steps)
    loader = DataLoader(dataset, batch_size=probe_batch_size, shuffle=True)

    for _ in range(probe_epochs):
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


def _stack_base_dataset(base_dataset: ArcStudentDataset) -> dict[str, torch.Tensor | list[str]]:
    grids = torch.stack([base_dataset[index]["grid"] for index in range(len(base_dataset))])
    action_indices = torch.stack([base_dataset[index]["action_index"] for index in range(len(base_dataset))])
    action_target_vectors = torch.stack([base_dataset[index]["action_target_vector"] for index in range(len(base_dataset))])
    halt_steps = torch.stack([base_dataset[index]["halt_step"] for index in range(len(base_dataset))])
    task_ids = [base_dataset.tasks[index].task_id for index in range(len(base_dataset))]
    return {
        "grid": grids,
        "action_index": action_indices,
        "action_target_vector": action_target_vectors,
        "halt_step": halt_steps,
        "task_id": task_ids,
    }


def _cache_split_payload(
    *,
    base_dataset: ArcStudentDataset,
    teacher_features: torch.Tensor,
    teacher_action_logits: torch.Tensor,
    teacher_halt_logits: torch.Tensor,
) -> dict[str, torch.Tensor | list[str]]:
    payload = _stack_base_dataset(base_dataset)
    payload["teacher_features"] = teacher_features
    payload["teacher_action_logits"] = teacher_action_logits
    payload["teacher_halt_logits"] = teacher_halt_logits
    return payload


def format_teacher_cache_summary(metadata: dict[str, object]) -> str:
    lines = [
        f"teacher_cache_complete output_dir={metadata['output_dir']}",
        f"foundation_model_id={metadata['foundation_model_id']}",
        f"teacher_layer_indices={','.join(str(x) for x in metadata['teacher_layer_indices'])}",
        f"teacher_feature_pooling={metadata['teacher_feature_pooling']}",
        f"teacher_feature_dim={metadata['teacher_feature_dim']}",
        f"train_task_count={metadata['train_task_count']}",
        f"eval_task_count={metadata['eval_task_count']}",
        f"train_cache_path={metadata['train_cache_path']}",
        f"eval_cache_path={metadata['eval_cache_path']}",
    ]
    return "\n".join(lines)


def build_teacher_target_cache(config: DistillationCacheConfig) -> dict[str, object]:
    """Builds and persists a reusable teacher-target cache."""

    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(config.device)
    layer_indices = _normalize_teacher_layer_indices(config.teacher_layer_indices)
    reasoner_config = ReasonerConfig(refinement_steps=config.refinement_steps)

    train_tasks = ARCDroneBench(BenchmarkConfig(task_count=config.task_count, seed=config.seed)).generate_tasks()
    eval_tasks = ARCDroneBench(BenchmarkConfig(task_count=config.eval_task_count, seed=config.seed + 1)).generate_tasks()

    train_base = ArcStudentDataset(train_tasks, reasoner_config)
    eval_base = ArcStudentDataset(eval_tasks, reasoner_config)

    train_features_by_layer, _, _, hidden_layer_count = build_teacher_features(
        tasks=train_tasks,
        foundation_model_id=config.foundation_model_id,
        layers=list(layer_indices),
        max_length=config.teacher_max_length,
        device=device,
    )
    eval_features_by_layer, _, _, _ = build_teacher_features(
        tasks=eval_tasks,
        foundation_model_id=config.foundation_model_id,
        layers=list(layer_indices),
        max_length=config.teacher_max_length,
        device=device,
    )

    combined_train_features = _combine_teacher_features(
        features_by_layer=train_features_by_layer,
        layer_indices=layer_indices,
        pooling=config.teacher_feature_pooling,
    )
    combined_eval_features = _combine_teacher_features(
        features_by_layer=eval_features_by_layer,
        layer_indices=layer_indices,
        pooling=config.teacher_feature_pooling,
    )

    train_action_indices = torch.stack([train_base[index]["action_index"] for index in range(len(train_base))])
    train_halt_steps = torch.stack([train_base[index]["halt_step"] for index in range(len(train_base))])
    teacher_probe = _fit_teacher_probe(
        teacher_features=combined_train_features,
        action_indices=train_action_indices,
        halt_steps=train_halt_steps,
        device=device,
        probe_batch_size=config.teacher_probe_batch_size,
        probe_epochs=config.teacher_probe_epochs,
        probe_learning_rate=config.teacher_probe_learning_rate,
        refinement_steps=config.refinement_steps,
    )
    train_teacher_action_logits, train_teacher_halt_logits = _project_teacher_logits(
        probe=teacher_probe,
        teacher_features=combined_train_features,
        device=device,
    )
    eval_teacher_action_logits, eval_teacher_halt_logits = _project_teacher_logits(
        probe=teacher_probe,
        teacher_features=combined_eval_features,
        device=device,
    )

    train_cache = _cache_split_payload(
        base_dataset=train_base,
        teacher_features=combined_train_features,
        teacher_action_logits=train_teacher_action_logits,
        teacher_halt_logits=train_teacher_halt_logits,
    )
    eval_cache = _cache_split_payload(
        base_dataset=eval_base,
        teacher_features=combined_eval_features,
        teacher_action_logits=eval_teacher_action_logits,
        teacher_halt_logits=eval_teacher_halt_logits,
    )

    train_cache_path = output_dir / "train_cache.pt"
    eval_cache_path = output_dir / "eval_cache.pt"
    teacher_probe_path = output_dir / "teacher_probe.pt"
    torch.save(train_cache, train_cache_path)
    torch.save(eval_cache, eval_cache_path)
    torch.save(
        {
            "state_dict": teacher_probe.state_dict(),
            "teacher_layer_indices": list(layer_indices),
            "teacher_feature_pooling": config.teacher_feature_pooling,
            "teacher_feature_dim": int(combined_train_features.shape[-1]),
            "refinement_steps": config.refinement_steps,
        },
        teacher_probe_path,
    )

    metadata: dict[str, object] = {
        "foundation_model_id": config.foundation_model_id,
        "output_dir": output_dir.as_posix(),
        "teacher_layer_indices": list(layer_indices),
        "teacher_feature_pooling": config.teacher_feature_pooling,
        "teacher_feature_dim": int(combined_train_features.shape[-1]),
        "hidden_layer_count": hidden_layer_count,
        "train_task_count": len(train_tasks),
        "eval_task_count": len(eval_tasks),
        "train_cache_path": train_cache_path.as_posix(),
        "eval_cache_path": eval_cache_path.as_posix(),
        "teacher_probe_path": teacher_probe_path.as_posix(),
        "refinement_steps": config.refinement_steps,
        "seed": config.seed,
    }
    (output_dir / "teacher_cache_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata


def _load_teacher_cache(cache_dir: str | Path) -> tuple[dict[str, torch.Tensor | list[str]], dict[str, torch.Tensor | list[str]], dict[str, object]]:
    cache_path = Path(cache_dir)
    metadata_path = cache_path / "teacher_cache_metadata.json"
    train_cache_path = cache_path / "train_cache.pt"
    eval_cache_path = cache_path / "eval_cache.pt"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    train_cache = torch.load(train_cache_path, map_location="cpu")
    eval_cache = torch.load(eval_cache_path, map_location="cpu")
    return train_cache, eval_cache, metadata


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
    ) * (temperature**2)


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
    """Runs heavy student distillation from a cached or freshly built teacher cache."""

    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(config.device)
    reasoner_config = build_reasoner_config(config)

    cache_dir = config.cache_dir
    if cache_dir is None:
        metadata = build_teacher_target_cache(
            DistillationCacheConfig(
                foundation_model_id=config.foundation_model_id,
                teacher_layer_indices=_normalize_teacher_layer_indices(config.teacher_layer_indices),
                teacher_feature_pooling=config.teacher_feature_pooling,
                task_count=config.task_count,
                eval_task_count=config.eval_task_count,
                seed=config.seed,
                device=config.device,
                output_dir=(output_dir / "teacher_cache").as_posix(),
                refinement_steps=config.refinement_steps,
                teacher_probe_epochs=config.teacher_probe_epochs,
                teacher_probe_learning_rate=config.teacher_probe_learning_rate,
                teacher_probe_batch_size=config.teacher_probe_batch_size,
                teacher_max_length=config.teacher_max_length,
            )
        )
        cache_dir = str(metadata["output_dir"])

    train_cache, eval_cache, cache_metadata = _load_teacher_cache(cache_dir)
    train_dataset = CachedDistillationDataset(train_cache)
    eval_dataset = CachedDistillationDataset(eval_cache)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)

    teacher_hidden_size = int(cache_metadata["teacher_feature_dim"])
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
            "cache_dir": cache_dir,
            "cache_metadata": cache_metadata,
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
    (output_dir / "distillation_summary.json").write_text(
        json.dumps(
            {
                **asdict(summary),
                "cache_dir": cache_dir,
                "cache_metadata": cache_metadata,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return summary
