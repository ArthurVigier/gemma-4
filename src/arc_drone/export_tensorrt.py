"""ONNX and TensorRT export helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .config import DeploymentConfig, ReasonerConfig
from .model import TRMReasoner


class _OnnxReasonerWrapper(nn.Module):
    """Torch export wrapper that exposes only ONNX-friendly tensors."""

    def __init__(self, model: TRMReasoner) -> None:
        super().__init__()
        self.model = model

    def forward(self, grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.model(grid)
        return output.action_logits, output.halt_probabilities, output.halted_at_step


def export_reasoner_to_onnx(
    output_path: str | Path,
    config: ReasonerConfig | None = None,
    deployment: DeploymentConfig | None = None,
) -> Path:
    """Exports the TRM-like reasoner to ONNX for downstream TensorRT conversion."""

    reasoner_config = config or ReasonerConfig()
    deployment_config = deployment or DeploymentConfig()
    model = TRMReasoner(reasoner_config).eval()
    return export_reasoner_model_to_onnx(
        model=model,
        output_path=output_path,
        deployment=deployment_config,
    )


def export_reasoner_model_to_onnx(
    model: TRMReasoner,
    output_path: str | Path,
    deployment: DeploymentConfig | None = None,
) -> Path:
    """Exports a specific reasoner instance to ONNX."""

    deployment_config = deployment or DeploymentConfig()
    model = _OnnxReasonerWrapper(model.eval()).eval()
    reasoner_config = model.model.config
    dummy = torch.zeros(
        1,
        reasoner_config.grid_height,
        reasoner_config.grid_width,
        dtype=torch.long,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        output_path.as_posix(),
        input_names=["grid"],
        output_names=["action_logits", "halt_probabilities", "halted_at_step"],
        dynamic_axes=deployment_config.dynamic_axes,
        opset_version=deployment_config.onnx_opset,
    )
    return output_path


def build_trtexec_command(onnx_path: str | Path, engine_path: str | Path, precision: str = "int8") -> list[str]:
    """Returns a reproducible `trtexec` command for the exported ONNX model."""

    precision_flag = {"fp16": "--fp16", "int8": "--int8", "int4": "--best"}.get(precision.lower(), "--fp16")
    return [
        "trtexec",
        f"--onnx={Path(onnx_path)}",
        f"--saveEngine={Path(engine_path)}",
        precision_flag,
        "--skipInference",
    ]
