"""Cloud-GPU smoke test and export helpers for the ARC-drone stack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch

from .config import DeploymentConfig, ReasonerConfig
from .export_tensorrt import build_trtexec_command, export_reasoner_to_onnx
from .model import TRMReasoner


@dataclass(frozen=True, slots=True)
class GpuSmokeReport:
    """Summarizes one CUDA smoke test run."""

    device_name: str
    batch_size: int
    latency_ms: float
    halted_at_step: int
    onnx_output_path: str | None
    engine_output_path: str | None
    trtexec_command: tuple[str, ...] | None


def require_cuda() -> torch.device:
    """Returns the CUDA device or raises a clear runtime error."""

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Run this command on a cloud GPU VM, for example an A100 40/80GB on RunPod or Vast.ai."
        )
    return torch.device("cuda")


def run_gpu_smoke(
    *,
    batch_size: int = 1,
    warmup_steps: int = 3,
    seed: int = 7,
    export_onnx: bool = False,
    onnx_output_path: str | Path = "artifacts/onnx/trm_reasoner.onnx",
    engine_output_path: str | Path = "artifacts/trt/trm_reasoner.plan",
    deployment: DeploymentConfig | None = None,
    config: ReasonerConfig | None = None,
) -> GpuSmokeReport:
    """Runs a minimal CUDA forward pass and optionally exports ONNX/TensorRT artifacts."""

    if batch_size <= 0:
        raise ValueError("batch_size must be strictly positive.")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative.")

    device = require_cuda()
    torch.manual_seed(seed)
    reasoner_config = config or ReasonerConfig()
    deployment_config = deployment or DeploymentConfig()

    model = TRMReasoner(reasoner_config).to(device).eval()
    grid = torch.randint(
        low=0,
        high=reasoner_config.color_count,
        size=(batch_size, reasoner_config.grid_height, reasoner_config.grid_width),
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        for _ in range(warmup_steps):
            model(grid)

        torch.cuda.synchronize()
        started = perf_counter()
        output = model(grid)
        torch.cuda.synchronize()
        latency_ms = (perf_counter() - started) * 1_000.0

    halted_at_step = int(output.halted_at_step.max().item())
    resolved_onnx_path: str | None = None
    resolved_engine_path: str | None = None
    trtexec_command: tuple[str, ...] | None = None

    if export_onnx:
        onnx_path = export_reasoner_to_onnx(
            output_path=onnx_output_path,
            config=reasoner_config,
            deployment=deployment_config,
        )
        resolved_onnx_path = onnx_path.as_posix()
        resolved_engine_path = Path(engine_output_path).as_posix()
        trtexec_command = tuple(
            build_trtexec_command(
                onnx_path=onnx_path,
                engine_path=resolved_engine_path,
                precision=deployment_config.trt_precision,
            )
        )

    return GpuSmokeReport(
        device_name=torch.cuda.get_device_name(device),
        batch_size=batch_size,
        latency_ms=latency_ms,
        halted_at_step=halted_at_step,
        onnx_output_path=resolved_onnx_path,
        engine_output_path=resolved_engine_path,
        trtexec_command=trtexec_command,
    )


def format_gpu_smoke_report(report: GpuSmokeReport) -> str:
    """Formats a concise, command-line friendly smoke report."""

    lines = [
        f"CUDA smoke test OK on {report.device_name}",
        f"batch_size={report.batch_size}",
        f"latency_ms={report.latency_ms:.3f}",
        f"halted_at_step={report.halted_at_step}",
    ]
    if report.onnx_output_path is not None:
        lines.append(f"onnx_output_path={report.onnx_output_path}")
    if report.engine_output_path is not None:
        lines.append(f"engine_output_path={report.engine_output_path}")
    if report.trtexec_command is not None:
        lines.append(f"trtexec_command={' '.join(report.trtexec_command)}")
    return "\n".join(lines)
