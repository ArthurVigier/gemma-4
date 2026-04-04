"""Configuration objects for the ARC-drone pipeline."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class VisionConfig:
    """Controls image to ARC-like grid conversion."""

    grid_height: int = 30
    grid_width: int = 30
    color_count: int = 10


@dataclass(slots=True)
class ReasonerConfig:
    """Parameters for the recursive TRM-like core."""

    grid_height: int = 30
    grid_width: int = 30
    color_count: int = 10
    hidden_size: int = 160
    refinement_steps: int = 6
    halting_threshold: float = 0.82
    action_dim: int = 8
    # Temporal video context: number of consecutive frames fed per inference.
    # T=1 reduces to the original single-frame behaviour.
    temporal_window: int = 4
    # Action chunking: number of future timesteps predicted in one forward pass.
    # chunk=1 reduces to the original single-action behaviour.
    action_chunk_size: int = 4


@dataclass(slots=True)
class BenchmarkConfig:
    """Benchmark generation and measurement parameters."""

    task_count: int = 200
    grid_height: int = 30
    grid_width: int = 30
    seed: int = 7
    latency_budget_ms: float = 80.0
    energy_budget_joules: float = 2.5
    real_data_path: str | None = None
    real_data_ratio: float = 0.0
    real_dataset: str | None = None
    real_dataset_split: str = "train"
    auto_discover_real_data: bool = True
    dataset_version: str = "arc_drone_v3_hybrid_auto"
    task_families: tuple[str, ...] = (
        "symmetry",
        "counting",
        "composition",
        "path_planning",
    )


@dataclass(slots=True)
class DeploymentConfig:
    """Deployment settings for ONNX and TensorRT export."""

    onnx_opset: int = 19
    trt_precision: str = "int8"
    ros_distro: str = "jazzy"
    primary_simulator: str = "gazebo_harmonic_px4_sitl"
    secondary_simulator: str = "airsim"
    dynamic_axes: dict[str, dict[int, str]] = field(
        default_factory=lambda: {
            "grids": {0: "batch", 1: "temporal_window"},
            "action_chunk_logits": {0: "batch"},
            "halt_probabilities": {0: "batch"},
        }
    )
