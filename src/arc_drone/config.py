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
    hidden_size: int = 128
    refinement_steps: int = 6
    halting_threshold: float = 0.82
    action_dim: int = 8


@dataclass(slots=True)
class BenchmarkConfig:
    """Benchmark generation and measurement parameters."""

    task_count: int = 200
    grid_height: int = 30
    grid_width: int = 30
    seed: int = 7
    latency_budget_ms: float = 80.0
    energy_budget_joules: float = 2.5
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
            "grid": {0: "batch"},
            "action_logits": {0: "batch"},
            "halt_probabilities": {0: "batch"},
        }
    )
