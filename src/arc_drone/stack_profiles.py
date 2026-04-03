"""Current validated stack profiles for the ARC-drone project."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Reference model family and downstream student target."""

    foundation_family: str
    foundation_variants: tuple[str, ...]
    student_target_max_params_millions: int
    quantized_student_ceiling_millions: int
    recursive_self_refinement_required: bool = True
    adaptive_halting_required: bool = True


@dataclass(frozen=True, slots=True)
class SimulationProfile:
    """Recommended simulator stack and fallback options."""

    primary_stack: str
    primary_components: tuple[str, ...]
    tolerated_stack: str
    tolerated_components: tuple[str, ...]
    secondary_backend: str
    deprecated_components: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrainingProfile:
    """Training, distillation, and deployment toolchain."""

    frameworks: tuple[str, ...]
    compression_tooling: tuple[str, ...]
    export_path: tuple[str, ...]
    recommended_cuda: str
    recommended_pytorch: str
    recommended_cloud_gpus: tuple[str, ...] = field(
        default_factory=lambda: ("A100 40GB", "A100 80GB", "H100 80GB")
    )

    def cloud_gpu_recommendation(self) -> str:
        """Returns the standard project recommendation used in docs and CLIs."""

        return (
            "For this stage, I recommend renting a GPU with at least 40 GB of VRAM "
            "(for example an A100 40/80GB) on RunPod or Vast.ai."
        )


@dataclass(frozen=True, slots=True)
class StackProfile:
    """Complete validated project stack."""

    validation_date: str
    model: ModelProfile
    simulation: SimulationProfile
    training: TrainingProfile


CURRENT_STACK_2026 = StackProfile(
    validation_date="2026-04-03",
    model=ModelProfile(
        foundation_family="Gemma 4",
        foundation_variants=("E2B", "E4B"),
        student_target_max_params_millions=50,
        quantized_student_ceiling_millions=100,
    ),
    simulation=SimulationProfile(
        primary_stack="ROS 2 Jazzy + Gazebo Harmonic + PX4 SITL",
        primary_components=("Ubuntu 24.04", "ROS 2 Jazzy", "Gazebo Harmonic", "PX4 SITL", "ros_gz"),
        tolerated_stack="ROS 2 Humble + Gazebo Harmonic",
        tolerated_components=("Ubuntu 22.04", "ROS 2 Humble", "Gazebo Harmonic", "PX4 SITL"),
        secondary_backend="AirSim",
        deprecated_components=("ROS 2 Iron",),
    ),
    training=TrainingProfile(
        frameworks=("PyTorch", "Transformers", "PEFT", "BitsAndBytes", "Unsloth"),
        compression_tooling=("NNCF", "PyTorch quantization"),
        export_path=("PyTorch", "ONNX", "TensorRT 10+", "TensorRT-LLM"),
        recommended_cuda="CUDA 12.4+",
        recommended_pytorch="PyTorch 2.5+",
    ),
)
