from arc_drone.config import DeploymentConfig
from arc_drone.stack_profiles import CURRENT_STACK_2026


def test_current_stack_uses_jazzy_and_gazebo_harmonic_as_primary() -> None:
    assert CURRENT_STACK_2026.simulation.primary_stack == "ROS 2 Jazzy + Gazebo Harmonic + PX4 SITL"
    assert "ROS 2 Iron" in CURRENT_STACK_2026.simulation.deprecated_components
    assert CURRENT_STACK_2026.simulation.secondary_backend == "AirSim"


def test_deployment_defaults_match_validated_stack() -> None:
    config = DeploymentConfig()

    assert config.ros_distro == "jazzy"
    assert config.primary_simulator == "gazebo_harmonic_px4_sitl"
    assert config.secondary_simulator == "airsim"


def test_cloud_gpu_recommendation_mentions_40gb() -> None:
    message = CURRENT_STACK_2026.training.cloud_gpu_recommendation()

    assert "40 GB of VRAM" in message
    assert "RunPod" in message
    assert "Vast.ai" in message
