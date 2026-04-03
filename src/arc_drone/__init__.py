"""ARC-Drone package.

The package root intentionally avoids importing heavy optional dependencies
such as torch so lightweight modules remain importable in constrained
environments and during partial test runs.
"""

__all__ = [
    "arc_drone_bench",
    "benchmark_export",
    "bringup",
    "cloud_gpu",
    "config",
    "export_tensorrt",
    "gazebo_px4_adapter",
    "live_benchmark",
    "live_ros2_app",
    "metrics",
    "model",
    "pipeline_vision",
    "ros_node",
    "simulators",
    "stack_profiles",
    "supervision",
]
