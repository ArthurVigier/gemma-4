"""Bringup helpers for ROS 2 Jazzy + Gazebo Harmonic + PX4 SITL."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .mission_targets import DEFAULT_MISSION_TARGET_TOPICS


@dataclass(frozen=True, slots=True)
class GazeboPx4BringupConfig:
    """Resolved bringup settings for the primary simulator stack."""

    px4_autopilot_path: str
    px4_make_target: str = "gz_x500_depth"
    world_name: str = "default"
    gz_image_topic: str = "/camera"
    gz_camera_info_topic: str = "/camera_info"
    ros_image_topic: str = "/camera/image_raw"
    ros_camera_info_topic: str = "/camera/camera_info"
    world_file_path: str | None = None
    ros_clock_topic: str = "/clock"
    gz_clock_topic: str = "/world/default/clock"
    offboard_rate_hz: float = 20.0
    uxrce_udp_port: int = 8888
    mission_marker_topics: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MISSION_TARGET_TOPICS))

    def px4_command(self) -> list[str]:
        """Returns the PX4 SITL command wrapped by the ROS 2 launcher."""

        return ["make", "px4_sitl", self.px4_make_target]

    def microxrce_agent_command(self) -> list[str]:
        """Returns the Micro XRCE-DDS agent command."""

        return ["MicroXRCEAgent", "udp4", "-p", str(self.uxrce_udp_port)]

    def bridge_arguments(self) -> list[str]:
        """Returns `parameter_bridge` topic rules for clock, camera, and mission markers."""

        marker_arguments = [
            f"/model/{entity_name}/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry"
            for entity_name in self.mission_marker_topics
        ]
        return [
            f"{self.gz_clock_topic}@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            f"{self.gz_image_topic}@sensor_msgs/msg/Image[gz.msgs.Image",
            f"{self.gz_camera_info_topic}@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
        ] + marker_arguments

    def bridge_remaps(self) -> list[tuple[str, str]]:
        """Maps Gazebo topic names to the local ROS topics consumed by the adapter."""

        marker_remaps = [
            (f"/model/{entity_name}/odometry", ros_topic)
            for entity_name, ros_topic in self.mission_marker_topics.items()
        ]
        return [
            (self.gz_clock_topic, self.ros_clock_topic),
            (self.gz_image_topic, self.ros_image_topic),
            (self.gz_camera_info_topic, self.ros_camera_info_topic),
        ] + marker_remaps

    def gazebo_resource_paths(self, repo_root: str | Path) -> list[str]:
        """Returns resource paths that should be visible to Gazebo."""

        repo_root_path = Path(repo_root)
        return [
            (repo_root_path / "assets" / "gazebo" / "models").as_posix(),
            (repo_root_path / "assets" / "gazebo" / "worlds").as_posix(),
        ]

    def validate(self) -> None:
        """Checks the minimal required bringup inputs."""

        px4_path = Path(self.px4_autopilot_path)
        if not px4_path.exists():
            raise ValueError(f"PX4 autopilot path does not exist: {px4_path}")
        if self.offboard_rate_hz <= 0:
            raise ValueError("offboard_rate_hz must be strictly positive.")
        if self.uxrce_udp_port <= 0:
            raise ValueError("uxrce_udp_port must be strictly positive.")
        if self.world_file_path is not None and not Path(self.world_file_path).exists():
            raise ValueError(f"Gazebo world file does not exist: {self.world_file_path}")
