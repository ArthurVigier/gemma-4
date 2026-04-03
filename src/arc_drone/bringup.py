"""Bringup helpers for ROS 2 Jazzy + Gazebo Harmonic + PX4 SITL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
    ros_clock_topic: str = "/clock"
    gz_clock_topic: str = "/world/default/clock"
    offboard_rate_hz: float = 20.0
    uxrce_udp_port: int = 8888

    def px4_command(self) -> list[str]:
        """Returns the PX4 SITL command wrapped by the ROS 2 launcher."""

        return ["make", "px4_sitl", self.px4_make_target]

    def microxrce_agent_command(self) -> list[str]:
        """Returns the Micro XRCE-DDS agent command."""

        return ["MicroXRCEAgent", "udp4", "-p", str(self.uxrce_udp_port)]

    def bridge_arguments(self) -> list[str]:
        """Returns `parameter_bridge` topic rules for clock and camera streams."""

        return [
            f"{self.gz_clock_topic}@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            f"{self.gz_image_topic}@sensor_msgs/msg/Image[gz.msgs.Image",
            f"{self.gz_camera_info_topic}@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
        ]

    def bridge_remaps(self) -> list[tuple[str, str]]:
        """Maps Gazebo topic names to the local ROS topics consumed by the adapter."""

        return [
            (self.gz_clock_topic, self.ros_clock_topic),
            (self.gz_image_topic, self.ros_image_topic),
            (self.gz_camera_info_topic, self.ros_camera_info_topic),
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
