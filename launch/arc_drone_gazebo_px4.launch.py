"""Launch Gazebo Harmonic + PX4 SITL + ROS 2 bridges + ARC-drone node."""

from __future__ import annotations

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Generates the primary simulator bringup for ARC-Drone-Bench."""

    px4_autopilot_path = LaunchConfiguration("px4_autopilot_path")
    px4_make_target = LaunchConfiguration("px4_make_target")
    world_name = LaunchConfiguration("world_name")
    gz_clock_topic = LaunchConfiguration("gz_clock_topic")
    gz_image_topic = LaunchConfiguration("gz_image_topic")
    gz_camera_info_topic = LaunchConfiguration("gz_camera_info_topic")
    ros_image_topic = LaunchConfiguration("ros_image_topic")
    ros_camera_info_topic = LaunchConfiguration("ros_camera_info_topic")
    offboard_rate_hz = LaunchConfiguration("offboard_rate_hz")
    uxrce_udp_port = LaunchConfiguration("uxrce_udp_port")

    repo_root = Path(__file__).resolve().parents[1]
    repo_src = (repo_root / "src").as_posix()
    pythonpath = repo_src
    if os.environ.get("PYTHONPATH"):
        pythonpath = f"{repo_src}:{os.environ['PYTHONPATH']}"

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "px4_autopilot_path",
                default_value=str(Path.home() / "PX4-Autopilot"),
                description="Absolute path to the PX4-Autopilot checkout.",
            ),
            DeclareLaunchArgument(
                "px4_make_target",
                default_value="gz_x500_depth",
                description="PX4 SITL make target, for example gz_x500_depth or gz_x500_vision.",
            ),
            DeclareLaunchArgument(
                "world_name",
                default_value="default",
                description="Gazebo world name used for the world clock topic.",
            ),
            DeclareLaunchArgument(
                "gz_clock_topic",
                default_value=["/world/", world_name, "/clock"],
                description="Gazebo clock topic bridged to /clock.",
            ),
            DeclareLaunchArgument(
                "gz_image_topic",
                default_value="/camera",
                description="Gazebo image topic for the active vehicle camera.",
            ),
            DeclareLaunchArgument(
                "gz_camera_info_topic",
                default_value="/camera_info",
                description="Gazebo camera_info topic for the active vehicle camera.",
            ),
            DeclareLaunchArgument(
                "ros_image_topic",
                default_value="/camera/image_raw",
                description="ROS image topic consumed by the ARC-drone adapter.",
            ),
            DeclareLaunchArgument(
                "ros_camera_info_topic",
                default_value="/camera/camera_info",
                description="ROS camera_info topic bridged from Gazebo.",
            ),
            DeclareLaunchArgument(
                "offboard_rate_hz",
                default_value="20.0",
                description="Control loop frequency for the ARC-drone ROS 2 node.",
            ),
            DeclareLaunchArgument(
                "uxrce_udp_port",
                default_value="8888",
                description="UDP port used by the Micro XRCE-DDS agent.",
            ),
            ExecuteProcess(
                cmd=["MicroXRCEAgent", "udp4", "-p", uxrce_udp_port],
                name="microxrce_agent",
                output="screen",
            ),
            ExecuteProcess(
                cmd=["make", "px4_sitl", px4_make_target],
                cwd=px4_autopilot_path,
                additional_env={"PX4_GZ_WORLD": world_name},
                name="px4_gazebo_sitl",
                output="screen",
            ),
            TimerAction(
                period=5.0,
                actions=[
                    Node(
                        package="ros_gz_bridge",
                        executable="parameter_bridge",
                        name="arc_drone_gz_bridge",
                        output="screen",
                        arguments=[
                            [gz_clock_topic, "@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
                            [gz_image_topic, "@sensor_msgs/msg/Image[gz.msgs.Image"],
                            [gz_camera_info_topic, "@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo"],
                            "--ros-args",
                            "--remap",
                            [gz_clock_topic, ":=", "/clock"],
                            "--remap",
                            [gz_image_topic, ":=", ros_image_topic],
                            "--remap",
                            [gz_camera_info_topic, ":=", ros_camera_info_topic],
                        ],
                    )
                ],
            ),
            TimerAction(
                period=7.0,
                actions=[
                    ExecuteProcess(
                        cmd=[
                            "python3",
                            "-m",
                            "arc_drone.live_ros2_app",
                            "--image-topic",
                            ros_image_topic,
                            "--offboard-rate-hz",
                            offboard_rate_hz,
                        ],
                        cwd=repo_root.as_posix(),
                        additional_env={"PYTHONPATH": pythonpath},
                        output="screen",
                    )
                ],
            ),
        ]
    )
