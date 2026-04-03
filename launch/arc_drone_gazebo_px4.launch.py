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
    world_file_path = LaunchConfiguration("world_file_path")
    gz_clock_topic = LaunchConfiguration("gz_clock_topic")
    gz_image_topic = LaunchConfiguration("gz_image_topic")
    gz_camera_info_topic = LaunchConfiguration("gz_camera_info_topic")
    gz_symmetry_marker_topic = LaunchConfiguration("gz_symmetry_marker_topic")
    gz_counting_marker_topic = LaunchConfiguration("gz_counting_marker_topic")
    gz_composition_marker_topic = LaunchConfiguration("gz_composition_marker_topic")
    gz_path_planning_marker_topic = LaunchConfiguration("gz_path_planning_marker_topic")
    ros_image_topic = LaunchConfiguration("ros_image_topic")
    ros_camera_info_topic = LaunchConfiguration("ros_camera_info_topic")
    ros_symmetry_marker_topic = LaunchConfiguration("ros_symmetry_marker_topic")
    ros_counting_marker_topic = LaunchConfiguration("ros_counting_marker_topic")
    ros_composition_marker_topic = LaunchConfiguration("ros_composition_marker_topic")
    ros_path_planning_marker_topic = LaunchConfiguration("ros_path_planning_marker_topic")
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
                default_value="arc_drone_bench_mission",
                description="Gazebo world name used for the world clock topic.",
            ),
            DeclareLaunchArgument(
                "world_file_path",
                default_value=(repo_root / "assets" / "gazebo" / "worlds" / "arc_drone_bench_mission.world").as_posix(),
                description="Absolute path to the Gazebo world file with mission markers.",
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
                "gz_symmetry_marker_topic",
                default_value="/model/arc_marker_symmetry/odometry",
                description="Gazebo odometry topic for the symmetry mission marker.",
            ),
            DeclareLaunchArgument(
                "gz_counting_marker_topic",
                default_value="/model/arc_marker_counting/odometry",
                description="Gazebo odometry topic for the counting mission marker.",
            ),
            DeclareLaunchArgument(
                "gz_composition_marker_topic",
                default_value="/model/arc_marker_composition/odometry",
                description="Gazebo odometry topic for the composition mission marker.",
            ),
            DeclareLaunchArgument(
                "gz_path_planning_marker_topic",
                default_value="/model/arc_marker_path_planning/odometry",
                description="Gazebo odometry topic for the path-planning mission marker.",
            ),
            DeclareLaunchArgument(
                "ros_symmetry_marker_topic",
                default_value="/arc_drone/mission_markers/arc_marker_symmetry/odometry",
                description="ROS odometry topic for the symmetry mission marker.",
            ),
            DeclareLaunchArgument(
                "ros_counting_marker_topic",
                default_value="/arc_drone/mission_markers/arc_marker_counting/odometry",
                description="ROS odometry topic for the counting mission marker.",
            ),
            DeclareLaunchArgument(
                "ros_composition_marker_topic",
                default_value="/arc_drone/mission_markers/arc_marker_composition/odometry",
                description="ROS odometry topic for the composition mission marker.",
            ),
            DeclareLaunchArgument(
                "ros_path_planning_marker_topic",
                default_value="/arc_drone/mission_markers/arc_marker_path_planning/odometry",
                description="ROS odometry topic for the path-planning mission marker.",
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
                additional_env={
                    "PX4_GZ_WORLD": world_name,
                    "PX4_GZ_WORLD_FILE": world_file_path,
                    "GZ_SIM_RESOURCE_PATH": f"{(repo_root / 'assets' / 'gazebo' / 'models').as_posix()}:{(repo_root / 'assets' / 'gazebo' / 'worlds').as_posix()}",
                },
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
                            [gz_symmetry_marker_topic, "@nav_msgs/msg/Odometry[gz.msgs.Odometry"],
                            [gz_counting_marker_topic, "@nav_msgs/msg/Odometry[gz.msgs.Odometry"],
                            [gz_composition_marker_topic, "@nav_msgs/msg/Odometry[gz.msgs.Odometry"],
                            [gz_path_planning_marker_topic, "@nav_msgs/msg/Odometry[gz.msgs.Odometry"],
                            "--ros-args",
                            "--remap",
                            [gz_clock_topic, ":=", "/clock"],
                            "--remap",
                            [gz_image_topic, ":=", ros_image_topic],
                            "--remap",
                            [gz_camera_info_topic, ":=", ros_camera_info_topic],
                            "--remap",
                            [gz_symmetry_marker_topic, ":=", ros_symmetry_marker_topic],
                            "--remap",
                            [gz_counting_marker_topic, ":=", ros_counting_marker_topic],
                            "--remap",
                            [gz_composition_marker_topic, ":=", ros_composition_marker_topic],
                            "--remap",
                            [gz_path_planning_marker_topic, ":=", ros_path_planning_marker_topic],
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
                            "--mission-marker-topic",
                            ["arc_marker_symmetry=", ros_symmetry_marker_topic],
                            "--mission-marker-topic",
                            ["arc_marker_counting=", ros_counting_marker_topic],
                            "--mission-marker-topic",
                            ["arc_marker_composition=", ros_composition_marker_topic],
                            "--mission-marker-topic",
                            ["arc_marker_path_planning=", ros_path_planning_marker_topic],
                        ],
                        cwd=repo_root.as_posix(),
                        additional_env={"PYTHONPATH": pythonpath},
                        output="screen",
                    )
                ],
            ),
        ]
    )
