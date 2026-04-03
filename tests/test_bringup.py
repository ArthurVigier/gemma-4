from pathlib import Path

from arc_drone.bringup import GazeboPx4BringupConfig


def test_bringup_bridge_arguments_cover_clock_and_camera_streams() -> None:
    config = GazeboPx4BringupConfig(
        px4_autopilot_path=Path.cwd().as_posix(),
        gz_clock_topic="/world/default/clock",
        gz_image_topic="/camera",
        gz_camera_info_topic="/camera_info",
    )

    assert config.bridge_arguments() == [
        "/world/default/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
        "/camera@sensor_msgs/msg/Image[gz.msgs.Image",
        "/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
    ]


def test_bringup_remaps_target_adapter_topics() -> None:
    config = GazeboPx4BringupConfig(
        px4_autopilot_path=Path.cwd().as_posix(),
        ros_image_topic="/camera/image_raw",
        ros_camera_info_topic="/camera/camera_info",
    )

    assert config.bridge_remaps() == [
        ("/world/default/clock", "/clock"),
        ("/camera", "/camera/image_raw"),
        ("/camera_info", "/camera/camera_info"),
    ]


def test_bringup_validation_rejects_missing_px4_checkout() -> None:
    config = GazeboPx4BringupConfig(px4_autopilot_path="/path/that/does/not/exist")

    try:
        config.validate()
    except ValueError as exc:
        assert "does not exist" in str(exc)
    else:
        raise AssertionError("Expected validate() to reject a missing PX4 checkout.")
