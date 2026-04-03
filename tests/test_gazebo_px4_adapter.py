from dataclasses import dataclass

import numpy as np
import pytest

from arc_drone.arc_types import DroneAction
from arc_drone.gazebo_px4_adapter import (
    DictionaryMessageFactory,
    GazeboPx4Adapter,
    GazeboPx4AdapterConfig,
    GazeboPx4Publishers,
)


@dataclass
class _FakeStamp:
    sec: int
    nanosec: int


@dataclass
class _FakeHeader:
    stamp: _FakeStamp


@dataclass
class _FakeImage:
    height: int
    width: int
    encoding: str
    step: int
    data: bytes
    header: _FakeHeader


@dataclass
class _FakeVehicleStatus:
    timestamp: int
    nav_state: int
    arming_state: int
    pre_flight_checks_pass: bool
    failsafe: bool = False


@dataclass
class _FakeVehicleOdometry:
    timestamp: int
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]
    pose_frame: int
    velocity_frame: int
    quality: int = 0


@dataclass
class _FakeVehicleCommandAck:
    timestamp: int
    command: int
    result: int
    result_param1: int = 0
    result_param2: int = 0


class _Recorder:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, message: object) -> None:
        self.messages.append(message)


def test_decode_bgr8_image_converts_to_rgb() -> None:
    bgr_pixels = np.array(
        [
            [[0, 0, 255], [0, 255, 0]],
            [[255, 0, 0], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    image = _FakeImage(
        height=2,
        width=2,
        encoding="bgr8",
        step=6,
        data=bgr_pixels.tobytes(),
        header=_FakeHeader(_FakeStamp(sec=3, nanosec=50)),
    )

    observation = GazeboPx4Adapter().ingest_image_message(image)

    assert observation.rgb_image.shape == (2, 2, 3)
    assert observation.timestamp_ns == 3_000_000_050
    assert observation.rgb_image[0, 0].tolist() == [255, 0, 0]
    assert observation.rgb_image[1, 0].tolist() == [0, 0, 255]


def test_ros_enu_to_px4_ned_matches_px4_frame_guidance() -> None:
    converted = GazeboPx4Adapter.ros_enu_to_px4_ned((1.5, -2.0, 0.25))

    assert converted == (-2.0, 1.5, -0.25)


def test_command_bundle_arms_after_warmup_cycles() -> None:
    config = GazeboPx4AdapterConfig(offboard_warmup_cycles=2)
    adapter = GazeboPx4Adapter(config=config, clock_ns=lambda: 25_000_000)
    action = DroneAction((0.2, 0.4, -0.1), yaw_rate=0.3, halt_probability=0.9)

    first = adapter.build_command_bundle(action)
    second = adapter.build_command_bundle(action)

    assert first.offboard_control_mode["velocity"] is True
    assert first.trajectory_setpoint["velocity"] == [0.4, 0.2, 0.1]
    assert first.trajectory_setpoint["yawspeed"] == -0.3
    assert first.vehicle_commands == []
    assert [message["command"] for message in second.vehicle_commands] == [
        DictionaryMessageFactory.VEHICLE_CMD_DO_SET_MODE,
        DictionaryMessageFactory.VEHICLE_CMD_COMPONENT_ARM_DISARM,
    ]
    assert second.vehicle_commands[0]["param1"] == 1.0
    assert second.vehicle_commands[0]["param2"] == 6.0
    assert second.vehicle_commands[1]["param1"] == 1.0


def test_control_state_is_ready_when_px4_reports_armed_offboard() -> None:
    adapter = GazeboPx4Adapter()
    adapter.ingest_vehicle_status_message(
        _FakeVehicleStatus(
            timestamp=1,
            nav_state=DictionaryMessageFactory.NAVIGATION_STATE_OFFBOARD,
            arming_state=DictionaryMessageFactory.ARMING_STATE_ARMED,
            pre_flight_checks_pass=True,
        )
    )
    adapter.ingest_vehicle_odometry_message(
        _FakeVehicleOdometry(
            timestamp=2,
            position=(1.0, 2.0, -3.0),
            velocity=(0.1, 0.2, -0.3),
            angular_velocity=(0.01, 0.02, 0.03),
            pose_frame=1,
            velocity_frame=1,
            quality=100,
        )
    )

    state = adapter.control_state()

    assert state.telemetry_available is True
    assert state.preflight_checks_pass is True
    assert state.armed is True
    assert state.offboard_enabled is True
    assert state.ready_for_control is True
    assert adapter.latest_odometry() is not None


def test_send_action_raises_when_preflight_checks_fail_after_warmup() -> None:
    publishers = GazeboPx4Publishers(
        offboard_control_mode=_Recorder(),
        trajectory_setpoint=_Recorder(),
        vehicle_command=_Recorder(),
    )
    adapter = GazeboPx4Adapter(
        config=GazeboPx4AdapterConfig(offboard_warmup_cycles=1),
        publishers=publishers,
        clock_ns=lambda: 10_000_000,
    )
    adapter.ingest_vehicle_status_message(
        _FakeVehicleStatus(
            timestamp=1,
            nav_state=0,
            arming_state=DictionaryMessageFactory.ARMING_STATE_DISARMED,
            pre_flight_checks_pass=False,
        )
    )

    with pytest.raises(RuntimeError, match="pre-flight checks"):
        adapter.send_action(DroneAction((0.1, 0.2, 0.3), yaw_rate=0.0, halt_probability=0.7))

    assert len(publishers.offboard_control_mode.messages) == 1
    assert len(publishers.trajectory_setpoint.messages) == 1
    assert len(publishers.vehicle_command.messages) == 0


def test_send_action_raises_when_px4_rejects_offboard_or_arm_command() -> None:
    publishers = GazeboPx4Publishers(
        offboard_control_mode=_Recorder(),
        trajectory_setpoint=_Recorder(),
        vehicle_command=_Recorder(),
    )
    adapter = GazeboPx4Adapter(
        config=GazeboPx4AdapterConfig(offboard_warmup_cycles=1),
        publishers=publishers,
        clock_ns=lambda: 20_000_000,
    )
    adapter.ingest_vehicle_status_message(
        _FakeVehicleStatus(
            timestamp=1,
            nav_state=0,
            arming_state=DictionaryMessageFactory.ARMING_STATE_DISARMED,
            pre_flight_checks_pass=True,
        )
    )
    adapter.ingest_vehicle_command_ack_message(
        _FakeVehicleCommandAck(
            timestamp=2,
            command=DictionaryMessageFactory.VEHICLE_CMD_DO_SET_MODE,
            result=DictionaryMessageFactory.VEHICLE_CMD_RESULT_DENIED,
        )
    )

    with pytest.raises(RuntimeError, match="rejected command"):
        adapter.send_action(DroneAction((0.1, 0.2, 0.3), yaw_rate=0.0, halt_probability=0.7))


def test_send_action_raises_when_transition_timeout_expires_without_offboard() -> None:
    publishers = GazeboPx4Publishers(
        offboard_control_mode=_Recorder(),
        trajectory_setpoint=_Recorder(),
        vehicle_command=_Recorder(),
    )
    clock = {"now": 0}
    adapter = GazeboPx4Adapter(
        config=GazeboPx4AdapterConfig(offboard_warmup_cycles=1, transition_timeout_s=0.5),
        publishers=publishers,
        clock_ns=lambda: clock["now"],
    )
    adapter.ingest_vehicle_status_message(
        _FakeVehicleStatus(
            timestamp=1,
            nav_state=0,
            arming_state=DictionaryMessageFactory.ARMING_STATE_DISARMED,
            pre_flight_checks_pass=True,
        )
    )

    adapter.send_action(DroneAction((0.1, 0.2, 0.3), yaw_rate=0.0, halt_probability=0.7))
    clock["now"] = 1_000_000_000

    with pytest.raises(RuntimeError, match="OFFBOARD mode"):
        adapter.send_action(DroneAction((0.1, 0.2, 0.3), yaw_rate=0.0, halt_probability=0.7))


def test_send_action_publishes_offboard_and_vehicle_commands() -> None:
    publishers = GazeboPx4Publishers(
        offboard_control_mode=_Recorder(),
        trajectory_setpoint=_Recorder(),
        vehicle_command=_Recorder(),
    )
    adapter = GazeboPx4Adapter(
        config=GazeboPx4AdapterConfig(offboard_warmup_cycles=1),
        publishers=publishers,
        clock_ns=lambda: 10_000_000,
    )

    adapter.send_action(DroneAction((0.1, 0.2, 0.3), yaw_rate=0.0, halt_probability=0.7))

    assert len(publishers.offboard_control_mode.messages) == 1
    assert len(publishers.trajectory_setpoint.messages) == 1
    assert len(publishers.vehicle_command.messages) == 2
