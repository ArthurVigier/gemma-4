from arc_drone.arc_types import DroneAction
from arc_drone.gazebo_px4_adapter import (
    Px4ControlState,
    Px4OdometryTelemetry,
    Px4VehicleStatusTelemetry,
)
from arc_drone.supervision import (
    build_control_state_snapshot,
    control_state_transition_summary,
    snapshot_to_json,
    telemetry_log_line,
)


def test_snapshot_to_json_includes_px4_state_and_last_action() -> None:
    snapshot = build_control_state_snapshot(
        simulator_name="gazebo_harmonic_px4_sitl",
        control_state=Px4ControlState(
            telemetry_available=True,
            preflight_checks_pass=True,
            preflight_checks_known=True,
            armed=True,
            offboard_enabled=True,
            failsafe_active=False,
            last_command_error=None,
            ready_for_control=True,
        ),
        vehicle_status=Px4VehicleStatusTelemetry(
            timestamp_us=1,
            nav_state=14,
            arming_state=2,
            pre_flight_checks_pass=True,
            failsafe=False,
        ),
        odometry=Px4OdometryTelemetry(
            timestamp_us=2,
            position=(1.0, 2.0, -3.0),
            velocity=(0.1, 0.2, -0.3),
            angular_velocity=(0.01, 0.02, 0.03),
            pose_frame=1,
            velocity_frame=1,
            quality=100,
        ),
        last_action=DroneAction((0.3, 0.0, 0.0), yaw_rate=0.25, halt_probability=0.91),
        last_halted_at_step=4,
    )

    payload = snapshot_to_json(snapshot)

    assert '"ready_for_control": true' in payload
    assert '"armed": true' in payload
    assert '"last_halt_probability": 0.91' in payload
    assert '"position_ned": [1.0, 2.0, -3.0]' in payload


def test_control_state_transition_summary_reports_relevant_changes() -> None:
    previous = build_control_state_snapshot(
        simulator_name="sim",
        control_state=Px4ControlState(
            telemetry_available=True,
            preflight_checks_pass=True,
            preflight_checks_known=True,
            armed=False,
            offboard_enabled=False,
            failsafe_active=False,
            last_command_error=None,
            ready_for_control=False,
        ),
        vehicle_status=None,
        odometry=None,
    )
    current = build_control_state_snapshot(
        simulator_name="sim",
        control_state=Px4ControlState(
            telemetry_available=True,
            preflight_checks_pass=True,
            preflight_checks_known=True,
            armed=True,
            offboard_enabled=True,
            failsafe_active=False,
            last_command_error=None,
            ready_for_control=True,
        ),
        vehicle_status=None,
        odometry=None,
    )

    summary = control_state_transition_summary(previous, current)

    assert summary is not None
    assert "ready_for_control=True" in summary
    assert "armed=True" in summary
    assert "offboard_enabled=True" in summary


def test_telemetry_log_line_mentions_pose_velocity_and_halt() -> None:
    snapshot = build_control_state_snapshot(
        simulator_name="sim",
        control_state=Px4ControlState(
            telemetry_available=True,
            preflight_checks_pass=True,
            preflight_checks_known=True,
            armed=True,
            offboard_enabled=True,
            failsafe_active=False,
            last_command_error=None,
            ready_for_control=True,
        ),
        vehicle_status=Px4VehicleStatusTelemetry(
            timestamp_us=1,
            nav_state=14,
            arming_state=2,
            pre_flight_checks_pass=True,
            failsafe=False,
        ),
        odometry=Px4OdometryTelemetry(
            timestamp_us=2,
            position=(1.0, 2.0, -3.0),
            velocity=(0.1, 0.2, -0.3),
            angular_velocity=(0.01, 0.02, 0.03),
            pose_frame=1,
            velocity_frame=1,
            quality=100,
        ),
        last_action=DroneAction((0.0, 0.3, 0.0), yaw_rate=0.0, halt_probability=0.88),
        last_halted_at_step=3,
    )

    line = telemetry_log_line(snapshot)

    assert "pos_ned=(1.0, 2.0, -3.0)" in line
    assert "vel_ned=(0.1, 0.2, -0.3)" in line
    assert "halt_step=3" in line
