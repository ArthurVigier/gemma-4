from arc_drone.supervision import ControlStateSnapshot, control_state_events, control_event_to_json


def _snapshot(
    *,
    ready: bool,
    armed: bool,
    offboard: bool,
    failsafe: bool = False,
    command_error: str | None = None,
) -> ControlStateSnapshot:
    return ControlStateSnapshot(
        simulator_name="gazebo_harmonic_px4_sitl",
        telemetry_available=True,
        preflight_checks_pass=True,
        preflight_checks_known=True,
        armed=armed,
        offboard_enabled=offboard,
        failsafe_active=failsafe,
        ready_for_control=ready,
        last_command_error=command_error,
        status_timestamp_us=100,
        nav_state=14 if offboard else 0,
        arming_state=2 if armed else 1,
        odometry_timestamp_us=100,
        position_ned=(0.0, 0.0, 0.0),
        velocity_ned=(0.0, 0.0, 0.0),
        angular_velocity_frd=(0.0, 0.0, 0.0),
        last_halted_at_step=None,
        last_halt_probability=None,
        last_action_velocity_enu=None,
        last_action_yaw_rate=None,
    )


def test_control_state_events_emit_initialized_event() -> None:
    events = control_state_events(None, _snapshot(ready=False, armed=False, offboard=False))

    assert len(events) == 1
    assert events[0].event_type == "control_state_initialized"
    assert '"event_type": "control_state_initialized"' in control_event_to_json(events[0])


def test_control_state_events_emit_structured_transition_events() -> None:
    previous = _snapshot(ready=False, armed=False, offboard=False)
    current = _snapshot(ready=True, armed=True, offboard=True, command_error="PX4 rejected command 176.")

    events = control_state_events(previous, current)
    event_types = [event.event_type for event in events]

    assert "ready_for_control_changed" in event_types
    assert "arming_state_changed" in event_types
    assert "offboard_state_changed" in event_types
    assert "command_error" in event_types
