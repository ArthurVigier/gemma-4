"""Supervision helpers for ROS2-facing PX4 control state publication."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from .arc_types import DroneAction
from .gazebo_px4_adapter import Px4ControlState, Px4OdometryTelemetry, Px4VehicleStatusTelemetry


@dataclass(slots=True)
class ControlStateSnapshot:
    """Serializable supervision snapshot exposed on the ROS2 control topic."""

    simulator_name: str
    telemetry_available: bool
    preflight_checks_pass: bool
    preflight_checks_known: bool
    armed: bool
    offboard_enabled: bool
    failsafe_active: bool
    ready_for_control: bool
    last_command_error: str | None
    status_timestamp_us: int | None
    nav_state: int | None
    arming_state: int | None
    odometry_timestamp_us: int | None
    position_ned: tuple[float, float, float] | None
    velocity_ned: tuple[float, float, float] | None
    angular_velocity_frd: tuple[float, float, float] | None
    last_halted_at_step: int | None
    last_halt_probability: float | None
    last_action_velocity_enu: tuple[float, float, float] | None
    last_action_yaw_rate: float | None


@dataclass(slots=True)
class ControlEvent:
    """Discrete supervision event published on the ROS2 control-events topic."""

    event_type: str
    severity: str
    message: str
    simulator_name: str
    status_timestamp_us: int | None
    ready_for_control: bool
    armed: bool
    offboard_enabled: bool
    failsafe_active: bool
    last_command_error: str | None


def build_control_state_snapshot(
    *,
    simulator_name: str,
    control_state: Px4ControlState,
    vehicle_status: Px4VehicleStatusTelemetry | None,
    odometry: Px4OdometryTelemetry | None,
    last_action: DroneAction | None = None,
    last_halted_at_step: int | None = None,
) -> ControlStateSnapshot:
    """Builds a JSON-serializable control state payload."""

    return ControlStateSnapshot(
        simulator_name=simulator_name,
        telemetry_available=control_state.telemetry_available,
        preflight_checks_pass=control_state.preflight_checks_pass,
        preflight_checks_known=control_state.preflight_checks_known,
        armed=control_state.armed,
        offboard_enabled=control_state.offboard_enabled,
        failsafe_active=control_state.failsafe_active,
        ready_for_control=control_state.ready_for_control,
        last_command_error=control_state.last_command_error,
        status_timestamp_us=None if vehicle_status is None else vehicle_status.timestamp_us,
        nav_state=None if vehicle_status is None else vehicle_status.nav_state,
        arming_state=None if vehicle_status is None else vehicle_status.arming_state,
        odometry_timestamp_us=None if odometry is None else odometry.timestamp_us,
        position_ned=None if odometry is None else odometry.position,
        velocity_ned=None if odometry is None else odometry.velocity,
        angular_velocity_frd=None if odometry is None else odometry.angular_velocity,
        last_halted_at_step=last_halted_at_step,
        last_halt_probability=None if last_action is None else last_action.halt_probability,
        last_action_velocity_enu=None if last_action is None else last_action.velocity_xyz,
        last_action_yaw_rate=None if last_action is None else last_action.yaw_rate,
    )


def snapshot_to_json(snapshot: ControlStateSnapshot) -> str:
    """Serializes the supervision snapshot as compact JSON."""

    return json.dumps(asdict(snapshot), sort_keys=True)


def control_event_to_json(event: ControlEvent) -> str:
    """Serializes a control event as compact JSON."""

    return json.dumps(asdict(event), sort_keys=True)


def control_state_events(
    previous: ControlStateSnapshot | None,
    current: ControlStateSnapshot,
) -> list[ControlEvent]:
    """Builds structured control events from snapshot transitions."""

    events: list[ControlEvent] = []

    def _add(event_type: str, severity: str, message: str) -> None:
        events.append(
            ControlEvent(
                event_type=event_type,
                severity=severity,
                message=message,
                simulator_name=current.simulator_name,
                status_timestamp_us=current.status_timestamp_us,
                ready_for_control=current.ready_for_control,
                armed=current.armed,
                offboard_enabled=current.offboard_enabled,
                failsafe_active=current.failsafe_active,
                last_command_error=current.last_command_error,
            )
        )

    if previous is None:
        _add(
            event_type="control_state_initialized",
            severity="info",
            message=(
                "PX4 control state initialized: "
                f"ready={current.ready_for_control} armed={current.armed} "
                f"offboard={current.offboard_enabled} failsafe={current.failsafe_active}"
            ),
        )
        return events

    if previous.ready_for_control != current.ready_for_control:
        _add(
            event_type="ready_for_control_changed",
            severity="info" if current.ready_for_control else "warning",
            message=f"ready_for_control={current.ready_for_control}",
        )
    if previous.armed != current.armed:
        _add(
            event_type="arming_state_changed",
            severity="info" if current.armed else "warning",
            message=f"armed={current.armed}",
        )
    if previous.offboard_enabled != current.offboard_enabled:
        _add(
            event_type="offboard_state_changed",
            severity="info" if current.offboard_enabled else "warning",
            message=f"offboard_enabled={current.offboard_enabled}",
        )
    if previous.failsafe_active != current.failsafe_active:
        _add(
            event_type="failsafe_state_changed",
            severity="error" if current.failsafe_active else "info",
            message=f"failsafe_active={current.failsafe_active}",
        )
    if previous.last_command_error != current.last_command_error:
        if current.last_command_error is not None:
            _add(
                event_type="command_error",
                severity="error",
                message=current.last_command_error,
            )
        else:
            _add(
                event_type="command_error_cleared",
                severity="info",
                message="PX4 command error cleared",
            )
    return events


def control_state_transition_summary(
    previous: ControlStateSnapshot | None,
    current: ControlStateSnapshot,
) -> str | None:
    """Returns a short human-readable summary when the control state changes."""

    if previous is None:
        return (
            "PX4 control state initialized: "
            f"ready={current.ready_for_control} armed={current.armed} "
            f"offboard={current.offboard_enabled} failsafe={current.failsafe_active}"
        )

    changed_fields: list[str] = []
    for field_name in ("ready_for_control", "armed", "offboard_enabled", "failsafe_active", "last_command_error"):
        if getattr(previous, field_name) != getattr(current, field_name):
            changed_fields.append(f"{field_name}={getattr(current, field_name)}")

    if not changed_fields:
        return None
    return "PX4 control state changed: " + " ".join(changed_fields)


def telemetry_log_line(snapshot: ControlStateSnapshot) -> str:
    """Formats a concise PX4 telemetry line for periodic logging."""

    position = snapshot.position_ned or (None, None, None)
    velocity = snapshot.velocity_ned or (None, None, None)
    return (
        "PX4 telemetry "
        f"ready={snapshot.ready_for_control} armed={snapshot.armed} "
        f"offboard={snapshot.offboard_enabled} failsafe={snapshot.failsafe_active} "
        f"pos_ned={position} vel_ned={velocity} "
        f"halt_step={snapshot.last_halted_at_step} halt_prob={snapshot.last_halt_probability}"
    )


def snapshot_to_dict(snapshot: ControlStateSnapshot) -> dict[str, Any]:
    """Exposes the snapshot as a dictionary for tests or custom publishers."""

    return asdict(snapshot)


def event_to_dict(event: ControlEvent) -> dict[str, Any]:
    """Exposes the event as a dictionary for tests or custom publishers."""

    return asdict(event)
